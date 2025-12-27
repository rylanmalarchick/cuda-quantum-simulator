// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Rylan Malarchick

#include "NoiseModel.cuh"

#include "Circuit.hpp"
#include "Constants.hpp"
#include "Gates.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace qsim {

// ============================================================================
// NoiseModel Implementation
// ============================================================================

void NoiseModel::addDepolarizing(const std::vector<int>& qubits, double probability) {
    for (int q : qubits) {
        channels_.emplace_back(NoiseType::Depolarizing, std::vector<int>{q}, probability);
    }
}

void NoiseModel::addAmplitudeDamping(const std::vector<int>& qubits, double gamma) {
    for (int q : qubits) {
        channels_.emplace_back(NoiseType::AmplitudeDamping, std::vector<int>{q}, gamma);
    }
}

void NoiseModel::addPhaseDamping(const std::vector<int>& qubits, double gamma) {
    for (int q : qubits) {
        channels_.emplace_back(NoiseType::PhaseDamping, std::vector<int>{q}, gamma);
    }
}

void NoiseModel::addBitFlip(const std::vector<int>& qubits, double probability) {
    for (int q : qubits) {
        channels_.emplace_back(NoiseType::BitFlip, std::vector<int>{q}, probability);
    }
}

void NoiseModel::addPhaseFlip(const std::vector<int>& qubits, double probability) {
    for (int q : qubits) {
        channels_.emplace_back(NoiseType::PhaseFlip, std::vector<int>{q}, probability);
    }
}

void NoiseModel::addBitPhaseFlip(const std::vector<int>& qubits, double probability) {
    for (int q : qubits) {
        channels_.emplace_back(NoiseType::BitPhaseFlip, std::vector<int>{q}, probability);
    }
}

void NoiseModel::addDepolarizingAll(int num_qubits, double probability) {
    std::vector<int> all_qubits(num_qubits);
    std::iota(all_qubits.begin(), all_qubits.end(), 0);
    addDepolarizing(all_qubits, probability);
}

void NoiseModel::addAmplitudeDampingAll(int num_qubits, double gamma) {
    std::vector<int> all_qubits(num_qubits);
    std::iota(all_qubits.begin(), all_qubits.end(), 0);
    addAmplitudeDamping(all_qubits, gamma);
}

void NoiseModel::addPhaseDampingAll(int num_qubits, double gamma) {
    std::vector<int> all_qubits(num_qubits);
    std::iota(all_qubits.begin(), all_qubits.end(), 0);
    addPhaseDamping(all_qubits, gamma);
}

// Convenience overloads - global noise (empty qubits = applies to all)
void NoiseModel::addDepolarizing(double probability) {
    channels_.emplace_back(NoiseType::Depolarizing, std::vector<int>{}, probability);
}

void NoiseModel::addAmplitudeDamping(double gamma) {
    channels_.emplace_back(NoiseType::AmplitudeDamping, std::vector<int>{}, gamma);
}

void NoiseModel::addPhaseDamping(double gamma) {
    channels_.emplace_back(NoiseType::PhaseDamping, std::vector<int>{}, gamma);
}

void NoiseModel::addBitFlip(double probability) {
    channels_.emplace_back(NoiseType::BitFlip, std::vector<int>{}, probability);
}

void NoiseModel::addPhaseFlip(double probability) {
    channels_.emplace_back(NoiseType::PhaseFlip, std::vector<int>{}, probability);
}

void NoiseModel::addBitPhaseFlip(double probability) {
    channels_.emplace_back(NoiseType::BitPhaseFlip, std::vector<int>{}, probability);
}

// ============================================================================
// CUDA Kernels for Noise Application
// ============================================================================

// Initialize curand states for each thread
__global__ void initRNGKernel(curandState* states, unsigned long seed, size_t n_states) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_states) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Bit flip (X error) kernel
// Apply X gate to pairs where random number < probability
__global__ void applyBitFlipKernel(cuDoubleComplex* state, int n_qubits, int target,
                                    double probability, curandState* rng_states) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    if (idx < n_pairs) {
        // Generate random number
        float r = curand_uniform(&rng_states[idx]);
        
        if (r < probability) {
            // Apply X gate (swap amplitudes)
            size_t mask = (1ULL << target) - 1;
            size_t i0 = (idx & mask) | ((idx & ~mask) << 1);
            size_t i1 = i0 | (1ULL << target);
            
            cuDoubleComplex tmp = state[i0];
            state[i0] = state[i1];
            state[i1] = tmp;
        }
    }
}

// Phase flip (Z error) kernel
__global__ void applyPhaseFlipKernel(cuDoubleComplex* state, int n_qubits, int target,
                                      double probability, curandState* rng_states) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    if (idx < n_pairs) {
        float r = curand_uniform(&rng_states[idx]);
        
        if (r < probability) {
            // Apply Z gate (negate |1⟩ amplitude)
            size_t mask = (1ULL << target) - 1;
            size_t i1 = ((idx & mask) | ((idx & ~mask) << 1)) | (1ULL << target);
            
            state[i1] = make_cuDoubleComplex(-cuCreal(state[i1]), -cuCimag(state[i1]));
        }
    }
}

// Bit-phase flip (Y error) kernel
__global__ void applyBitPhaseFlipKernel(cuDoubleComplex* state, int n_qubits, int target,
                                         double probability, curandState* rng_states) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    if (idx < n_pairs) {
        float r = curand_uniform(&rng_states[idx]);
        
        if (r < probability) {
            // Apply Y gate
            size_t mask = (1ULL << target) - 1;
            size_t i0 = (idx & mask) | ((idx & ~mask) << 1);
            size_t i1 = i0 | (1ULL << target);
            
            cuDoubleComplex a0 = state[i0];
            cuDoubleComplex a1 = state[i1];
            
            // Y = [[0, -i], [i, 0]]
            state[i0] = make_cuDoubleComplex(cuCimag(a1), -cuCreal(a1));
            state[i1] = make_cuDoubleComplex(-cuCimag(a0), cuCreal(a0));
        }
    }
}

// Depolarizing channel kernel
// With probability p, apply X, Y, or Z with equal probability p/3 each
__global__ void applyDepolarizingKernel(cuDoubleComplex* state, int n_qubits, int target,
                                         double probability, curandState* rng_states) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    if (idx < n_pairs) {
        float r1 = curand_uniform(&rng_states[idx]);
        
        if (r1 < probability) {
            // Choose which Pauli to apply
            float r2 = curand_uniform(&rng_states[idx]);
            
            size_t mask = (1ULL << target) - 1;
            size_t i0 = (idx & mask) | ((idx & ~mask) << 1);
            size_t i1 = i0 | (1ULL << target);
            
            cuDoubleComplex a0 = state[i0];
            cuDoubleComplex a1 = state[i1];
            
            if (r2 < 1.0f/3.0f) {
                // X gate
                state[i0] = a1;
                state[i1] = a0;
            } else if (r2 < 2.0f/3.0f) {
                // Y gate
                state[i0] = make_cuDoubleComplex(cuCimag(a1), -cuCreal(a1));
                state[i1] = make_cuDoubleComplex(-cuCimag(a0), cuCreal(a0));
            } else {
                // Z gate
                state[i1] = make_cuDoubleComplex(-cuCreal(a1), -cuCimag(a1));
            }
        }
    }
}

// Amplitude damping kernel (T1 decay)
// Kraus operators: K0 = [[1, 0], [0, sqrt(1-gamma)]], K1 = [[0, sqrt(gamma)], [0, 0]]
// Monte Carlo: with prob |a1|^2 * gamma, collapse to |0⟩
//              otherwise, apply K0 (rescale |1⟩ by sqrt(1-gamma))
__global__ void applyAmplitudeDampingKernel(cuDoubleComplex* state, int n_qubits, int target,
                                             double gamma, curandState* rng_states) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    if (idx < n_pairs) {
        size_t mask = (1ULL << target) - 1;
        size_t i0 = (idx & mask) | ((idx & ~mask) << 1);
        size_t i1 = i0 | (1ULL << target);
        
        cuDoubleComplex a0 = state[i0];
        cuDoubleComplex a1 = state[i1];
        
        // Probability of decay from |1⟩ to |0⟩
        double p1 = cuCreal(a1) * cuCreal(a1) + cuCimag(a1) * cuCimag(a1);  // |a1|^2
        double decay_prob = p1 * gamma;
        
        float r = curand_uniform(&rng_states[idx]);
        
        if (r < decay_prob) {
            // Decay occurred: apply K1, then renormalize
            // K1|ψ⟩ = sqrt(gamma) * a1 |0⟩
            // After normalization: |0⟩
            double norm = sqrt(cuCreal(a0)*cuCreal(a0) + cuCimag(a0)*cuCimag(a0) + 
                              gamma * p1);
            if (norm > 1e-15) {
                // New state: (a0 + sqrt(gamma)*a1) |0⟩ + 0 |1⟩, normalized
                double new_a0_real = (cuCreal(a0) + sqrt(gamma) * cuCreal(a1)) / norm;
                double new_a0_imag = (cuCimag(a0) + sqrt(gamma) * cuCimag(a1)) / norm;
                state[i0] = make_cuDoubleComplex(new_a0_real, new_a0_imag);
                state[i1] = make_cuDoubleComplex(0.0, 0.0);
            }
        } else {
            // No decay: apply K0
            // K0|ψ⟩ = a0|0⟩ + sqrt(1-gamma)*a1|1⟩
            double sqrt_1_minus_gamma = sqrt(1.0 - gamma);
            double norm = sqrt(cuCreal(a0)*cuCreal(a0) + cuCimag(a0)*cuCimag(a0) + 
                              (1.0 - gamma) * p1);
            if (norm > 1e-15) {
                state[i0] = make_cuDoubleComplex(cuCreal(a0) / norm, cuCimag(a0) / norm);
                state[i1] = make_cuDoubleComplex(sqrt_1_minus_gamma * cuCreal(a1) / norm,
                                                  sqrt_1_minus_gamma * cuCimag(a1) / norm);
            }
        }
    }
}

// Phase damping kernel (T2 dephasing - pure dephasing part)
// Kraus operators: K0 = [[1, 0], [0, sqrt(1-gamma)]], K1 = [[0, 0], [0, sqrt(gamma)]]
// This causes loss of coherence without energy loss
__global__ void applyPhaseDampingKernel(cuDoubleComplex* state, int n_qubits, int target,
                                         double gamma, curandState* rng_states) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    if (idx < n_pairs) {
        size_t mask = (1ULL << target) - 1;
        size_t i0 = (idx & mask) | ((idx & ~mask) << 1);
        size_t i1 = i0 | (1ULL << target);
        
        cuDoubleComplex a1 = state[i1];
        double p1 = cuCreal(a1) * cuCreal(a1) + cuCimag(a1) * cuCimag(a1);
        
        float r = curand_uniform(&rng_states[idx]);
        
        // With probability gamma * p1, apply K1 (project |1⟩ component)
        // Otherwise apply K0
        if (r < gamma * p1) {
            // K1 applied: only |1⟩ component survives
            state[i0] = make_cuDoubleComplex(0.0, 0.0);
            // Normalize
            if (p1 > 1e-15) {
                double norm = sqrt(p1);
                state[i1] = make_cuDoubleComplex(cuCreal(a1) / norm, cuCimag(a1) / norm);
            }
        } else {
            // K0 applied: |1⟩ component scaled by sqrt(1-gamma)
            cuDoubleComplex a0 = state[i0];
            double sqrt_1_minus_gamma = sqrt(1.0 - gamma);
            
            double norm_sq = cuCreal(a0)*cuCreal(a0) + cuCimag(a0)*cuCimag(a0) +
                            (1.0 - gamma) * p1;
            if (norm_sq > 1e-15) {
                double norm = sqrt(norm_sq);
                state[i0] = make_cuDoubleComplex(cuCreal(a0) / norm, cuCimag(a0) / norm);
                state[i1] = make_cuDoubleComplex(sqrt_1_minus_gamma * cuCreal(a1) / norm,
                                                  sqrt_1_minus_gamma * cuCimag(a1) / norm);
            }
        }
    }
}

// ============================================================================
// NoisySimulator Implementation
// ============================================================================

NoisySimulator::NoisySimulator(int num_qubits, const NoiseModel& noise_model)
    : num_qubits_(num_qubits)
    , size_(1ULL << num_qubits)
    , d_state_(size_)
    , noise_model_(noise_model)
    , rng_(std::random_device{}())
    , uniform_dist_(0.0, 1.0)
    , d_rng_states_(size_ / 2)
    , rng_initialized_(false)
{
    reset();
    initializeRNG(rng_());
}

NoisySimulator::NoisySimulator(int num_qubits)
    : NoisySimulator(num_qubits, NoiseModel{})
{
}

void NoisySimulator::initializeRNG(unsigned int seed) {
    size_t n_states = size_ / 2;
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = (n_states + threads - 1) / threads;
    
    initRNGKernel<<<blocks, threads>>>(d_rng_states_.get(), seed, n_states);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    rng_initialized_ = true;
}

void NoisySimulator::setSeed(unsigned int seed) {
    rng_.seed(seed);
    initializeRNG(seed);
}

void NoisySimulator::setNoiseModel(const NoiseModel& noise_model) {
    noise_model_ = noise_model;
}

void NoisySimulator::reset() {
    // Initialize to |0...0⟩
    CUDA_CHECK(cudaMemset(d_state_.get(), 0, size_ * sizeof(cuDoubleComplex)));
    
    // Set first element to 1
    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    CUDA_CHECK(cudaMemcpy(d_state_.get(), &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
}

void NoisySimulator::run(const Circuit& circuit) {
    if (circuit.getNumQubits() != num_qubits_) {
        throw std::invalid_argument("Circuit qubit count doesn't match simulator");
    }
    
    for (const auto& gate : circuit.getGates()) {
        applyGate(gate);
        
        // Apply noise after each gate if noise model exists
        if (noise_model_.hasNoise()) {
            applyAllNoiseChannels();
        }
    }
}

void NoisySimulator::applyGate(const GateOp& gate) {
    if (gate.qubits.size() == 1) {
        launchSingleQubitGate(static_cast<int>(gate.type), gate.qubits[0], gate.parameter);
    } else if (gate.qubits.size() == 2) {
        launchTwoQubitGate(static_cast<int>(gate.type), gate.qubits[0], gate.qubits[1], gate.parameter);
    } else if (gate.qubits.size() == 3) {
        launchThreeQubitGate(static_cast<int>(gate.type), gate.qubits[0], gate.qubits[1], gate.qubits[2]);
    }
}

void NoisySimulator::launchSingleQubitGate(int gate_type, int target, double param) {
    size_t n_pairs = 1ULL << (num_qubits_ - 1);
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(n_pairs, threads);
    
    GateType type = static_cast<GateType>(gate_type);
    
    switch (type) {
        case GateType::X:
            applyX<<<blocks, threads>>>(d_state_.get(), num_qubits_, target);
            break;
        case GateType::Y:
            applyY<<<blocks, threads>>>(d_state_.get(), num_qubits_, target);
            break;
        case GateType::Z:
            applyZ<<<blocks, threads>>>(d_state_.get(), num_qubits_, target);
            break;
        case GateType::H:
            applyH<<<blocks, threads>>>(d_state_.get(), num_qubits_, target);
            break;
        case GateType::S:
            applyS<<<blocks, threads>>>(d_state_.get(), num_qubits_, target);
            break;
        case GateType::T:
            applyT<<<blocks, threads>>>(d_state_.get(), num_qubits_, target);
            break;
        case GateType::Sdag:
            applySdag<<<blocks, threads>>>(d_state_.get(), num_qubits_, target);
            break;
        case GateType::Tdag:
            applyTdag<<<blocks, threads>>>(d_state_.get(), num_qubits_, target);
            break;
        case GateType::Rx:
            applyRx<<<blocks, threads>>>(d_state_.get(), num_qubits_, target, param);
            break;
        case GateType::Ry:
            applyRy<<<blocks, threads>>>(d_state_.get(), num_qubits_, target, param);
            break;
        case GateType::Rz:
            applyRz<<<blocks, threads>>>(d_state_.get(), num_qubits_, target, param);
            break;
        default:
            throw std::runtime_error("Unknown single-qubit gate type");
    }
    
    CUDA_CHECK_LAST_ERROR();
}

void NoisySimulator::launchTwoQubitGate(int gate_type, int qubit1, int qubit2, double param) {
    size_t n_states = 1ULL << num_qubits_;
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(n_states, threads);
    
    GateType type = static_cast<GateType>(gate_type);
    
    switch (type) {
        case GateType::CNOT:
            applyCNOT<<<blocks, threads>>>(d_state_.get(), num_qubits_, qubit1, qubit2);
            break;
        case GateType::CZ:
            applyCZ<<<blocks, threads>>>(d_state_.get(), num_qubits_, qubit1, qubit2);
            break;
        case GateType::CRY:
            applyCRY<<<blocks, threads>>>(d_state_.get(), num_qubits_, qubit1, qubit2, param);
            break;
        case GateType::CRZ:
            applyCRZ<<<blocks, threads>>>(d_state_.get(), num_qubits_, qubit1, qubit2, param);
            break;
        case GateType::SWAP:
            applySWAP<<<blocks, threads>>>(d_state_.get(), num_qubits_, qubit1, qubit2);
            break;
        default:
            throw std::runtime_error("Unknown two-qubit gate type");
    }
    
    CUDA_CHECK_LAST_ERROR();
}

void NoisySimulator::launchThreeQubitGate(int gate_type, int qubit1, int qubit2, int qubit3) {
    size_t n_states = 1ULL << num_qubits_;
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(n_states, threads);
    
    GateType type = static_cast<GateType>(gate_type);
    
    switch (type) {
        case GateType::Toffoli:
            applyToffoli<<<blocks, threads>>>(d_state_.get(), num_qubits_, qubit1, qubit2, qubit3);
            break;
        default:
            throw std::runtime_error("Unknown three-qubit gate type");
    }
    
    CUDA_CHECK_LAST_ERROR();
}

void NoisySimulator::applyNoise(const NoiseChannel& channel) {
    for (int qubit : channel.qubits) {
        applyNoiseToQubit(channel.type, qubit, channel.probability);
    }
}

void NoisySimulator::applyNoiseToQubit(NoiseType type, int qubit, double probability) {
    switch (type) {
        case NoiseType::Depolarizing:
            applyDepolarizing(qubit, probability);
            break;
        case NoiseType::AmplitudeDamping:
            applyAmplitudeDamping(qubit, probability);
            break;
        case NoiseType::PhaseDamping:
            applyPhaseDamping(qubit, probability);
            break;
        case NoiseType::BitFlip:
            applyBitFlip(qubit, probability);
            break;
        case NoiseType::PhaseFlip:
            applyPhaseFlip(qubit, probability);
            break;
        case NoiseType::BitPhaseFlip:
            applyBitPhaseFlip(qubit, probability);
            break;
    }
}

void NoisySimulator::applyDepolarizing(int qubit, double p) {
    size_t n_pairs = 1ULL << (num_qubits_ - 1);
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(n_pairs, threads);
    
    applyDepolarizingKernel<<<blocks, threads>>>(d_state_.get(), num_qubits_, qubit, p, d_rng_states_.get());
    CUDA_CHECK_LAST_ERROR();
}

void NoisySimulator::applyAmplitudeDamping(int qubit, double gamma) {
    size_t n_pairs = 1ULL << (num_qubits_ - 1);
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(n_pairs, threads);
    
    applyAmplitudeDampingKernel<<<blocks, threads>>>(d_state_.get(), num_qubits_, qubit, gamma, d_rng_states_.get());
    CUDA_CHECK_LAST_ERROR();
}

void NoisySimulator::applyPhaseDamping(int qubit, double gamma) {
    size_t n_pairs = 1ULL << (num_qubits_ - 1);
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(n_pairs, threads);
    
    applyPhaseDampingKernel<<<blocks, threads>>>(d_state_.get(), num_qubits_, qubit, gamma, d_rng_states_.get());
    CUDA_CHECK_LAST_ERROR();
}

void NoisySimulator::applyBitFlip(int qubit, double p) {
    size_t n_pairs = 1ULL << (num_qubits_ - 1);
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(n_pairs, threads);
    
    applyBitFlipKernel<<<blocks, threads>>>(d_state_.get(), num_qubits_, qubit, p, d_rng_states_.get());
    CUDA_CHECK_LAST_ERROR();
}

void NoisySimulator::applyPhaseFlip(int qubit, double p) {
    size_t n_pairs = 1ULL << (num_qubits_ - 1);
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(n_pairs, threads);
    
    applyPhaseFlipKernel<<<blocks, threads>>>(d_state_.get(), num_qubits_, qubit, p, d_rng_states_.get());
    CUDA_CHECK_LAST_ERROR();
}

void NoisySimulator::applyBitPhaseFlip(int qubit, double p) {
    size_t n_pairs = 1ULL << (num_qubits_ - 1);
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(n_pairs, threads);
    
    applyBitPhaseFlipKernel<<<blocks, threads>>>(d_state_.get(), num_qubits_, qubit, p, d_rng_states_.get());
    CUDA_CHECK_LAST_ERROR();
}

void NoisySimulator::applyAllNoiseChannels() {
    for (const auto& channel : noise_model_.getChannels()) {
        applyNoise(channel);
    }
}

std::vector<std::complex<double>> NoisySimulator::getStateVector() const {
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<std::complex<double>> host_state(size_);
    CUDA_CHECK(cudaMemcpy(host_state.data(), d_state_.get(), 
                          size_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    return host_state;
}

std::vector<double> NoisySimulator::getProbabilities() const {
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto state = getStateVector();
    std::vector<double> probs(size_);
    for (size_t i = 0; i < size_; ++i) {
        probs[i] = std::norm(state[i]);
    }
    return probs;
}

std::vector<int> NoisySimulator::sample(int n_shots) {
    auto probs = getProbabilities();
    
    std::vector<double> cumulative(size_);
    std::partial_sum(probs.begin(), probs.end(), cumulative.begin());
    
    std::vector<int> samples(n_shots);
    for (int i = 0; i < n_shots; ++i) {
        double r = uniform_dist_(rng_);
        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), r);
        samples[i] = static_cast<int>(it - cumulative.begin());
    }
    
    return samples;
}

int NoisySimulator::measureQubit(int qubit) {
    // Get probabilities
    auto probs = getProbabilities();
    
    // Calculate probability of measuring 0
    double p0 = 0.0;
    for (size_t i = 0; i < size_; ++i) {
        if (!((i >> qubit) & 1)) {
            p0 += probs[i];
        }
    }
    
    // Measure
    double r = uniform_dist_(rng_);
    int result = (r < p0) ? 0 : 1;
    
    // Collapse state (would need a kernel for GPU, simplified here)
    auto state = getStateVector();
    double norm = 0.0;
    for (size_t i = 0; i < size_; ++i) {
        if (((i >> qubit) & 1) == static_cast<size_t>(result)) {
            norm += std::norm(state[i]);
        } else {
            state[i] = std::complex<double>(0.0, 0.0);
        }
    }
    norm = std::sqrt(norm);
    for (size_t i = 0; i < size_; ++i) {
        state[i] /= norm;
    }
    
    // Copy back to device
    CUDA_CHECK(cudaMemcpy(d_state_.get(), state.data(), 
                          size_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    
    return result;
}

// ============================================================================
// BatchedSimulator Implementation
// ============================================================================

BatchedSimulator::BatchedSimulator(int num_qubits, int batch_size)
    : num_qubits_(num_qubits)
    , batch_size_(batch_size)
    , state_size_(1ULL << num_qubits)
    , d_states_(batch_size * (1ULL << num_qubits))
    , d_rng_states_(batch_size * ((1ULL << num_qubits) / 2))
    , rng_(std::random_device{}())
{
    reset();
    initializeRNG(rng_());
}

BatchedSimulator::BatchedSimulator(int num_qubits, int batch_size, const NoiseModel& noise_model)
    : BatchedSimulator(num_qubits, batch_size)
{
    noise_model_ = noise_model;
}

void BatchedSimulator::initializeRNG(unsigned int seed) {
    size_t n_states = batch_size_ * (state_size_ / 2);
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = (n_states + threads - 1) / threads;
    
    initRNGKernel<<<blocks, threads>>>(d_rng_states_.get(), seed, n_states);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void BatchedSimulator::setNoiseModel(const NoiseModel& noise_model) {
    noise_model_ = noise_model;
}

void BatchedSimulator::setSeed(unsigned int seed) {
    rng_.seed(seed);
    initializeRNG(seed);
}

// Kernel to initialize all trajectories to |0...0⟩
__global__ void initBatchedZeroKernel(cuDoubleComplex* states, size_t state_size, int batch_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_size = batch_size * state_size;
    
    if (idx < total_size) {
        size_t local_idx = idx % state_size;
        states[idx] = (local_idx == 0) ? make_cuDoubleComplex(1.0, 0.0) 
                                        : make_cuDoubleComplex(0.0, 0.0);
    }
}

void BatchedSimulator::reset() {
    size_t total_size = batch_size_ * state_size_;
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = (total_size + threads - 1) / threads;
    
    initBatchedZeroKernel<<<blocks, threads>>>(d_states_.get(), state_size_, batch_size_);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Batched single-qubit gate kernel
__global__ void applyBatchedSingleQubitGate(cuDoubleComplex* states, int n_qubits, int target,
                                             int batch_size, int gate_type, double param) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = 1ULL << (n_qubits - 1);
    size_t state_size = 1ULL << n_qubits;
    size_t total_pairs = batch_size * n_pairs;
    
    if (idx < total_pairs) {
        int traj = idx / n_pairs;
        size_t pair_idx = idx % n_pairs;
        
        size_t mask = (1ULL << target) - 1;
        size_t i0 = (pair_idx & mask) | ((pair_idx & ~mask) << 1);
        size_t i1 = i0 | (1ULL << target);
        
        // Offset for this trajectory
        size_t offset = traj * state_size;
        
        cuDoubleComplex a0 = states[offset + i0];
        cuDoubleComplex a1 = states[offset + i1];
        
        // Apply gate based on type
        // gate_type: 0=X, 1=Y, 2=Z, 3=H, etc.
        const double inv_sqrt2 = 0.7071067811865476;
        
        switch (gate_type) {
            case 0: // X
                states[offset + i0] = a1;
                states[offset + i1] = a0;
                break;
            case 1: // Y
                states[offset + i0] = make_cuDoubleComplex(cuCimag(a1), -cuCreal(a1));
                states[offset + i1] = make_cuDoubleComplex(-cuCimag(a0), cuCreal(a0));
                break;
            case 2: // Z
                states[offset + i1] = make_cuDoubleComplex(-cuCreal(a1), -cuCimag(a1));
                break;
            case 3: // H
                states[offset + i0] = make_cuDoubleComplex(
                    (cuCreal(a0) + cuCreal(a1)) * inv_sqrt2,
                    (cuCimag(a0) + cuCimag(a1)) * inv_sqrt2);
                states[offset + i1] = make_cuDoubleComplex(
                    (cuCreal(a0) - cuCreal(a1)) * inv_sqrt2,
                    (cuCimag(a0) - cuCimag(a1)) * inv_sqrt2);
                break;
            // Add more gates as needed
        }
    }
}

void BatchedSimulator::launchBatchedSingleQubitGate(int gate_type, int target, double param) {
    size_t n_pairs = 1ULL << (num_qubits_ - 1);
    size_t total_pairs = batch_size_ * n_pairs;
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = (total_pairs + threads - 1) / threads;
    
    applyBatchedSingleQubitGate<<<blocks, threads>>>(d_states_.get(), num_qubits_, target,
                                                      batch_size_, gate_type, param);
    CUDA_CHECK_LAST_ERROR();
}

// Batched CNOT kernel
__global__ void applyBatchedCNOT(cuDoubleComplex* states, int n_qubits, 
                                  int control, int target, int batch_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t state_size = 1ULL << n_qubits;
    size_t total_states = batch_size * state_size;
    
    if (idx < total_states) {
        int traj = idx / state_size;
        size_t state_idx = idx % state_size;
        
        bool control_is_1 = (state_idx >> control) & 1;
        bool target_is_0 = !((state_idx >> target) & 1);
        
        if (control_is_1 && target_is_0) {
            size_t offset = traj * state_size;
            size_t partner = state_idx ^ (1ULL << target);
            
            cuDoubleComplex tmp = states[offset + state_idx];
            states[offset + state_idx] = states[offset + partner];
            states[offset + partner] = tmp;
        }
    }
}

void BatchedSimulator::launchBatchedTwoQubitGate(int gate_type, int qubit1, int qubit2, double /*param*/) {
    size_t total_states = batch_size_ * state_size_;
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = (total_states + threads - 1) / threads;
    
    // For now, only CNOT implemented
    if (gate_type == static_cast<int>(GateType::CNOT)) {
        applyBatchedCNOT<<<blocks, threads>>>(d_states_.get(), num_qubits_, qubit1, qubit2, batch_size_);
    }
    CUDA_CHECK_LAST_ERROR();
}

void BatchedSimulator::run(const Circuit& circuit) {
    if (circuit.getNumQubits() != num_qubits_) {
        throw std::invalid_argument("Circuit qubit count doesn't match simulator");
    }
    
    for (const auto& gate : circuit.getGates()) {
        if (gate.qubits.size() == 1) {
            launchBatchedSingleQubitGate(static_cast<int>(gate.type), gate.qubits[0], gate.parameter);
        } else if (gate.qubits.size() == 2) {
            launchBatchedTwoQubitGate(static_cast<int>(gate.type), gate.qubits[0], gate.qubits[1], gate.parameter);
        }
        
        if (noise_model_.hasNoise()) {
            applyBatchedNoise();
        }
    }
}

// Batched noise application kernel
__global__ void applyBatchedDepolarizingKernel(cuDoubleComplex* states, int n_qubits, int target,
                                                double probability, int batch_size,
                                                curandState* rng_states) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = 1ULL << (n_qubits - 1);
    size_t state_size = 1ULL << n_qubits;
    size_t total_pairs = batch_size * n_pairs;
    
    if (idx < total_pairs) {
        int traj = idx / n_pairs;
        size_t pair_idx = idx % n_pairs;
        
        float r1 = curand_uniform(&rng_states[idx]);
        
        if (r1 < probability) {
            float r2 = curand_uniform(&rng_states[idx]);
            
            size_t mask = (1ULL << target) - 1;
            size_t i0 = (pair_idx & mask) | ((pair_idx & ~mask) << 1);
            size_t i1 = i0 | (1ULL << target);
            
            size_t offset = traj * state_size;
            cuDoubleComplex a0 = states[offset + i0];
            cuDoubleComplex a1 = states[offset + i1];
            
            if (r2 < 1.0f/3.0f) {
                // X
                states[offset + i0] = a1;
                states[offset + i1] = a0;
            } else if (r2 < 2.0f/3.0f) {
                // Y
                states[offset + i0] = make_cuDoubleComplex(cuCimag(a1), -cuCreal(a1));
                states[offset + i1] = make_cuDoubleComplex(-cuCimag(a0), cuCreal(a0));
            } else {
                // Z
                states[offset + i1] = make_cuDoubleComplex(-cuCreal(a1), -cuCimag(a1));
            }
        }
    }
}

void BatchedSimulator::applyBatchedNoise() {
    for (const auto& channel : noise_model_.getChannels()) {
        if (channel.type == NoiseType::Depolarizing) {
            for (int qubit : channel.qubits) {
                size_t n_pairs = 1ULL << (num_qubits_ - 1);
                size_t total_pairs = batch_size_ * n_pairs;
                int threads = cuda_config::DEFAULT_BLOCK_SIZE;
                int blocks = (total_pairs + threads - 1) / threads;
                
                applyBatchedDepolarizingKernel<<<blocks, threads>>>(
                    d_states_.get(), num_qubits_, qubit, channel.probability,
                    batch_size_, d_rng_states_.get());
            }
        }
        // Add other noise types as needed
    }
    CUDA_CHECK_LAST_ERROR();
}

std::vector<double> BatchedSimulator::getAverageProbabilities() const {
    CUDA_CHECK(cudaDeviceSynchronize());
    
    size_t total_size = batch_size_ * state_size_;
    std::vector<cuDoubleComplex> all_states(total_size);
    CUDA_CHECK(cudaMemcpy(all_states.data(), d_states_.get(), 
                          total_size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    
    std::vector<double> avg_probs(state_size_, 0.0);
    
    for (int traj = 0; traj < batch_size_; ++traj) {
        size_t offset = traj * state_size_;
        for (size_t i = 0; i < state_size_; ++i) {
            double real = cuCreal(all_states[offset + i]);
            double imag = cuCimag(all_states[offset + i]);
            avg_probs[i] += (real * real + imag * imag) / batch_size_;
        }
    }
    
    return avg_probs;
}

std::vector<double> BatchedSimulator::getProbabilities(int trajectory_idx) const {
    if (trajectory_idx < 0 || trajectory_idx >= batch_size_) {
        throw std::out_of_range("Invalid trajectory index");
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<cuDoubleComplex> state(state_size_);
    size_t offset = trajectory_idx * state_size_;
    CUDA_CHECK(cudaMemcpy(state.data(), d_states_.get() + offset,
                          state_size_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    
    std::vector<double> probs(state_size_);
    for (size_t i = 0; i < state_size_; ++i) {
        double real = cuCreal(state[i]);
        double imag = cuCimag(state[i]);
        probs[i] = real * real + imag * imag;
    }
    
    return probs;
}

std::vector<std::vector<int>> BatchedSimulator::sample(int n_shots) {
    std::vector<std::vector<int>> all_samples(n_shots, std::vector<int>(batch_size_));
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (int traj = 0; traj < batch_size_; ++traj) {
        auto probs = getProbabilities(traj);
        
        std::vector<double> cumulative(state_size_);
        std::partial_sum(probs.begin(), probs.end(), cumulative.begin());
        
        for (int shot = 0; shot < n_shots; ++shot) {
            double r = dist(rng_);
            auto it = std::lower_bound(cumulative.begin(), cumulative.end(), r);
            all_samples[shot][traj] = static_cast<int>(it - cumulative.begin());
        }
    }
    
    return all_samples;
}

std::vector<int> BatchedSimulator::getHistogram(int n_shots) {
    std::vector<int> histogram(state_size_, 0);
    
    auto samples = sample(n_shots);
    for (const auto& shot_samples : samples) {
        for (int outcome : shot_samples) {
            if (outcome < static_cast<int>(state_size_)) {
                histogram[outcome]++;
            }
        }
    }
    
    return histogram;
}

} // namespace qsim
