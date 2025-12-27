// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Rylan Malarchick

#include "Simulator.hpp"

#include "Constants.hpp"
#include "Gates.cuh"

#include <cuda_runtime.h>
#include <curand.h>

#include <algorithm>
#include <numeric>
#include <random>

namespace qsim {

// ============================================================================
// GPU Simulator
// ============================================================================

Simulator::Simulator(int num_qubits) : state_(num_qubits) {}

void Simulator::reset() {
    state_.initializeZero();
}

void Simulator::run(const Circuit& circuit) {
    if (circuit.getNumQubits() != state_.getNumQubits()) {
        throw std::invalid_argument("Circuit qubit count doesn't match simulator");
    }
    
    for (const auto& gate : circuit.getGates()) {
        applyGate(gate);
    }
}

void Simulator::applyGate(const GateOp& gate) {
    if (gate.qubits.size() == 1) {
        launchSingleQubitGate(gate.type, gate.qubits[0], gate.parameter);
    } else if (gate.qubits.size() == 2) {
        launchTwoQubitGate(gate.type, gate.qubits[0], gate.qubits[1], gate.parameter);
    } else if (gate.qubits.size() == 3) {
        launchThreeQubitGate(gate.type, gate.qubits[0], gate.qubits[1], gate.qubits[2]);
    }
}

void Simulator::launchSingleQubitGate(GateType type, int target, double param) {
    int n_qubits = state_.getNumQubits();
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(n_pairs, threads);
    
    cuDoubleComplex* d_state = state_.devicePtr();
    
    switch (type) {
        case GateType::X:
            applyX<<<blocks, threads>>>(d_state, n_qubits, target);
            break;
        case GateType::Y:
            applyY<<<blocks, threads>>>(d_state, n_qubits, target);
            break;
        case GateType::Z:
            applyZ<<<blocks, threads>>>(d_state, n_qubits, target);
            break;
        case GateType::H:
            applyH<<<blocks, threads>>>(d_state, n_qubits, target);
            break;
        case GateType::S:
            applyS<<<blocks, threads>>>(d_state, n_qubits, target);
            break;
        case GateType::T:
            applyT<<<blocks, threads>>>(d_state, n_qubits, target);
            break;
        case GateType::Sdag:
            applySdag<<<blocks, threads>>>(d_state, n_qubits, target);
            break;
        case GateType::Tdag:
            applyTdag<<<blocks, threads>>>(d_state, n_qubits, target);
            break;
        case GateType::Rx:
            applyRx<<<blocks, threads>>>(d_state, n_qubits, target, param);
            break;
        case GateType::Ry:
            applyRy<<<blocks, threads>>>(d_state, n_qubits, target, param);
            break;
        case GateType::Rz:
            applyRz<<<blocks, threads>>>(d_state, n_qubits, target, param);
            break;
        default:
            throw std::runtime_error("Unknown single-qubit gate type");
    }
    
    CUDA_CHECK_LAST_ERROR();
    // Note: No cudaDeviceSynchronize() here - CUDA streams serialize operations
    // automatically. Sync only when reading results (getStateVector, sample, etc.)
}

void Simulator::launchTwoQubitGate(GateType type, int qubit1, int qubit2, double param) {
    int n_qubits = state_.getNumQubits();
    size_t n_states = 1ULL << n_qubits;
    
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(n_states, threads);
    
    cuDoubleComplex* d_state = state_.devicePtr();
    
    switch (type) {
        case GateType::CNOT:
            applyCNOT<<<blocks, threads>>>(d_state, n_qubits, qubit1, qubit2);
            break;
        case GateType::CZ:
            applyCZ<<<blocks, threads>>>(d_state, n_qubits, qubit1, qubit2);
            break;
        case GateType::CRY:
            applyCRY<<<blocks, threads>>>(d_state, n_qubits, qubit1, qubit2, param);
            break;
        case GateType::CRZ:
            applyCRZ<<<blocks, threads>>>(d_state, n_qubits, qubit1, qubit2, param);
            break;
        case GateType::SWAP:
            applySWAP<<<blocks, threads>>>(d_state, n_qubits, qubit1, qubit2);
            break;
        default:
            throw std::runtime_error("Unknown two-qubit gate type");
    }
    
    CUDA_CHECK_LAST_ERROR();
    // Note: No cudaDeviceSynchronize() here - CUDA streams serialize operations
    // automatically. Sync only when reading results (getStateVector, sample, etc.)
}

void Simulator::launchThreeQubitGate(GateType type, int qubit1, int qubit2, int qubit3) {
    int n_qubits = state_.getNumQubits();
    size_t n_states = 1ULL << n_qubits;
    
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(n_states, threads);
    
    cuDoubleComplex* d_state = state_.devicePtr();
    
    switch (type) {
        case GateType::Toffoli:
            applyToffoli<<<blocks, threads>>>(d_state, n_qubits, qubit1, qubit2, qubit3);
            break;
        default:
            throw std::runtime_error("Unknown three-qubit gate type");
    }
    
    CUDA_CHECK_LAST_ERROR();
    // Note: No cudaDeviceSynchronize() here - CUDA streams serialize operations
    // automatically. Sync only when reading results (getStateVector, sample, etc.)
}

std::vector<std::complex<double>> Simulator::getStateVector() const {
    return state_.toHost();
}

std::vector<double> Simulator::getProbabilities() const {
    return state_.getProbabilities();
}

std::vector<int> Simulator::sample(int n_shots) {
    auto probs = getProbabilities();
    size_t n_states = probs.size();
    
    // Build cumulative distribution
    std::vector<double> cumulative(n_states);
    std::partial_sum(probs.begin(), probs.end(), cumulative.begin());
    
    // Sample
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    std::vector<int> samples(n_shots);
    for (int i = 0; i < n_shots; ++i) {
        double r = dist(rng);
        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), r);
        samples[i] = static_cast<int>(it - cumulative.begin());
    }
    
    return samples;
}

int Simulator::measureQubit(int qubit) {
    return state_.measure(qubit);
}

// ============================================================================
// CPU Simulator (for benchmarking comparison)
// ============================================================================

CPUSimulator::CPUSimulator(int num_qubits) 
    : num_qubits_(num_qubits)
    , size_(1ULL << num_qubits)
    , state_(size_, std::complex<double>(0.0, 0.0))
{
    reset();
}

void CPUSimulator::reset() {
    std::fill(state_.begin(), state_.end(), std::complex<double>(0.0, 0.0));
    state_[0] = std::complex<double>(1.0, 0.0);
}

void CPUSimulator::run(const Circuit& circuit) {
    for (const auto& gate : circuit.getGates()) {
        applyGate(gate);
    }
}

void CPUSimulator::applyGate(const GateOp& gate) {
    if (gate.qubits.size() == 1) {
        applySingleQubitGate(gate.type, gate.qubits[0], gate.parameter);
    } else if (gate.qubits.size() == 2) {
        applyTwoQubitGate(gate.type, gate.qubits[0], gate.qubits[1]);
    }
}

void CPUSimulator::applySingleQubitGate(GateType type, int target, double param) {
    size_t n_pairs = 1ULL << (num_qubits_ - 1);
    
    for (size_t idx = 0; idx < n_pairs; ++idx) {
        // Calculate pair indices
        size_t mask = (1ULL << target) - 1;
        size_t i0 = (idx & mask) | ((idx & ~mask) << 1);
        size_t i1 = i0 | (1ULL << target);
        
        std::complex<double> a0 = state_[i0];
        std::complex<double> a1 = state_[i1];
        
        switch (type) {
            case GateType::X:
                state_[i0] = a1;
                state_[i1] = a0;
                break;
            case GateType::Y:
                state_[i0] = std::complex<double>(0, -1) * a1;
                state_[i1] = std::complex<double>(0, 1) * a0;
                break;
            case GateType::Z:
                state_[i1] = -a1;
                break;
            case GateType::H:
                state_[i0] = (a0 + a1) * constants::INV_SQRT2;
                state_[i1] = (a0 - a1) * constants::INV_SQRT2;
                break;
            case GateType::S:
                state_[i1] = std::complex<double>(0, 1) * a1;
                break;
            case GateType::T:
                state_[i1] = std::complex<double>(constants::INV_SQRT2, constants::INV_SQRT2) * a1;
                break;
            case GateType::Sdag:
                state_[i1] = std::complex<double>(0, -1) * a1;
                break;
            case GateType::Tdag:
                state_[i1] = std::complex<double>(constants::INV_SQRT2, -constants::INV_SQRT2) * a1;
                break;
            case GateType::Rx: {
                double c = std::cos(param / 2.0);
                double s = std::sin(param / 2.0);
                state_[i0] = c * a0 - std::complex<double>(0, s) * a1;
                state_[i1] = -std::complex<double>(0, s) * a0 + c * a1;
                break;
            }
            case GateType::Ry: {
                double c = std::cos(param / 2.0);
                double s = std::sin(param / 2.0);
                state_[i0] = c * a0 - s * a1;
                state_[i1] = s * a0 + c * a1;
                break;
            }
            case GateType::Rz: {
                double c = std::cos(param / 2.0);
                double s = std::sin(param / 2.0);
                state_[i0] = std::complex<double>(c, -s) * a0;
                state_[i1] = std::complex<double>(c, s) * a1;
                break;
            }
            default:
                break;
        }
    }
}

void CPUSimulator::applyTwoQubitGate(GateType type, int qubit1, int qubit2) {
    for (size_t idx = 0; idx < size_; ++idx) {
        bool q1_bit = (idx >> qubit1) & 1;
        bool q2_bit = (idx >> qubit2) & 1;
        
        switch (type) {
            case GateType::CNOT:
                // qubit1 = control, qubit2 = target
                if (q1_bit && !q2_bit) {
                    size_t partner = idx ^ (1ULL << qubit2);
                    std::swap(state_[idx], state_[partner]);
                }
                break;
            case GateType::CZ:
                if (q1_bit && q2_bit) {
                    state_[idx] = -state_[idx];
                }
                break;
            case GateType::SWAP:
                if (!q1_bit && q2_bit) {
                    size_t partner = idx ^ (1ULL << qubit1) ^ (1ULL << qubit2);
                    std::swap(state_[idx], state_[partner]);
                }
                break;
            default:
                break;
        }
    }
}

std::vector<double> CPUSimulator::getProbabilities() const {
    std::vector<double> probs(size_);
    for (size_t i = 0; i < size_; ++i) {
        probs[i] = std::norm(state_[i]);
    }
    return probs;
}

std::vector<int> CPUSimulator::sample(int n_shots) {
    auto probs = getProbabilities();
    
    std::vector<double> cumulative(size_);
    std::partial_sum(probs.begin(), probs.end(), cumulative.begin());
    
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    std::vector<int> samples(n_shots);
    for (int i = 0; i < n_shots; ++i) {
        double r = dist(rng);
        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), r);
        samples[i] = static_cast<int>(it - cumulative.begin());
    }
    
    return samples;
}

} // namespace qsim
