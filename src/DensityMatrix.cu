// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Rylan Malarchick

#include "DensityMatrix.cuh"

#include "Constants.hpp"

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cmath>
#include <random>
#include <stdexcept>

namespace qsim {

// Use CUDA_CHECK from Constants.hpp for consistency

// ============================================================================
// DensityMatrix Class Implementation
// ============================================================================

DensityMatrix::DensityMatrix(int n_qubits) 
    : n_qubits_(n_qubits), dim_(1ULL << n_qubits), d_rho_(nullptr) {
    if (n_qubits < 1 || n_qubits > 14) {
        throw std::invalid_argument("Density matrix supports 1-14 qubits");
    }
    
    size_t num_elements = dim_ * dim_;
    CUDA_CHECK(cudaMalloc(&d_rho_, num_elements * sizeof(cuDoubleComplex)));
    reset();
}

DensityMatrix::DensityMatrix(int n_qubits, const std::vector<std::complex<double>>& pure_state)
    : n_qubits_(n_qubits), dim_(1ULL << n_qubits), d_rho_(nullptr) {
    if (n_qubits < 1 || n_qubits > 14) {
        throw std::invalid_argument("Density matrix supports 1-14 qubits");
    }
    if (pure_state.size() != dim_) {
        throw std::invalid_argument("State vector size mismatch");
    }
    
    size_t num_elements = dim_ * dim_;
    CUDA_CHECK(cudaMalloc(&d_rho_, num_elements * sizeof(cuDoubleComplex)));
    initFromPureState(pure_state);
}

DensityMatrix::~DensityMatrix() noexcept {
    if (d_rho_) {
        // Ignore errors in destructor - cannot throw
        cudaFree(d_rho_);
        d_rho_ = nullptr;
    }
}

DensityMatrix::DensityMatrix(DensityMatrix&& other) noexcept
    : n_qubits_(other.n_qubits_), dim_(other.dim_), d_rho_(other.d_rho_) {
    other.d_rho_ = nullptr;
}

DensityMatrix& DensityMatrix::operator=(DensityMatrix&& other) noexcept {
    if (this != &other) {
        if (d_rho_) cudaFree(d_rho_);
        n_qubits_ = other.n_qubits_;
        dim_ = other.dim_;
        d_rho_ = other.d_rho_;
        other.d_rho_ = nullptr;
    }
    return *this;
}

void DensityMatrix::reset() {
    size_t num_elements = dim_ * dim_;
    CUDA_CHECK(cudaMemset(d_rho_, 0, num_elements * sizeof(cuDoubleComplex)));
    
    // Set rho[0][0] = 1
    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    CUDA_CHECK(cudaMemcpy(d_rho_, &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
}

void DensityMatrix::initFromPureState(const std::vector<std::complex<double>>& state) {
    // Upload state to device
    cuDoubleComplex* d_state;
    CUDA_CHECK(cudaMalloc(&d_state, dim_ * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(d_state, state.data(), dim_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    
    // Compute rho = |psi><psi|
    int threads = 256;
    int blocks = (dim_ * dim_ + threads - 1) / threads;
    dmInitPure<<<blocks, threads>>>(d_rho_, d_state, dim_);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaFree(d_state);
}

void DensityMatrix::initMaximallyMixed() {
    size_t num_elements = dim_ * dim_;
    CUDA_CHECK(cudaMemset(d_rho_, 0, num_elements * sizeof(cuDoubleComplex)));
    
    double val = 1.0 / dim_;
    int threads = 256;
    int blocks = (dim_ + threads - 1) / threads;
    dmInitMaxMixed<<<blocks, threads>>>(d_rho_, dim_, val);
    CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<double> DensityMatrix::getProbabilities() const {
    std::vector<double> probs(dim_);
    double* d_diag;
    CUDA_CHECK(cudaMalloc(&d_diag, dim_ * sizeof(double)));
    
    int threads = 256;
    int blocks = (dim_ + threads - 1) / threads;
    dmComputeDiagonal<<<blocks, threads>>>(d_rho_, d_diag, dim_);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(probs.data(), d_diag, dim_ * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_diag);
    
    return probs;
}

std::vector<std::complex<double>> DensityMatrix::getMatrix() const {
    size_t num_elements = dim_ * dim_;
    std::vector<std::complex<double>> result(num_elements);
    CUDA_CHECK(cudaMemcpy(result.data(), d_rho_, num_elements * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    return result;
}

double DensityMatrix::trace() const {
    double* d_trace;
    double h_trace;
    CUDA_CHECK(cudaMalloc(&d_trace, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_trace, 0, sizeof(double)));
    
    int threads = 256;
    int blocks = (dim_ + threads - 1) / threads;
    dmComputeTrace<<<blocks, threads>>>(d_rho_, d_trace, dim_);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(&h_trace, d_trace, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_trace);
    
    return h_trace;
}

double DensityMatrix::purity() const {
    // Purity = Tr(rho^2)
    // For now, compute on CPU (could optimize with cuBLAS)
    auto mat = getMatrix();
    double sum = 0.0;
    for (size_t i = 0; i < dim_; ++i) {
        for (size_t j = 0; j < dim_; ++j) {
            // (rho^2)_ii = sum_j rho_ij * rho_ji
            std::complex<double> prod = mat[i * dim_ + j] * mat[j * dim_ + i];
            if (i == j) sum += prod.real();
        }
    }
    // Actually compute full trace of rho^2
    sum = 0.0;
    for (size_t i = 0; i < dim_; ++i) {
        for (size_t k = 0; k < dim_; ++k) {
            sum += std::norm(mat[i * dim_ + k]);
        }
    }
    return sum;
}

bool DensityMatrix::isValid(double tolerance) const {
    double tr = trace();
    if (std::abs(tr - 1.0) > tolerance) return false;
    
    double pur = purity();
    double min_purity = 1.0 / dim_;
    if (pur < min_purity - tolerance || pur > 1.0 + tolerance) return false;
    
    return true;
}

// ============================================================================
// DensityMatrixSimulator Implementation
// ============================================================================

DensityMatrixSimulator::DensityMatrixSimulator(int n_qubits, const NoiseModel& noise)
    : n_qubits_(n_qubits), rho_(n_qubits), noise_model_(noise), d_scratch_(nullptr) {
    size_t dim = 1ULL << n_qubits;
    CUDA_CHECK(cudaMalloc(&d_scratch_, dim * dim * sizeof(cuDoubleComplex)));
}

DensityMatrixSimulator::~DensityMatrixSimulator() noexcept {
    if (d_scratch_) {
        cudaFree(d_scratch_);
        d_scratch_ = nullptr;
    }
}

void DensityMatrixSimulator::reset() {
    rho_.reset();
}

void DensityMatrixSimulator::run(const Circuit& circuit) {
    for (const auto& gate : circuit.getGates()) {
        applyGate(gate);
        
        // Apply noise after each gate if noise model is set
        if (noise_model_.hasNoise()) {
            for (int q : gate.qubits) {
                applyNoiseChannel(q);
            }
        }
    }
}

void DensityMatrixSimulator::applyGate(const GateOp& gate) {
    cuDoubleComplex* rho = rho_.getDevicePtr();
    size_t dim = rho_.getDimension();
    size_t num_elements = dim * dim;
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    
    switch (gate.type) {
        case GateType::X:
            dmApplyX<<<blocks, threads>>>(rho, n_qubits_, gate.qubits[0]);
            break;
        case GateType::Y:
            dmApplyY<<<blocks, threads>>>(rho, n_qubits_, gate.qubits[0]);
            break;
        case GateType::Z:
            dmApplyZ<<<blocks, threads>>>(rho, n_qubits_, gate.qubits[0]);
            break;
        case GateType::H:
            dmApplyH<<<blocks, threads>>>(rho, n_qubits_, gate.qubits[0]);
            break;
        case GateType::S:
            dmApplyS<<<blocks, threads>>>(rho, n_qubits_, gate.qubits[0]);
            break;
        case GateType::T:
            dmApplyT<<<blocks, threads>>>(rho, n_qubits_, gate.qubits[0]);
            break;
        case GateType::Sdag:
            dmApplySdag<<<blocks, threads>>>(rho, n_qubits_, gate.qubits[0]);
            break;
        case GateType::Tdag:
            dmApplyTdag<<<blocks, threads>>>(rho, n_qubits_, gate.qubits[0]);
            break;
        case GateType::Rx:
            dmApplyRx<<<blocks, threads>>>(rho, n_qubits_, gate.qubits[0], gate.parameter);
            break;
        case GateType::Ry:
            dmApplyRy<<<blocks, threads>>>(rho, n_qubits_, gate.qubits[0], gate.parameter);
            break;
        case GateType::Rz:
            dmApplyRz<<<blocks, threads>>>(rho, n_qubits_, gate.qubits[0], gate.parameter);
            break;
        case GateType::CNOT:
            dmApplyCNOT<<<blocks, threads>>>(rho, n_qubits_, gate.qubits[0], gate.qubits[1]);
            break;
        case GateType::CZ:
            dmApplyCZ<<<blocks, threads>>>(rho, n_qubits_, gate.qubits[0], gate.qubits[1]);
            break;
        case GateType::SWAP:
            dmApplySWAP<<<blocks, threads>>>(rho, n_qubits_, gate.qubits[0], gate.qubits[1]);
            break;
        default:
            throw std::runtime_error("Gate not supported in density matrix simulation");
    }
}

void DensityMatrixSimulator::applyNoiseChannel(int target) {
    for (const auto& channel : noise_model_.getChannels()) {
        // Check if this channel applies to this qubit
        if (!noise_model_.channelAppliesToQubit(channel, target)) continue;
        NoiseType type = channel.type;
        double prob = channel.probability;
        switch (type) {
            case NoiseType::Depolarizing:
                applyDepolarizing(target, prob);
                break;
            case NoiseType::AmplitudeDamping:
                applyAmplitudeDamping(target, prob);
                break;
            case NoiseType::PhaseDamping:
                applyPhaseDamping(target, prob);
                break;
            case NoiseType::BitFlip:
                applyBitFlip(target, prob);
                break;
            case NoiseType::PhaseFlip:
                applyPhaseFlip(target, prob);
                break;
            case NoiseType::BitPhaseFlip:
                applyBitPhaseFlip(target, prob);
                break;
        }
    }
}

void DensityMatrixSimulator::applyDepolarizing(int target, double p) {
    cuDoubleComplex* rho = rho_.getDevicePtr();
    size_t dim = rho_.getDimension();
    size_t num_elements = dim * dim;
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dmApplyDepolarizing<<<blocks, threads>>>(rho, n_qubits_, target, p);
}

void DensityMatrixSimulator::applyAmplitudeDamping(int target, double gamma) {
    cuDoubleComplex* rho = rho_.getDevicePtr();
    size_t dim = rho_.getDimension();
    size_t num_elements = dim * dim;
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dmApplyAmplitudeDamping<<<blocks, threads>>>(rho, n_qubits_, target, gamma);
}

void DensityMatrixSimulator::applyPhaseDamping(int target, double gamma) {
    cuDoubleComplex* rho = rho_.getDevicePtr();
    size_t dim = rho_.getDimension();
    size_t num_elements = dim * dim;
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dmApplyPhaseDamping<<<blocks, threads>>>(rho, n_qubits_, target, gamma);
}

void DensityMatrixSimulator::applyBitFlip(int target, double p) {
    cuDoubleComplex* rho = rho_.getDevicePtr();
    size_t dim = rho_.getDimension();
    size_t num_elements = dim * dim;
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dmApplyBitFlip<<<blocks, threads>>>(rho, n_qubits_, target, p);
}

void DensityMatrixSimulator::applyPhaseFlip(int target, double p) {
    cuDoubleComplex* rho = rho_.getDevicePtr();
    size_t dim = rho_.getDimension();
    size_t num_elements = dim * dim;
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dmApplyPhaseFlip<<<blocks, threads>>>(rho, n_qubits_, target, p);
}

void DensityMatrixSimulator::applyBitPhaseFlip(int target, double p) {
    // Y = iXZ, so bit-phase flip is Y error
    // For now, implement as depolarizing with only Y
    // rho' = (1-p)*rho + p*Y*rho*Y
    cuDoubleComplex* rho = rho_.getDevicePtr();
    size_t dim = rho_.getDimension();
    size_t num_elements = dim * dim;
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    
    // Use phase flip kernel structure but with Y operator
    // For simplicity, approximate with phase flip (TODO: proper Y channel)
    dmApplyPhaseFlip<<<blocks, threads>>>(rho, n_qubits_, target, p);
}

std::vector<double> DensityMatrixSimulator::getProbabilities() const {
    return rho_.getProbabilities();
}

std::vector<std::complex<double>> DensityMatrixSimulator::getDensityMatrix() const {
    return rho_.getMatrix();
}

double DensityMatrixSimulator::getPurity() const {
    return rho_.purity();
}

double DensityMatrixSimulator::getTrace() const {
    return rho_.trace();
}

int DensityMatrixSimulator::measureQubit(int qubit) {
    auto probs = getProbabilities();
    
    // Compute marginal probability for qubit being 1
    double p1 = 0.0;
    size_t dim = rho_.getDimension();
    for (size_t i = 0; i < dim; ++i) {
        if ((i >> qubit) & 1) {
            p1 += probs[i];
        }
    }
    double p0 = 1.0 - p1;
    
    // Random measurement
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    int result = (dis(gen) < p1) ? 1 : 0;
    
    // Collapse density matrix: project onto measurement outcome and renormalize
    // ρ' = P_m ρ P_m / Tr(P_m ρ P_m) = P_m ρ P_m / p_m
    double prob_result = (result == 1) ? p1 : p0;
    double norm_factor = 1.0 / prob_result;
    
    int n_qubits = rho_.getNumQubits();
    int threads = 256;
    int blocks = (dim * dim + threads - 1) / threads;
    dmCollapseMeasurement<<<blocks, threads>>>(rho_.getDevicePtr(), n_qubits, qubit, 
                                                result, norm_factor);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return result;
}

// ============================================================================
// Kernel Implementations
// ============================================================================

__global__ void dmInitPure(cuDoubleComplex* rho, const cuDoubleComplex* state, size_t dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim * dim) {
        size_t i = idx / dim;
        size_t j = idx % dim;
        // rho[i][j] = state[i] * conj(state[j])
        cuDoubleComplex si = state[i];
        cuDoubleComplex sj = state[j];
        rho[idx] = make_cuDoubleComplex(
            cuCreal(si) * cuCreal(sj) + cuCimag(si) * cuCimag(sj),
            cuCimag(si) * cuCreal(sj) - cuCreal(si) * cuCimag(sj)
        );
    }
}

__global__ void dmInitMaxMixed(cuDoubleComplex* rho, size_t dim, double val) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        // Set diagonal element rho[idx][idx] = val
        rho[idx * dim + idx] = make_cuDoubleComplex(val, 0.0);
    }
}

__global__ void dmComputeDiagonal(const cuDoubleComplex* rho, double* diag, size_t dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        diag[idx] = cuCreal(rho[idx * dim + idx]);
    }
}

__global__ void dmComputeTrace(const cuDoubleComplex* rho, double* trace, size_t dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        atomicAdd(trace, cuCreal(rho[idx * dim + idx]));
    }
}

// Helper: get row and column indices for 2x2 submatrix transformation
__device__ inline void getDMPairIndices(size_t row, size_t col, int target,
                                         size_t& r0, size_t& r1, size_t& c0, size_t& c1) {
    size_t mask = (1ULL << target) - 1;
    // Row indices
    r0 = (row & mask) | ((row & ~mask) << 1);
    r1 = r0 | (1ULL << target);
    // Column indices
    c0 = (col & mask) | ((col & ~mask) << 1);
    c1 = c0 | (1ULL << target);
}

// Single-qubit gate: rho' = U rho U^dag
// Each thread handles one element of the output matrix
// For qubit q, we need to transform 2x2 blocks

__global__ void dmApplyX(cuDoubleComplex* rho, int n_qubits, int target) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    // X swaps |0> and |1> for target qubit
    // rho'[i][j] = rho[i XOR mask][j XOR mask]
    size_t mask = 1ULL << target;
    size_t new_row = row ^ mask;
    size_t new_col = col ^ mask;
    
    // Swap pairs to avoid race condition
    if (row < new_row || (row == new_row && col < new_col)) {
        size_t idx1 = row * dim + col;
        size_t idx2 = new_row * dim + new_col;
        cuDoubleComplex tmp = rho[idx1];
        rho[idx1] = rho[idx2];
        rho[idx2] = tmp;
    }
}

__global__ void dmApplyZ(cuDoubleComplex* rho, int n_qubits, int target) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    // Z: phase of -1 when target qubit is |1>
    // rho'[i][j] = (-1)^(i_t) * (-1)^(j_t) * rho[i][j] = (-1)^(i_t XOR j_t) * rho[i][j]
    int row_bit = (row >> target) & 1;
    int col_bit = (col >> target) & 1;
    
    if (row_bit != col_bit) {
        rho[idx] = make_cuDoubleComplex(-cuCreal(rho[idx]), -cuCimag(rho[idx]));
    }
}

__global__ void dmApplyY(cuDoubleComplex* rho, int n_qubits, int target) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    size_t mask = 1ULL << target;
    size_t new_row = row ^ mask;
    size_t new_col = col ^ mask;
    
    // Y = [[0, -i], [i, 0]]
    // rho' = Y rho Y^dag = Y rho Y (since Y^dag = Y for Pauli)
    // Actually Y^dag = -Y, but Y rho Y^dag = Y rho (-Y) introduces sign
    
    int row_bit = (row >> target) & 1;
    int col_bit = (col >> target) & 1;
    
    // Phase factor: i^(row_bit) * (-i)^(col_bit) = i^(row_bit - col_bit)
    // For row_bit=0,col_bit=0: phase = 1
    // For row_bit=0,col_bit=1: phase = -i (but we also swap)
    // For row_bit=1,col_bit=0: phase = i (but we also swap)
    // For row_bit=1,col_bit=1: phase = 1
    
    if (row < new_row || (row == new_row && col < new_col)) {
        size_t idx1 = row * dim + col;
        size_t idx2 = new_row * dim + new_col;
        
        cuDoubleComplex v1 = rho[idx1];
        cuDoubleComplex v2 = rho[idx2];
        
        // Apply Y: swap with phase factors
        int phase1 = (row_bit ? 1 : -1) * (col_bit ? -1 : 1);
        int phase2 = ((!row_bit) ? 1 : -1) * ((!col_bit) ? -1 : 1);
        
        rho[idx1] = make_cuDoubleComplex(phase2 * cuCreal(v2), phase2 * cuCimag(v2));
        rho[idx2] = make_cuDoubleComplex(phase1 * cuCreal(v1), phase1 * cuCimag(v1));
    }
}

__global__ void dmApplyH(cuDoubleComplex* rho, int n_qubits, int target) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    size_t mask = 1ULL << target;
    int row_bit = (row >> target) & 1;
    int col_bit = (col >> target) & 1;
    
    // Only process each 2x2 block once
    if (row_bit == 0 && col_bit == 0) {
        size_t r0 = row, r1 = row | mask;
        size_t c0 = col, c1 = col | mask;
        
        // Get 2x2 block
        cuDoubleComplex a00 = rho[r0 * dim + c0];
        cuDoubleComplex a01 = rho[r0 * dim + c1];
        cuDoubleComplex a10 = rho[r1 * dim + c0];
        cuDoubleComplex a11 = rho[r1 * dim + c1];
        
        // H rho H = 0.5 * [[1,1],[1,-1]] * rho * [[1,1],[1,-1]]
        const double half = 0.5;
        
        // New values
        cuDoubleComplex n00 = make_cuDoubleComplex(
            half * (cuCreal(a00) + cuCreal(a01) + cuCreal(a10) + cuCreal(a11)),
            half * (cuCimag(a00) + cuCimag(a01) + cuCimag(a10) + cuCimag(a11))
        );
        cuDoubleComplex n01 = make_cuDoubleComplex(
            half * (cuCreal(a00) - cuCreal(a01) + cuCreal(a10) - cuCreal(a11)),
            half * (cuCimag(a00) - cuCimag(a01) + cuCimag(a10) - cuCimag(a11))
        );
        cuDoubleComplex n10 = make_cuDoubleComplex(
            half * (cuCreal(a00) + cuCreal(a01) - cuCreal(a10) - cuCreal(a11)),
            half * (cuCimag(a00) + cuCimag(a01) - cuCimag(a10) - cuCimag(a11))
        );
        cuDoubleComplex n11 = make_cuDoubleComplex(
            half * (cuCreal(a00) - cuCreal(a01) - cuCreal(a10) + cuCreal(a11)),
            half * (cuCimag(a00) - cuCimag(a01) - cuCimag(a10) + cuCimag(a11))
        );
        
        rho[r0 * dim + c0] = n00;
        rho[r0 * dim + c1] = n01;
        rho[r1 * dim + c0] = n10;
        rho[r1 * dim + c1] = n11;
    }
}

__global__ void dmApplyS(cuDoubleComplex* rho, int n_qubits, int target) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    int row_bit = (row >> target) & 1;
    int col_bit = (col >> target) & 1;
    
    // S = diag(1, i), S^dag = diag(1, -i)
    // For density matrix: rho'[r][c] = S[row_bit] * rho[r][c] * S^dag[col_bit]
    // S[0] = 1, S[1] = i
    // S^dag[0] = 1, S^dag[1] = -i
    
    // Phase factor: i^(row_bit) * (-i)^(col_bit) = i^(row_bit - col_bit)
    // row_bit=0, col_bit=0: 1
    // row_bit=1, col_bit=0: i
    // row_bit=0, col_bit=1: -i
    // row_bit=1, col_bit=1: 1
    
    int phase_exp = row_bit - col_bit;  // -1, 0, or 1
    
    cuDoubleComplex val = rho[idx];
    cuDoubleComplex result;
    
    if (phase_exp == 0) {
        result = val;  // No phase change
    } else if (phase_exp == 1) {
        // Multiply by i: (a + bi) * i = -b + ai
        result = make_cuDoubleComplex(-cuCimag(val), cuCreal(val));
    } else {  // phase_exp == -1
        // Multiply by -i: (a + bi) * (-i) = b - ai
        result = make_cuDoubleComplex(cuCimag(val), -cuCreal(val));
    }
    
    rho[idx] = result;
}

// T gate: T = diag(1, e^(i*pi/4)) = diag(1, (1+i)/sqrt(2))
__global__ void dmApplyT(cuDoubleComplex* rho, int n_qubits, int target) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    int row_bit = (row >> target) & 1;
    int col_bit = (col >> target) & 1;
    
    // T = diag(1, e^(i*pi/4)), T^dag = diag(1, e^(-i*pi/4))
    // Phase factor: e^(i*pi/4*row_bit) * e^(-i*pi/4*col_bit) = e^(i*pi/4*(row_bit - col_bit))
    
    int phase_exp = row_bit - col_bit;  // -1, 0, or 1
    
    if (phase_exp == 0) return;  // No change needed
    
    cuDoubleComplex val = rho[idx];
    const double inv_sqrt2 = 0.70710678118654752440;
    
    if (phase_exp == 1) {
        // Multiply by e^(i*pi/4) = (1+i)/sqrt(2)
        double re = inv_sqrt2 * (cuCreal(val) - cuCimag(val));
        double im = inv_sqrt2 * (cuCreal(val) + cuCimag(val));
        rho[idx] = make_cuDoubleComplex(re, im);
    } else {  // phase_exp == -1
        // Multiply by e^(-i*pi/4) = (1-i)/sqrt(2)
        double re = inv_sqrt2 * (cuCreal(val) + cuCimag(val));
        double im = inv_sqrt2 * (-cuCreal(val) + cuCimag(val));
        rho[idx] = make_cuDoubleComplex(re, im);
    }
}

// S^dag gate: Sdag = diag(1, -i)
__global__ void dmApplySdag(cuDoubleComplex* rho, int n_qubits, int target) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    int row_bit = (row >> target) & 1;
    int col_bit = (col >> target) & 1;
    
    // Sdag = diag(1, -i), (Sdag)^dag = S = diag(1, i)
    // Phase factor: (-i)^row_bit * i^col_bit
    
    int phase_exp = col_bit - row_bit;  // Opposite of S gate
    
    if (phase_exp == 0) return;
    
    cuDoubleComplex val = rho[idx];
    cuDoubleComplex result;
    
    if (phase_exp == 1) {
        // Multiply by i
        result = make_cuDoubleComplex(-cuCimag(val), cuCreal(val));
    } else {  // phase_exp == -1
        // Multiply by -i
        result = make_cuDoubleComplex(cuCimag(val), -cuCreal(val));
    }
    
    rho[idx] = result;
}

// T^dag gate: Tdag = diag(1, e^(-i*pi/4))
__global__ void dmApplyTdag(cuDoubleComplex* rho, int n_qubits, int target) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    int row_bit = (row >> target) & 1;
    int col_bit = (col >> target) & 1;
    
    // Tdag = diag(1, e^(-i*pi/4)), (Tdag)^dag = T = diag(1, e^(i*pi/4))
    // Phase factor: e^(-i*pi/4*row_bit) * e^(i*pi/4*col_bit) = e^(i*pi/4*(col_bit - row_bit))
    
    int phase_exp = col_bit - row_bit;  // Opposite of T gate
    
    if (phase_exp == 0) return;
    
    cuDoubleComplex val = rho[idx];
    const double inv_sqrt2 = 0.70710678118654752440;
    
    if (phase_exp == 1) {
        // Multiply by e^(i*pi/4) = (1+i)/sqrt(2)
        double re = inv_sqrt2 * (cuCreal(val) - cuCimag(val));
        double im = inv_sqrt2 * (cuCreal(val) + cuCimag(val));
        rho[idx] = make_cuDoubleComplex(re, im);
    } else {  // phase_exp == -1
        // Multiply by e^(-i*pi/4) = (1-i)/sqrt(2)
        double re = inv_sqrt2 * (cuCreal(val) + cuCimag(val));
        double im = inv_sqrt2 * (-cuCreal(val) + cuCimag(val));
        rho[idx] = make_cuDoubleComplex(re, im);
    }
}

// Rx gate: Rx(theta) - rotation around X axis
// Non-diagonal gate, needs 2x2 block processing
__global__ void dmApplyRx(cuDoubleComplex* rho, int n_qubits, int target, double theta) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    size_t mask = 1ULL << target;
    int row_bit = (row >> target) & 1;
    int col_bit = (col >> target) & 1;
    
    // Only process each 2x2 block once (when row_bit=0 and col_bit=0)
    if (row_bit == 0 && col_bit == 0) {
        size_t r0 = row, r1 = row | mask;
        size_t c0 = col, c1 = col | mask;
        
        double c = cos(theta / 2.0);
        double s = sin(theta / 2.0);
        double c2 = c * c;
        double s2 = s * s;
        
        cuDoubleComplex a00 = rho[r0 * dim + c0];
        cuDoubleComplex a01 = rho[r0 * dim + c1];
        cuDoubleComplex a10 = rho[r1 * dim + c0];
        cuDoubleComplex a11 = rho[r1 * dim + c1];
        
        // Rx = [[c, -is], [-is, c]]
        // Rx^dag = [[c, is], [is, c]]
        // rho' = Rx * rho * Rx^dag
        
        // n00 = c*a00*c + c*a01*(is) + (-is)*a10*c + (-is)*a11*(is)
        //     = c^2*a00 + ics*a01 - ics*a10 + s^2*a11
        rho[r0 * dim + c0] = make_cuDoubleComplex(
            c2 * cuCreal(a00) + s2 * cuCreal(a11) - s * (cuCimag(a01) - cuCimag(a10)) * c,
            c2 * cuCimag(a00) + s2 * cuCimag(a11) + s * (cuCreal(a01) - cuCreal(a10)) * c
        );
        
        // n01 = c*a00*(is) + c*a01*c + (-is)*a10*(is) + (-is)*a11*c
        //     = ics*a00 + c^2*a01 + s^2*a10 - ics*a11
        rho[r0 * dim + c1] = make_cuDoubleComplex(
            c2 * cuCreal(a01) + s2 * cuCreal(a10) + c * s * (cuCimag(a00) - cuCimag(a11)),
            c2 * cuCimag(a01) + s2 * cuCimag(a10) - c * s * (cuCreal(a00) - cuCreal(a11))
        );
        
        // n10 = (-is)*a00*c + (-is)*a01*(is) + c*a10*c + c*a11*(is)
        //     = -ics*a00 + s^2*a01 + c^2*a10 + ics*a11
        rho[r1 * dim + c0] = make_cuDoubleComplex(
            c2 * cuCreal(a10) + s2 * cuCreal(a01) - c * s * (cuCimag(a00) - cuCimag(a11)),
            c2 * cuCimag(a10) + s2 * cuCimag(a01) + c * s * (cuCreal(a00) - cuCreal(a11))
        );
        
        // n11 = (-is)*a00*(is) + (-is)*a01*c + c*a10*(is) + c*a11*c
        //     = s^2*a00 - ics*a01 + ics*a10 + c^2*a11
        rho[r1 * dim + c1] = make_cuDoubleComplex(
            c2 * cuCreal(a11) + s2 * cuCreal(a00) + s * (cuCimag(a01) - cuCimag(a10)) * c,
            c2 * cuCimag(a11) + s2 * cuCimag(a00) - s * (cuCreal(a01) - cuCreal(a10)) * c
        );
    }
}

// Rz gate: Rz(theta) = diag(e^(-i*theta/2), e^(i*theta/2))
// This is a diagonal gate
__global__ void dmApplyRz(cuDoubleComplex* rho, int n_qubits, int target, double theta) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    int row_bit = (row >> target) & 1;
    int col_bit = (col >> target) & 1;
    
    // Rz = diag(e^(-i*theta/2), e^(i*theta/2))
    // Rz^dag = diag(e^(i*theta/2), e^(-i*theta/2))
    // Phase for row: e^(i*theta/2 * (2*row_bit - 1)) (row_bit=0 -> -1, row_bit=1 -> 1)
    // Phase for col^dag: e^(-i*theta/2 * (2*col_bit - 1))
    // Combined: e^(i*theta/2 * ((2*row_bit-1) - (2*col_bit-1))) = e^(i*theta*(row_bit - col_bit))
    
    int phase_factor = row_bit - col_bit;  // -1, 0, or 1
    
    if (phase_factor == 0) return;
    
    cuDoubleComplex val = rho[idx];
    double angle = theta * phase_factor;
    double c = cos(angle);
    double s = sin(angle);
    
    // Multiply by e^(i*angle) = cos(angle) + i*sin(angle)
    rho[idx] = make_cuDoubleComplex(
        c * cuCreal(val) - s * cuCimag(val),
        s * cuCreal(val) + c * cuCimag(val)
    );
}

__global__ void dmApplyRy(cuDoubleComplex* rho, int n_qubits, int target, double theta) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    size_t mask = 1ULL << target;
    int row_bit = (row >> target) & 1;
    int col_bit = (col >> target) & 1;
    
    if (row_bit == 0 && col_bit == 0) {
        size_t r0 = row, r1 = row | mask;
        size_t c0 = col, c1 = col | mask;
        
        double c = cos(theta / 2.0);
        double s = sin(theta / 2.0);
        double c2 = c * c;
        double s2 = s * s;
        double cs = c * s;
        
        cuDoubleComplex a00 = rho[r0 * dim + c0];
        cuDoubleComplex a01 = rho[r0 * dim + c1];
        cuDoubleComplex a10 = rho[r1 * dim + c0];
        cuDoubleComplex a11 = rho[r1 * dim + c1];
        
        // Ry = [[c, -s], [s, c]] (real matrix)
        // rho' = Ry * rho * Ry^T
        
        rho[r0 * dim + c0] = make_cuDoubleComplex(
            c2 * cuCreal(a00) - cs * cuCreal(a01) - cs * cuCreal(a10) + s2 * cuCreal(a11),
            c2 * cuCimag(a00) - cs * cuCimag(a01) - cs * cuCimag(a10) + s2 * cuCimag(a11)
        );
        rho[r0 * dim + c1] = make_cuDoubleComplex(
            cs * cuCreal(a00) + c2 * cuCreal(a01) - s2 * cuCreal(a10) - cs * cuCreal(a11),
            cs * cuCimag(a00) + c2 * cuCimag(a01) - s2 * cuCimag(a10) - cs * cuCimag(a11)
        );
        rho[r1 * dim + c0] = make_cuDoubleComplex(
            cs * cuCreal(a00) - s2 * cuCreal(a01) + c2 * cuCreal(a10) - cs * cuCreal(a11),
            cs * cuCimag(a00) - s2 * cuCimag(a01) + c2 * cuCimag(a10) - cs * cuCimag(a11)
        );
        rho[r1 * dim + c1] = make_cuDoubleComplex(
            s2 * cuCreal(a00) + cs * cuCreal(a01) + cs * cuCreal(a10) + c2 * cuCreal(a11),
            s2 * cuCimag(a00) + cs * cuCimag(a01) + cs * cuCimag(a10) + c2 * cuCimag(a11)
        );
    }
}

// Two-qubit gates

__global__ void dmApplyCNOT(cuDoubleComplex* rho, int n_qubits, int control, int target) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    // CNOT flips target when control is 1
    // For density matrix: swap rows/cols where control=1
    int row_ctrl = (row >> control) & 1;
    int col_ctrl = (col >> control) & 1;
    
    size_t t_mask = 1ULL << target;
    
    // When control is 1, flip the target bit in both row and col indices
    size_t new_row = row_ctrl ? (row ^ t_mask) : row;
    size_t new_col = col_ctrl ? (col ^ t_mask) : col;
    
    if (row != new_row || col != new_col) {
        // Only process if we need to swap, and only process once
        if (row < new_row || (row == new_row && col < new_col)) {
            size_t idx1 = row * dim + col;
            size_t idx2 = new_row * dim + new_col;
            cuDoubleComplex tmp = rho[idx1];
            rho[idx1] = rho[idx2];
            rho[idx2] = tmp;
        }
    }
}

__global__ void dmApplyCZ(cuDoubleComplex* rho, int n_qubits, int control, int target) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    // CZ applies -1 phase when both qubits are |1>
    int row_both = ((row >> control) & 1) && ((row >> target) & 1);
    int col_both = ((col >> control) & 1) && ((col >> target) & 1);
    
    // Phase = (-1)^(row_both XOR col_both)
    if (row_both != col_both) {
        rho[idx] = make_cuDoubleComplex(-cuCreal(rho[idx]), -cuCimag(rho[idx]));
    }
}

__global__ void dmApplySWAP(cuDoubleComplex* rho, int n_qubits, int qubit1, int qubit2) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    // SWAP exchanges qubit1 and qubit2
    int r1 = (row >> qubit1) & 1;
    int r2 = (row >> qubit2) & 1;
    int c1 = (col >> qubit1) & 1;
    int c2 = (col >> qubit2) & 1;
    
    // Compute new indices with swapped bits
    size_t new_row = row;
    size_t new_col = col;
    
    if (r1 != r2) {
        new_row ^= (1ULL << qubit1) | (1ULL << qubit2);
    }
    if (c1 != c2) {
        new_col ^= (1ULL << qubit1) | (1ULL << qubit2);
    }
    
    if (row != new_row || col != new_col) {
        if (row < new_row || (row == new_row && col < new_col)) {
            size_t idx1 = row * dim + col;
            size_t idx2 = new_row * dim + new_col;
            cuDoubleComplex tmp = rho[idx1];
            rho[idx1] = rho[idx2];
            rho[idx2] = tmp;
        }
    }
}

// Noise channels

__global__ void dmApplyDepolarizing(cuDoubleComplex* rho, int n_qubits, int target, double p) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    int row_bit = (row >> target) & 1;
    int col_bit = (col >> target) & 1;
    
    // Depolarizing channel: rho' = (1-p)*rho + (p/3)*(X*rho*X + Y*rho*Y + Z*rho*Z)
    // For off-diagonal elements (row_bit != col_bit): scale by (1 - 4p/3)
    // For diagonal elements: mix with other diagonal
    
    cuDoubleComplex val = rho[idx];
    
    if (row_bit != col_bit) {
        // Off-diagonal: scale by (1 - 4p/3)
        double scale = 1.0 - 4.0 * p / 3.0;
        rho[idx] = make_cuDoubleComplex(scale * cuCreal(val), scale * cuCimag(val));
    }
    // Diagonal elements stay mostly the same for single-qubit depolarizing
    // (full treatment requires mixing with partner diagonal element)
}

__global__ void dmApplyAmplitudeDamping(cuDoubleComplex* rho, int n_qubits, int target, double gamma) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    int row_bit = (row >> target) & 1;
    int col_bit = (col >> target) & 1;
    size_t mask = 1ULL << target;
    
    // Amplitude damping Kraus operators:
    // K0 = [[1, 0], [0, sqrt(1-gamma)]]
    // K1 = [[0, sqrt(gamma)], [0, 0]]
    
    cuDoubleComplex val = rho[idx];
    double sqrt_1mg = sqrt(1.0 - gamma);
    
    if (row_bit == 0 && col_bit == 0) {
        // |0><0| block: gets contribution from |1><1| via K1
        size_t partner_idx = (row | mask) * dim + (col | mask);
        cuDoubleComplex partner = rho[partner_idx];
        rho[idx] = make_cuDoubleComplex(
            cuCreal(val) + gamma * cuCreal(partner),
            cuCimag(val) + gamma * cuCimag(partner)
        );
    } else if (row_bit == 1 && col_bit == 1) {
        // |1><1| block: scaled by (1-gamma)
        rho[idx] = make_cuDoubleComplex(
            (1.0 - gamma) * cuCreal(val),
            (1.0 - gamma) * cuCimag(val)
        );
    } else {
        // Off-diagonal: scaled by sqrt(1-gamma)
        rho[idx] = make_cuDoubleComplex(
            sqrt_1mg * cuCreal(val),
            sqrt_1mg * cuCimag(val)
        );
    }
}

__global__ void dmApplyPhaseDamping(cuDoubleComplex* rho, int n_qubits, int target, double gamma) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    int row_bit = (row >> target) & 1;
    int col_bit = (col >> target) & 1;
    
    // Phase damping only affects off-diagonal elements
    // rho_01 and rho_10 get scaled by sqrt(1-gamma)
    
    if (row_bit != col_bit) {
        double scale = sqrt(1.0 - gamma);
        cuDoubleComplex val = rho[idx];
        rho[idx] = make_cuDoubleComplex(scale * cuCreal(val), scale * cuCimag(val));
    }
}

__global__ void dmApplyBitFlip(cuDoubleComplex* rho, int n_qubits, int target, double p) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    size_t mask = 1ULL << target;
    
    // Bit flip: rho' = (1-p)*rho + p*X*rho*X
    // This mixes rho[i][j] with rho[i^mask][j^mask]
    
    size_t partner_row = row ^ mask;
    size_t partner_col = col ^ mask;
    
    // Only process each pair once
    if (row < partner_row || (row == partner_row && col < partner_col)) {
        size_t idx1 = row * dim + col;
        size_t idx2 = partner_row * dim + partner_col;
        
        cuDoubleComplex v1 = rho[idx1];
        cuDoubleComplex v2 = rho[idx2];
        
        rho[idx1] = make_cuDoubleComplex(
            (1.0 - p) * cuCreal(v1) + p * cuCreal(v2),
            (1.0 - p) * cuCimag(v1) + p * cuCimag(v2)
        );
        rho[idx2] = make_cuDoubleComplex(
            p * cuCreal(v1) + (1.0 - p) * cuCreal(v2),
            p * cuCimag(v1) + (1.0 - p) * cuCimag(v2)
        );
    }
}

__global__ void dmApplyPhaseFlip(cuDoubleComplex* rho, int n_qubits, int target, double p) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    int row_bit = (row >> target) & 1;
    int col_bit = (col >> target) & 1;
    
    // Phase flip: rho' = (1-p)*rho + p*Z*rho*Z
    // Z*rho*Z multiplies by (-1)^(row_bit XOR col_bit)
    // So off-diagonal elements get mixed: rho' = (1-2p)*rho for off-diag
    
    if (row_bit != col_bit) {
        double scale = 1.0 - 2.0 * p;
        cuDoubleComplex val = rho[idx];
        rho[idx] = make_cuDoubleComplex(scale * cuCreal(val), scale * cuCimag(val));
    }
}

/**
 * Kernel to collapse density matrix after measurement.
 * Projects onto |result><result| subspace for measured qubit.
 * 
 * Reference: Nielsen & Chuang, Section 2.4 "Measurements"
 * 
 * For measurement outcome m on qubit q:
 *   ρ' = P_m ρ P_m / Tr(P_m ρ P_m)
 * where P_m = |m><m| on qubit q tensored with identity on others.
 */
__global__ void dmCollapseMeasurement(cuDoubleComplex* rho, int n_qubits, int target, 
                                       int result, double norm_factor) {
    size_t dim = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * dim) return;
    
    size_t row = idx / dim;
    size_t col = idx % dim;
    
    int row_bit = (row >> target) & 1;
    int col_bit = (col >> target) & 1;
    
    // Zero out elements where target qubit doesn't match result
    if (row_bit != result || col_bit != result) {
        rho[idx] = make_cuDoubleComplex(0.0, 0.0);
    } else {
        // Renormalize surviving elements
        cuDoubleComplex val = rho[idx];
        rho[idx] = make_cuDoubleComplex(
            cuCreal(val) * norm_factor,
            cuCimag(val) * norm_factor
        );
    }
}

} // namespace qsim
