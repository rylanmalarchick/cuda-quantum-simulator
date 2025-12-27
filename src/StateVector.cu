#include "StateVector.cuh"
#include "Constants.hpp"
#include "CudaMemory.cuh"
#include <cuda_runtime.h>
#include <curand.h>
#include <stdexcept>
#include <cstring>

namespace qsim {

// ============================================================================
// Kernels
// ============================================================================

__global__ void initializeZeroKernel(cuDoubleComplex* state, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // |0...0⟩ = [1, 0, 0, ..., 0]
        state[idx] = (idx == 0) ? make_cuDoubleComplex(1.0, 0.0) 
                                 : make_cuDoubleComplex(0.0, 0.0);
    }
}

__global__ void initializeBasisKernel(cuDoubleComplex* state, size_t size, size_t basis_idx) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        state[idx] = (idx == basis_idx) ? make_cuDoubleComplex(1.0, 0.0) 
                                         : make_cuDoubleComplex(0.0, 0.0);
    }
}

__global__ void probabilityKernel(const cuDoubleComplex* state, double* probs, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double real = cuCreal(state[idx]);
        double imag = cuCimag(state[idx]);
        probs[idx] = real * real + imag * imag;
    }
}

// Simple parallel reduction (not fully optimized, good enough for now)
__global__ void sumReductionKernel(double* data, size_t size) {
    extern __shared__ double sdata[];
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < size) ? data[idx] : 0.0;
    __syncthreads();
    
    // Reduction in shared memory
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        data[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// StateVector Implementation
// ============================================================================

StateVector::StateVector(int num_qubits)
    : num_qubits_(num_qubits)
    , size_(1ULL << num_qubits)
    , d_state_(nullptr)
{
    if (!isValidQubitCount(num_qubits)) {
        throw std::invalid_argument(
            "Number of qubits must be between " + 
            std::to_string(cuda_config::MIN_QUBITS) + " and " + 
            std::to_string(cuda_config::MAX_QUBITS)
        );
    }
    allocate();
    initializeZero();
}

StateVector::~StateVector() {
    deallocate();
}

StateVector::StateVector(StateVector&& other) noexcept
    : num_qubits_(other.num_qubits_)
    , size_(other.size_)
    , d_state_(other.d_state_)
{
    other.d_state_ = nullptr;
    other.size_ = 0;
    other.num_qubits_ = 0;
}

StateVector& StateVector::operator=(StateVector&& other) noexcept {
    if (this != &other) {
        deallocate();
        num_qubits_ = other.num_qubits_;
        size_ = other.size_;
        d_state_ = other.d_state_;
        other.d_state_ = nullptr;
        other.size_ = 0;
        other.num_qubits_ = 0;
    }
    return *this;
}

void StateVector::allocate() {
    size_t bytes = size_ * sizeof(cuDoubleComplex);
    CUDA_CHECK(cudaMalloc(&d_state_, bytes));
}

void StateVector::deallocate() {
    if (d_state_) {
        cudaFree(d_state_);
        d_state_ = nullptr;
    }
}

void StateVector::initializeZero() {
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(size_, threads);
    initializeZeroKernel<<<blocks, threads>>>(d_state_, size_);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void StateVector::initializeBasis(size_t basis_idx) {
    if (basis_idx >= size_) {
        throw std::invalid_argument("Basis index out of range");
    }
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(size_, threads);
    initializeBasisKernel<<<blocks, threads>>>(d_state_, size_, basis_idx);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<std::complex<double>> StateVector::toHost() const {
    std::vector<cuDoubleComplex> temp(size_);
    CUDA_CHECK(cudaMemcpy(temp.data(), d_state_, size_ * sizeof(cuDoubleComplex), 
                          cudaMemcpyDeviceToHost));
    
    // Convert cuDoubleComplex to std::complex<double>
    std::vector<std::complex<double>> result(size_);
    for (size_t i = 0; i < size_; ++i) {
        result[i] = std::complex<double>(cuCreal(temp[i]), cuCimag(temp[i]));
    }
    return result;
}

std::vector<double> StateVector::getProbabilities() const {
    // RAII wrapper - memory automatically freed even if exception thrown
    CudaMemory<double> d_probs(size_);
    
    // Calculate |amplitude|^2
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(size_, threads);
    probabilityKernel<<<blocks, threads>>>(d_state_, d_probs.get(), size_);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy to host
    std::vector<double> probs(size_);
    d_probs.copyToHost(probs.data(), size_);
    
    return probs;
}

double StateVector::getTotalProbability() const {
    auto probs = getProbabilities();
    double sum = 0.0;
    for (double p : probs) {
        sum += p;
    }
    return sum;
}

bool StateVector::isNormalized(double tolerance) const {
    double total = getTotalProbability();
    return std::abs(total - 1.0) <= tolerance;
}

void StateVector::assertNormalized(double tolerance) const {
    double total = getTotalProbability();
    if (std::abs(total - 1.0) > tolerance) {
        throw std::runtime_error(
            "State vector not normalized: total probability = " + 
            std::to_string(total) + " (expected 1.0, tolerance = " + 
            std::to_string(tolerance) + ")"
        );
    }
}

int StateVector::measure(int qubit) {
    // TODO: Implement measurement with state collapse
    (void)qubit;
    throw std::runtime_error("Measurement not yet implemented");
}

std::vector<int> StateVector::sample(int n_shots) {
    // TODO: Implement sampling
    (void)n_shots;
    throw std::runtime_error("Sampling not yet implemented");
}

} // namespace qsim
