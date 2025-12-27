// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Rylan Malarchick

/**
 * @file test_optimized_gates.cu
 * @brief Tests for OptimizedGates.cuh
 * @author Rylan Malarchick
 * @date 2024
 *
 * Verifies that optimized gate kernels produce identical results to standard
 * gate kernels across all qubit positions, testing both shared memory path
 * (low-order qubits) and coalesced access path (high-order qubits).
 */

#include <gtest/gtest.h>
#include "OptimizedGates.cuh"
#include "Gates.cuh"
#include "Constants.hpp"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <complex>
#include <random>
#include <cmath>

using namespace qsim;

const double EPSILON = 1e-10;

class OptimizedGatesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Seed RNG for reproducible tests
        rng_.seed(42);
    }
    
    // Initialize state vector to |0...0⟩
    void initializeToZero(cuDoubleComplex* d_state, size_t size) {
        std::vector<cuDoubleComplex> h_state(size, make_cuDoubleComplex(0.0, 0.0));
        h_state[0] = make_cuDoubleComplex(1.0, 0.0);
        cudaMemcpy(d_state, h_state.data(), size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    }
    
    // Initialize state vector to random normalized state
    void initializeToRandom(cuDoubleComplex* d_state, size_t size) {
        std::vector<cuDoubleComplex> h_state(size);
        std::normal_distribution<double> dist(0.0, 1.0);
        
        double norm = 0.0;
        for (size_t i = 0; i < size; ++i) {
            h_state[i] = make_cuDoubleComplex(dist(rng_), dist(rng_));
            norm += cuCreal(h_state[i]) * cuCreal(h_state[i]) + 
                    cuCimag(h_state[i]) * cuCimag(h_state[i]);
        }
        norm = sqrt(norm);
        for (size_t i = 0; i < size; ++i) {
            h_state[i] = make_cuDoubleComplex(cuCreal(h_state[i]) / norm, 
                                               cuCimag(h_state[i]) / norm);
        }
        cudaMemcpy(d_state, h_state.data(), size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    }
    
    // Copy device state to host
    std::vector<std::complex<double>> getState(cuDoubleComplex* d_state, size_t size) {
        std::vector<cuDoubleComplex> h_state(size);
        cudaMemcpy(h_state.data(), d_state, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        
        std::vector<std::complex<double>> result(size);
        for (size_t i = 0; i < size; ++i) {
            result[i] = std::complex<double>(cuCreal(h_state[i]), cuCimag(h_state[i]));
        }
        return result;
    }
    
    // Compare two state vectors
    void expectStatesEqual(const std::vector<std::complex<double>>& a,
                           const std::vector<std::complex<double>>& b,
                           const std::string& msg = "") {
        ASSERT_EQ(a.size(), b.size()) << msg;
        for (size_t i = 0; i < a.size(); ++i) {
            EXPECT_NEAR(a[i].real(), b[i].real(), EPSILON) 
                << msg << " - Real part mismatch at index " << i;
            EXPECT_NEAR(a[i].imag(), b[i].imag(), EPSILON) 
                << msg << " - Imag part mismatch at index " << i;
        }
    }
    
    std::mt19937 rng_;
};

// ============================================================================
// Hadamard Gate Tests
// ============================================================================

TEST_F(OptimizedGatesTest, HadamardLowQubit_SharedMemPath) {
    // Test H gate on qubit 0 (uses shared memory path)
    const int n_qubits = 8;
    const size_t size = 1ULL << n_qubits;
    const int target = 0;  // Low qubit - shared memory path
    
    cuDoubleComplex *d_standard, *d_optimized;
    cudaMalloc(&d_standard, size * sizeof(cuDoubleComplex));
    cudaMalloc(&d_optimized, size * sizeof(cuDoubleComplex));
    
    // Initialize both to same random state
    initializeToRandom(d_standard, size);
    cudaMemcpy(d_optimized, d_standard, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = (size / 2 + threads - 1) / threads;
    
    // Apply standard and optimized H gates
    applyH<<<blocks, threads>>>(d_standard, n_qubits, target);
    applyH_opt<<<blocks, threads>>>(d_optimized, n_qubits, target);
    cudaDeviceSynchronize();
    
    auto state_std = getState(d_standard, size);
    auto state_opt = getState(d_optimized, size);
    
    expectStatesEqual(state_std, state_opt, "H gate on qubit 0 (shared memory path)");
    
    cudaFree(d_standard);
    cudaFree(d_optimized);
}

TEST_F(OptimizedGatesTest, HadamardHighQubit_CoalescedPath) {
    // Test H gate on qubit 7 (uses coalesced access path)
    const int n_qubits = 10;
    const size_t size = 1ULL << n_qubits;
    const int target = 7;  // High qubit - coalesced path
    
    cuDoubleComplex *d_standard, *d_optimized;
    cudaMalloc(&d_standard, size * sizeof(cuDoubleComplex));
    cudaMalloc(&d_optimized, size * sizeof(cuDoubleComplex));
    
    initializeToRandom(d_standard, size);
    cudaMemcpy(d_optimized, d_standard, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = (size / 2 + threads - 1) / threads;
    
    applyH<<<blocks, threads>>>(d_standard, n_qubits, target);
    applyH_opt<<<blocks, threads>>>(d_optimized, n_qubits, target);
    cudaDeviceSynchronize();
    
    auto state_std = getState(d_standard, size);
    auto state_opt = getState(d_optimized, size);
    
    expectStatesEqual(state_std, state_opt, "H gate on qubit 7 (coalesced path)");
    
    cudaFree(d_standard);
    cudaFree(d_optimized);
}

TEST_F(OptimizedGatesTest, HadamardAllQubits) {
    // Test H gate on all qubit positions to verify both paths
    const int n_qubits = 10;
    const size_t size = 1ULL << n_qubits;
    
    cuDoubleComplex *d_standard, *d_optimized;
    cudaMalloc(&d_standard, size * sizeof(cuDoubleComplex));
    cudaMalloc(&d_optimized, size * sizeof(cuDoubleComplex));
    
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = (size / 2 + threads - 1) / threads;
    
    for (int target = 0; target < n_qubits; ++target) {
        initializeToRandom(d_standard, size);
        cudaMemcpy(d_optimized, d_standard, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        
        applyH<<<blocks, threads>>>(d_standard, n_qubits, target);
        applyH_opt<<<blocks, threads>>>(d_optimized, n_qubits, target);
        cudaDeviceSynchronize();
        
        auto state_std = getState(d_standard, size);
        auto state_opt = getState(d_optimized, size);
        
        expectStatesEqual(state_std, state_opt, 
                          "H gate on qubit " + std::to_string(target));
    }
    
    cudaFree(d_standard);
    cudaFree(d_optimized);
}

// ============================================================================
// X Gate Tests
// ============================================================================

TEST_F(OptimizedGatesTest, XGateAllQubits) {
    const int n_qubits = 10;
    const size_t size = 1ULL << n_qubits;
    
    cuDoubleComplex *d_standard, *d_optimized;
    cudaMalloc(&d_standard, size * sizeof(cuDoubleComplex));
    cudaMalloc(&d_optimized, size * sizeof(cuDoubleComplex));
    
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = (size / 2 + threads - 1) / threads;
    
    for (int target = 0; target < n_qubits; ++target) {
        initializeToRandom(d_standard, size);
        cudaMemcpy(d_optimized, d_standard, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        
        applyX<<<blocks, threads>>>(d_standard, n_qubits, target);
        applyX_opt<<<blocks, threads>>>(d_optimized, n_qubits, target);
        cudaDeviceSynchronize();
        
        auto state_std = getState(d_standard, size);
        auto state_opt = getState(d_optimized, size);
        
        expectStatesEqual(state_std, state_opt, 
                          "X gate on qubit " + std::to_string(target));
    }
    
    cudaFree(d_standard);
    cudaFree(d_optimized);
}

// ============================================================================
// CNOT Gate Tests
// ============================================================================

TEST_F(OptimizedGatesTest, CNOTGateVariousPositions) {
    const int n_qubits = 8;
    const size_t size = 1ULL << n_qubits;
    
    cuDoubleComplex *d_standard, *d_optimized;
    cudaMalloc(&d_standard, size * sizeof(cuDoubleComplex));
    cudaMalloc(&d_optimized, size * sizeof(cuDoubleComplex));
    
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = (size / 2 + threads - 1) / threads;
    
    // Test various control-target combinations
    std::vector<std::pair<int, int>> test_cases = {
        {0, 1}, {1, 0},           // Adjacent, low qubits
        {0, 7}, {7, 0},           // Far apart
        {3, 4}, {4, 3},           // Middle qubits
        {6, 7}, {7, 6},           // High qubits
        {2, 5}, {5, 2},           // Various distances
    };
    
    for (const auto& [control, target] : test_cases) {
        initializeToRandom(d_standard, size);
        cudaMemcpy(d_optimized, d_standard, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        
        applyCNOT<<<blocks, threads>>>(d_standard, n_qubits, control, target);
        applyCNOT_opt<<<blocks, threads>>>(d_optimized, n_qubits, control, target);
        cudaDeviceSynchronize();
        
        auto state_std = getState(d_standard, size);
        auto state_opt = getState(d_optimized, size);
        
        expectStatesEqual(state_std, state_opt, 
                          "CNOT with control=" + std::to_string(control) + 
                          ", target=" + std::to_string(target));
    }
    
    cudaFree(d_standard);
    cudaFree(d_optimized);
}

// ============================================================================
// General 1Q Gate Tests
// ============================================================================

TEST_F(OptimizedGatesTest, General1QGate_MatchesStandard) {
    const int n_qubits = 8;
    const size_t size = 1ULL << n_qubits;
    
    cuDoubleComplex *d_standard, *d_optimized;
    cudaMalloc(&d_standard, size * sizeof(cuDoubleComplex));
    cudaMalloc(&d_optimized, size * sizeof(cuDoubleComplex));
    
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = (size / 2 + threads - 1) / threads;
    
    // Test with Y gate matrix: [[0, -i], [i, 0]]
    cuDoubleComplex a = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex b = make_cuDoubleComplex(0.0, -1.0);
    cuDoubleComplex c = make_cuDoubleComplex(0.0, 1.0);
    cuDoubleComplex d = make_cuDoubleComplex(0.0, 0.0);
    
    for (int target = 0; target < n_qubits; ++target) {
        initializeToRandom(d_standard, size);
        cudaMemcpy(d_optimized, d_standard, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        
        // Apply Y using standard kernel
        applyY<<<blocks, threads>>>(d_standard, n_qubits, target);
        // Apply Y using optimized general 1Q kernel
        applyGate1Q_opt<<<blocks, threads>>>(d_optimized, n_qubits, target, a, b, c, d);
        cudaDeviceSynchronize();
        
        auto state_std = getState(d_standard, size);
        auto state_opt = getState(d_optimized, size);
        
        expectStatesEqual(state_std, state_opt, 
                          "General 1Q gate (Y) on qubit " + std::to_string(target));
    }
    
    cudaFree(d_standard);
    cudaFree(d_optimized);
}

// ============================================================================
// Larger System Tests
// ============================================================================

TEST_F(OptimizedGatesTest, LargerSystem_16Qubits) {
    // Test on a larger system to stress-test memory access patterns
    const int n_qubits = 16;
    const size_t size = 1ULL << n_qubits;
    
    cuDoubleComplex *d_standard, *d_optimized;
    cudaMalloc(&d_standard, size * sizeof(cuDoubleComplex));
    cudaMalloc(&d_optimized, size * sizeof(cuDoubleComplex));
    
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = (size / 2 + threads - 1) / threads;
    
    // Test a few key qubit positions
    std::vector<int> test_targets = {0, 4, 8, 12, 15};
    
    for (int target : test_targets) {
        initializeToRandom(d_standard, size);
        cudaMemcpy(d_optimized, d_standard, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        
        applyH<<<blocks, threads>>>(d_standard, n_qubits, target);
        applyH_opt<<<blocks, threads>>>(d_optimized, n_qubits, target);
        cudaDeviceSynchronize();
        
        auto state_std = getState(d_standard, size);
        auto state_opt = getState(d_optimized, size);
        
        expectStatesEqual(state_std, state_opt, 
                          "H on 16-qubit system, target=" + std::to_string(target));
    }
    
    cudaFree(d_standard);
    cudaFree(d_optimized);
}

// ============================================================================
// Circuit Simulation Test (Multiple Gates)
// ============================================================================

TEST_F(OptimizedGatesTest, CircuitSequence_MatchesStandard) {
    // Apply a sequence of gates and verify results match
    const int n_qubits = 8;
    const size_t size = 1ULL << n_qubits;
    
    cuDoubleComplex *d_standard, *d_optimized;
    cudaMalloc(&d_standard, size * sizeof(cuDoubleComplex));
    cudaMalloc(&d_optimized, size * sizeof(cuDoubleComplex));
    
    initializeToZero(d_standard, size);
    cudaMemcpy(d_optimized, d_standard, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = (size / 2 + threads - 1) / threads;
    
    // Apply circuit: H(0), H(1), CNOT(0,1), H(0), CNOT(1,2), X(3)
    // Standard
    applyH<<<blocks, threads>>>(d_standard, n_qubits, 0);
    applyH<<<blocks, threads>>>(d_standard, n_qubits, 1);
    applyCNOT<<<blocks, threads>>>(d_standard, n_qubits, 0, 1);
    applyH<<<blocks, threads>>>(d_standard, n_qubits, 0);
    applyCNOT<<<blocks, threads>>>(d_standard, n_qubits, 1, 2);
    applyX<<<blocks, threads>>>(d_standard, n_qubits, 3);
    
    // Optimized
    applyH_opt<<<blocks, threads>>>(d_optimized, n_qubits, 0);
    applyH_opt<<<blocks, threads>>>(d_optimized, n_qubits, 1);
    applyCNOT_opt<<<blocks, threads>>>(d_optimized, n_qubits, 0, 1);
    applyH_opt<<<blocks, threads>>>(d_optimized, n_qubits, 0);
    applyCNOT_opt<<<blocks, threads>>>(d_optimized, n_qubits, 1, 2);
    applyX_opt<<<blocks, threads>>>(d_optimized, n_qubits, 3);
    
    cudaDeviceSynchronize();
    
    auto state_std = getState(d_standard, size);
    auto state_opt = getState(d_optimized, size);
    
    expectStatesEqual(state_std, state_opt, "Circuit sequence");
    
    cudaFree(d_standard);
    cudaFree(d_optimized);
}
