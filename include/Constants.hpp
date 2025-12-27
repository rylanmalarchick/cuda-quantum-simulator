// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Rylan Malarchick

/**
 * @file Constants.hpp
 * @brief Mathematical constants, CUDA configuration, and error handling macros
 * @author Rylan Malarchick
 * @date 2024
 *
 * Centralizes all compile-time constants and configuration parameters for
 * the quantum simulator. Includes:
 *
 * - Mathematical constants (pi, sqrt(2), tolerances)
 * - CUDA kernel configuration (block sizes, qubit limits)
 * - Error handling macros (CUDA_CHECK, CUSOLVER_CHECK)
 *
 * Using centralized constants ensures consistency across the codebase and
 * makes it easy to tune parameters for different GPU architectures.
 *
 * @note CUDA_CHECK and CUSOLVER_CHECK macros provide file/line information
 *       for debugging CUDA errors.
 */
#pragma once

#include <cstddef>
#include <cmath>

namespace qsim {

// ============================================================================
// Mathematical Constants
// ============================================================================

namespace constants {

// Pi and related values
constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;
constexpr double HALF_PI = PI / 2.0;
constexpr double QUARTER_PI = PI / 4.0;

// Square roots
constexpr double SQRT2 = 1.41421356237309504880;
constexpr double INV_SQRT2 = 0.70710678118654752440;  // 1/sqrt(2)

// Tolerance for numerical comparisons
constexpr double EPSILON = 1e-10;
constexpr double PROBABILITY_EPSILON = 1e-12;  // For probability normalization

} // namespace constants

// ============================================================================
// CUDA Configuration
// ============================================================================

namespace cuda_config {

// Thread block size for most kernels
// 256 is a good default: divisible by warp size (32), good occupancy
constexpr int DEFAULT_BLOCK_SIZE = 256;

// For reduction kernels, smaller blocks can sometimes be better
constexpr int REDUCTION_BLOCK_SIZE = 256;

// Maximum qubits supported (limited by GPU memory and indexing)
// At 27 qubits: 2^27 * 16 bytes = 2 GB state vector
// At 30 qubits: 2^30 * 16 bytes = 16 GB state vector
constexpr int MAX_QUBITS = 30;
constexpr int MIN_QUBITS = 1;

// Compute capability target (RTX 4070 = Ada Lovelace = 8.9)
constexpr int TARGET_CC_MAJOR = 8;
constexpr int TARGET_CC_MINOR = 9;

} // namespace cuda_config

// ============================================================================
// CUDA Error Checking Macro
// ============================================================================

// Use this macro to check CUDA API calls
// Throws std::runtime_error on failure with file/line info
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + \
                cudaGetErrorString(err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while(0)

// Use after kernel launches to check for async errors
#define CUDA_CHECK_LAST_ERROR() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA kernel error: ") + \
                cudaGetErrorString(err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while(0)

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Calculate number of thread blocks needed
 * @param n Total number of threads needed
 * @param block_size Threads per block
 * @return Number of blocks to launch
 */
inline int calcBlocks(size_t n, int block_size = cuda_config::DEFAULT_BLOCK_SIZE) {
    return static_cast<int>((n + block_size - 1) / block_size);
}

/**
 * Check if a qubit index is valid
 * @param qubit Qubit index to check
 * @param num_qubits Total number of qubits in system
 * @return true if valid
 */
inline bool isValidQubit(int qubit, int num_qubits) {
    return qubit >= 0 && qubit < num_qubits;
}

/**
 * Check if number of qubits is in valid range
 */
inline bool isValidQubitCount(int num_qubits) {
    return num_qubits >= cuda_config::MIN_QUBITS && 
           num_qubits <= cuda_config::MAX_QUBITS;
}

} // namespace qsim
