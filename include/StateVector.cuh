// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Rylan Malarchick

/**
 * @file StateVector.cuh
 * @brief GPU-resident quantum state vector for quantum circuit simulation
 * @author Rylan Malarchick
 * @date 2024
 * 
 * This file implements a state vector representation for simulating quantum
 * circuits on NVIDIA GPUs using CUDA. The state vector stores 2^n complex
 * amplitudes representing n-qubit quantum states.
 * 
 * @references
 * - Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and Quantum
 *   Information (10th Anniversary Edition). Cambridge University Press.
 *   ISBN: 978-1107002173
 * - Preskill, J. (1998). Lecture Notes for Physics 229: Quantum Information
 *   and Computation. California Institute of Technology.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <complex>
#include <cstddef>

namespace qsim {

/**
 * StateVector - GPU-resident quantum state vector
 * 
 * Represents the state of n qubits as 2^n complex amplitudes.
 * Index i corresponds to computational basis state |i⟩.
 * 
 * QUBIT ORDERING CONVENTION:
 * --------------------------
 * This simulator uses BIG-ENDIAN (most significant qubit first) ordering:
 * - Qubit 0 is the MOST significant bit (leftmost in ket notation)
 * - Qubit n-1 is the LEAST significant bit (rightmost in ket notation)
 * 
 * For a 3-qubit system with state |q0 q1 q2⟩:
 *   Index = q0 * 4 + q1 * 2 + q2 * 1 = q0 * 2^2 + q1 * 2^1 + q2 * 2^0
 * 
 * Example (3 qubits):
 *   Index:  0     1     2     3     4     5     6     7
 *   State: |000⟩ |001⟩ |010⟩ |011⟩ |100⟩ |101⟩ |110⟩ |111⟩
 *   q0:     0     0     0     0     1     1     1     1
 *   q1:     0     0     1     1     0     0     1     1
 *   q2:     0     1     0     1     0     1     0     1
 * 
 * This matches Cirq's default convention. Qiskit uses little-endian
 * (reversed) ordering by default, so comparisons require reordering.
 * 
 * For CNOT(control, target):
 *   - control is the qubit that controls the operation
 *   - target is the qubit that gets flipped when control is |1⟩
 * 
 * Memory: 2^n * sizeof(cuDoubleComplex) = 2^n * 16 bytes
 *   20 qubits = 16 MB
 *   25 qubits = 512 MB
 *   27 qubits = 2 GB (max for 8GB GPU with headroom)
 */
class StateVector {
public:
    // Constructors/Destructor
    explicit StateVector(int num_qubits);
    ~StateVector();
    
    // Disable copy (GPU resource management)
    StateVector(const StateVector&) = delete;
    StateVector& operator=(const StateVector&) = delete;
    
    // Enable move
    StateVector(StateVector&& other) noexcept;
    StateVector& operator=(StateVector&& other) noexcept;
    
    // Initialization
    void initializeZero();                    // |0...0⟩
    void initializeBasis(size_t basis_idx);   // |basis_idx⟩
    
    // State access
    int getNumQubits() const { return num_qubits_; }
    size_t getSize() const { return size_; }
    
    // GPU memory access (for kernel launches)
    cuDoubleComplex* devicePtr() { return d_state_; }
    const cuDoubleComplex* devicePtr() const { return d_state_; }
    
    // Copy to host for inspection/validation
    std::vector<std::complex<double>> toHost() const;
    
    // Probability calculations
    std::vector<double> getProbabilities() const;
    double getTotalProbability() const;  // Should be 1.0 (debugging)
    
    /**
     * Check if state is properly normalized (sum of |amplitude|^2 = 1)
     * @param tolerance Maximum allowed deviation from 1.0
     * @return true if normalized within tolerance
     */
    bool isNormalized(double tolerance = 1e-10) const;
    
    /**
     * Assert state is normalized, throw if not
     * @param tolerance Maximum allowed deviation from 1.0
     * @throws std::runtime_error if not normalized
     */
    void assertNormalized(double tolerance = 1e-10) const;
    
    // Measurement
    int measure(int qubit);  // Measures single qubit, collapses state
    std::vector<int> sample(int n_shots);  // Sample without collapse
    
private:
    int num_qubits_;
    size_t size_;              // 2^num_qubits
    cuDoubleComplex* d_state_; // Device pointer
    
    void allocate();
    void deallocate();
};

// ============================================================================
// CUDA Kernels (declared here, defined in StateVector.cu)
// ============================================================================

// Initialize state to |0...0⟩
__global__ void initializeZeroKernel(cuDoubleComplex* state, size_t size);

// Initialize state to |idx⟩
__global__ void initializeBasisKernel(cuDoubleComplex* state, size_t size, size_t basis_idx);

// Calculate |amplitude|^2 for each element (for probability extraction)
__global__ void probabilityKernel(const cuDoubleComplex* state, double* probs, size_t size);

// Parallel reduction to sum probabilities
__global__ void sumReductionKernel(double* data, size_t size);

// Calculate probability of |0⟩ for a specific qubit (for measurement)
__global__ void qubitProbabilityKernel(const cuDoubleComplex* state, double* probs,
                                        size_t size, int num_qubits, int qubit);

// Collapse state after measurement
__global__ void collapseStateKernel(cuDoubleComplex* state, size_t size,
                                     int num_qubits, int qubit, int result,
                                     double normalization_factor);

} // namespace qsim
