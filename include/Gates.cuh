#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>

namespace qsim {

/**
 * Gate Kernels for Quantum Simulation
 * 
 * Gate Application Strategy:
 * For a gate on qubit q in an n-qubit system:
 * - State vector has 2^n elements
 * - Each single-qubit gate affects pairs of elements differing only in bit q
 * - We launch 2^(n-1) threads, each handling one pair
 * 
 * Index Calculation for Pairs:
 * Given thread index idx, the pair indices are:
 *   i0 = insert 0 at bit position q in idx
 *   i1 = insert 1 at bit position q in idx
 * 
 * For two-qubit gates (e.g., CNOT), we work with groups of 4 elements.
 */

// ============================================================================
// Single-Qubit Gates
// ============================================================================

// Pauli gates
__global__ void applyX(cuDoubleComplex* state, int n_qubits, int target);
__global__ void applyY(cuDoubleComplex* state, int n_qubits, int target);
__global__ void applyZ(cuDoubleComplex* state, int n_qubits, int target);

// Hadamard
__global__ void applyH(cuDoubleComplex* state, int n_qubits, int target);

// Phase gates
__global__ void applyS(cuDoubleComplex* state, int n_qubits, int target);
__global__ void applyT(cuDoubleComplex* state, int n_qubits, int target);
__global__ void applySdag(cuDoubleComplex* state, int n_qubits, int target);
__global__ void applyTdag(cuDoubleComplex* state, int n_qubits, int target);

// Rotation gates
__global__ void applyRx(cuDoubleComplex* state, int n_qubits, int target, double theta);
__global__ void applyRy(cuDoubleComplex* state, int n_qubits, int target, double theta);
__global__ void applyRz(cuDoubleComplex* state, int n_qubits, int target, double theta);

// ============================================================================
// Two-Qubit Gates
// ============================================================================

// Controlled-X (CNOT)
__global__ void applyCNOT(cuDoubleComplex* state, int n_qubits, int control, int target);

// Controlled-Z
__global__ void applyCZ(cuDoubleComplex* state, int n_qubits, int control, int target);

// SWAP
__global__ void applySWAP(cuDoubleComplex* state, int n_qubits, int qubit1, int qubit2);

// ============================================================================
// Helper Functions
// ============================================================================

// Calculate launch configuration
inline void getKernelConfig(size_t n_elements, int& blocks, int& threads) {
    threads = 256;  // Typical good choice for most GPUs
    blocks = (n_elements + threads - 1) / threads;
}

} // namespace qsim
