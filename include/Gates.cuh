/**
 * @file Gates.cuh
 * @brief CUDA kernels for quantum gate operations on state vectors
 * @author Rylan Malarchick
 * @date 2024
 * 
 * Implements CUDA kernels for applying quantum gates to GPU-resident state
 * vectors. Each gate is implemented as a parallel kernel operating on pairs
 * (for 1-qubit gates) or groups (for multi-qubit gates) of amplitudes.
 * 
 * Gate matrices follow the standard definitions from Nielsen & Chuang:
 * - Pauli gates: X, Y, Z (Section 1.2)
 * - Hadamard: H = (1/√2)[[1,1],[1,-1]] (Section 1.2)
 * - Phase gates: S = [[1,0],[0,i]], T = [[1,0],[0,e^(iπ/4)]] (Section 4.2)
 * - Rotation gates: Rx(θ), Ry(θ), Rz(θ) (Section 4.2)
 * - Controlled gates: CNOT, CZ, CRY, CRZ (Section 4.3)
 * - Three-qubit gates: Toffoli/CCX (Section 4.3)
 * 
 * @references
 * - Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and Quantum
 *   Information (10th Anniversary Edition). Cambridge University Press.
 * - Barenco, A., et al. (1995). Elementary gates for quantum computation.
 *   Physical Review A, 52(5), 3457. arXiv:quant-ph/9503016
 */

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

// Controlled-Ry (parameterized)
__global__ void applyCRY(cuDoubleComplex* state, int n_qubits, int control, int target, double theta);

// Controlled-Rz (parameterized)
__global__ void applyCRZ(cuDoubleComplex* state, int n_qubits, int control, int target, double theta);

// SWAP
__global__ void applySWAP(cuDoubleComplex* state, int n_qubits, int qubit1, int qubit2);

// ============================================================================
// Three-Qubit Gates
// ============================================================================

// Toffoli (CCX) - Controlled-Controlled-X
__global__ void applyToffoli(cuDoubleComplex* state, int n_qubits, 
                              int control1, int control2, int target);

// ============================================================================
// Helper Functions
// ============================================================================

// Calculate launch configuration
inline void getKernelConfig(size_t n_elements, int& blocks, int& threads) {
    threads = 256;  // Typical good choice for most GPUs
    blocks = (n_elements + threads - 1) / threads;
}

} // namespace qsim
