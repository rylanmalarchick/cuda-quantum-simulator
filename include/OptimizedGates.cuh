#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>

namespace qsim {

/**
 * Optimized Gate Kernels for Quantum Simulation
 * 
 * These kernels use two key optimizations:
 * 
 * 1. COALESCED MEMORY ACCESS
 *    - Threads in a warp access consecutive memory locations
 *    - For gates on high-order qubits, we reorganize the computation
 *    - For gates on low-order qubits, natural indexing is already coalesced
 * 
 * 2. SHARED MEMORY TILING
 *    - Load tiles of the state vector into shared memory
 *    - Perform gate operations in shared memory (much faster)
 *    - Write results back to global memory
 *    - Best for small target qubit indices where pairs are close together
 * 
 * The optimization strategy depends on the target qubit:
 * - Low qubits (0-4): Pairs are close in memory, use shared memory tiling
 * - High qubits (5+): Pairs are far apart, focus on coalesced access patterns
 */

// Block size for optimized kernels (must be power of 2)
constexpr int OPT_BLOCK_SIZE = 256;

// Threshold qubit index: below this, use shared memory; above, use coalesced pattern
constexpr int SHARED_MEM_QUBIT_THRESHOLD = 5;

// ============================================================================
// Optimized Single-Qubit Gates
// ============================================================================

/**
 * Optimized Hadamard gate using shared memory for low target qubits
 * and coalesced access for high target qubits.
 */
__global__ void applyH_opt(cuDoubleComplex* state, int n_qubits, int target);

/**
 * Optimized X gate
 */
__global__ void applyX_opt(cuDoubleComplex* state, int n_qubits, int target);

/**
 * Optimized general single-qubit gate
 * Matrix is provided as 4 complex numbers: [a, b, c, d] for [[a,b],[c,d]]
 */
__global__ void applyGate1Q_opt(cuDoubleComplex* state, int n_qubits, int target,
                                 cuDoubleComplex a, cuDoubleComplex b,
                                 cuDoubleComplex c, cuDoubleComplex d);

// ============================================================================
// Optimized Two-Qubit Gates
// ============================================================================

/**
 * Optimized CNOT using coalesced access patterns
 */
__global__ void applyCNOT_opt(cuDoubleComplex* state, int n_qubits, int control, int target);

// ============================================================================
// Shared Memory Tiled Kernels (for low-order qubits)
// ============================================================================

/**
 * Hadamard with shared memory tiling
 * Each block processes a tile of state vector elements
 * Works best when target qubit < log2(blockSize)
 */
__global__ void applyH_shared(cuDoubleComplex* state, int n_qubits, int target);

/**
 * General rotation gate with shared memory
 * Uses tiling for efficient memory access when target is a low-order qubit
 */
__global__ void applyRotation_shared(cuDoubleComplex* state, int n_qubits, int target,
                                      double cos_half, double sin_half, bool is_rx);

// ============================================================================
// Coalesced Access Pattern Kernels (for high-order qubits)
// ============================================================================

/**
 * Hadamard with coalesced access for high-order target qubits
 * Reorganizes thread indexing to ensure consecutive threads access consecutive memory
 */
__global__ void applyH_coalesced(cuDoubleComplex* state, int n_qubits, int target);

/**
 * General single-qubit gate with coalesced access
 */
__global__ void applyGate1Q_coalesced(cuDoubleComplex* state, int n_qubits, int target,
                                       cuDoubleComplex a, cuDoubleComplex b,
                                       cuDoubleComplex c, cuDoubleComplex d);

// ============================================================================
// Multi-Gate Fusion (apply multiple gates in one kernel launch)
// ============================================================================

/**
 * Fused layer of single-qubit gates
 * Applies one gate per qubit in a single kernel launch
 * Reduces kernel launch overhead for layers of parallel gates
 * 
 * gate_params: array of 4 complex numbers per qubit (2x2 matrix elements)
 *              [qubit0_a, qubit0_b, qubit0_c, qubit0_d, qubit1_a, ...]
 * active_qubits: bitmask indicating which qubits have gates
 */
__global__ void applyFusedSingleQubitLayer(cuDoubleComplex* state, int n_qubits,
                                            const cuDoubleComplex* gate_params,
                                            unsigned int active_qubits);

// ============================================================================
// Dispatcher Functions (choose best kernel based on parameters)
// ============================================================================

/**
 * Apply Hadamard gate using the most efficient kernel for the given target qubit
 */
void applyHadamardOptimized(cuDoubleComplex* state, int n_qubits, int target, cudaStream_t stream = 0);

/**
 * Apply CNOT using the most efficient kernel
 */
void applyCNOTOptimized(cuDoubleComplex* state, int n_qubits, int control, int target, cudaStream_t stream = 0);

} // namespace qsim
