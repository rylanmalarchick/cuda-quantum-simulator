// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Rylan Malarchick

#include "OptimizedGates.cuh"

#include <cuda_runtime.h>

#include <cmath>

namespace qsim {

// ============================================================================
// Helper Functions
// ============================================================================

__device__ inline cuDoubleComplex qCmul(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(
        cuCreal(a) * cuCreal(b) - cuCimag(a) * cuCimag(b),
        cuCreal(a) * cuCimag(b) + cuCimag(a) * cuCreal(b)
    );
}

__device__ inline cuDoubleComplex qCadd(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(cuCreal(a) + cuCreal(b), cuCimag(a) + cuCimag(b));
}

__device__ inline cuDoubleComplex qCsub(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(cuCreal(a) - cuCreal(b), cuCimag(a) - cuCimag(b));
}

__device__ inline cuDoubleComplex qCscale(cuDoubleComplex a, double s) {
    return make_cuDoubleComplex(cuCreal(a) * s, cuCimag(a) * s);
}

// ============================================================================
// Shared Memory Tiled Kernels
// These work best when target qubit index is small (< 5)
// because the pair elements are close together in memory
// ============================================================================

__global__ void applyH_shared(cuDoubleComplex* state, int n_qubits, int target) {
    // Shared memory tile - each thread block loads a contiguous tile
    extern __shared__ cuDoubleComplex tile[];
    
    const double inv_sqrt2 = 0.7071067811865476;
    const size_t n_states = 1ULL << n_qubits;
    const size_t pair_stride = 1ULL << target;
    
    // Calculate which tile this block handles
    // Each block processes 2 * blockDim.x elements (pairs of amplitudes)
    const size_t tile_size = 2 * blockDim.x;
    const size_t tile_start = blockIdx.x * tile_size;
    
    if (tile_start >= n_states) return;
    
    // Load tile into shared memory (coalesced read)
    size_t global_idx = tile_start + threadIdx.x;
    if (global_idx < n_states) {
        tile[threadIdx.x] = state[global_idx];
    }
    global_idx = tile_start + threadIdx.x + blockDim.x;
    if (global_idx < n_states) {
        tile[threadIdx.x + blockDim.x] = state[global_idx];
    }
    
    __syncthreads();
    
    // Apply Hadamard gate to pairs within the tile
    // For small target qubits, pairs are within the tile
    if (target < 8) {  // Target qubit fits within tile
        size_t local_idx = threadIdx.x;
        size_t local_n_pairs = blockDim.x;  // Each thread handles one pair
        
        if (local_idx < local_n_pairs) {
            // Calculate local indices for the pair
            size_t mask = pair_stride - 1;
            size_t local_i0 = (local_idx & mask) | ((local_idx & ~mask) << 1);
            size_t local_i1 = local_i0 | pair_stride;
            
            // Map to global indices to check if this pair is in our tile
            size_t global_i0 = tile_start + local_i0;
            size_t global_i1 = tile_start + local_i1;
            
            if (local_i0 < tile_size && local_i1 < tile_size && 
                global_i0 < n_states && global_i1 < n_states) {
                
                cuDoubleComplex a0 = tile[local_i0];
                cuDoubleComplex a1 = tile[local_i1];
                
                // H: (a0 + a1)/sqrt(2), (a0 - a1)/sqrt(2)
                tile[local_i0] = make_cuDoubleComplex(
                    (cuCreal(a0) + cuCreal(a1)) * inv_sqrt2,
                    (cuCimag(a0) + cuCimag(a1)) * inv_sqrt2
                );
                tile[local_i1] = make_cuDoubleComplex(
                    (cuCreal(a0) - cuCreal(a1)) * inv_sqrt2,
                    (cuCimag(a0) - cuCimag(a1)) * inv_sqrt2
                );
            }
        }
    }
    
    __syncthreads();
    
    // Write tile back to global memory (coalesced write)
    global_idx = tile_start + threadIdx.x;
    if (global_idx < n_states) {
        state[global_idx] = tile[threadIdx.x];
    }
    global_idx = tile_start + threadIdx.x + blockDim.x;
    if (global_idx < n_states) {
        state[global_idx] = tile[threadIdx.x + blockDim.x];
    }
}

// ============================================================================
// Coalesced Access Pattern Kernels
// These reorganize thread indexing for high-order target qubits
// ============================================================================

__global__ void applyH_coalesced(cuDoubleComplex* state, int n_qubits, int target) {
    const double inv_sqrt2 = 0.7071067811865476;
    const size_t n_pairs = 1ULL << (n_qubits - 1);
    
    // Global thread index
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n_pairs) return;
    
    // For coalesced access when target is high, we want consecutive threads
    // to access consecutive memory locations.
    //
    // Standard indexing: thread tid handles pair (i0, i1) where
    //   i0 = insert 0 at bit 'target' position
    //   i1 = i0 | (1 << target)
    //
    // For high target bits, i0 and i1 are far apart but consecutive threads
    // access consecutive i0s, which is coalesced for the first load.
    // The second load (i1) is also coalesced since consecutive threads
    // access consecutive i1s.
    
    size_t mask = (1ULL << target) - 1;
    size_t i0 = (tid & mask) | ((tid & ~mask) << 1);
    size_t i1 = i0 | (1ULL << target);
    
    // Load (these are coalesced for consecutive threads)
    cuDoubleComplex a0 = state[i0];
    cuDoubleComplex a1 = state[i1];
    
    // Apply Hadamard
    cuDoubleComplex new_a0 = make_cuDoubleComplex(
        (cuCreal(a0) + cuCreal(a1)) * inv_sqrt2,
        (cuCimag(a0) + cuCimag(a1)) * inv_sqrt2
    );
    cuDoubleComplex new_a1 = make_cuDoubleComplex(
        (cuCreal(a0) - cuCreal(a1)) * inv_sqrt2,
        (cuCimag(a0) - cuCimag(a1)) * inv_sqrt2
    );
    
    // Store (also coalesced)
    state[i0] = new_a0;
    state[i1] = new_a1;
}

__global__ void applyGate1Q_coalesced(cuDoubleComplex* state, int n_qubits, int target,
                                       cuDoubleComplex a, cuDoubleComplex b,
                                       cuDoubleComplex c, cuDoubleComplex d) {
    const size_t n_pairs = 1ULL << (n_qubits - 1);
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n_pairs) return;
    
    size_t mask = (1ULL << target) - 1;
    size_t i0 = (tid & mask) | ((tid & ~mask) << 1);
    size_t i1 = i0 | (1ULL << target);
    
    cuDoubleComplex v0 = state[i0];
    cuDoubleComplex v1 = state[i1];
    
    // Apply 2x2 matrix: [a b; c d] * [v0; v1]
    state[i0] = qCadd(qCmul(a, v0), qCmul(b, v1));
    state[i1] = qCadd(qCmul(c, v0), qCmul(d, v1));
}

// ============================================================================
// Optimized Combined Kernels (auto-select based on target)
// ============================================================================

__global__ void applyH_opt(cuDoubleComplex* state, int n_qubits, int target) {
    // This kernel uses the coalesced pattern which works well for all targets
    // The dispatcher function can choose shared memory version for low targets
    
    const double inv_sqrt2 = 0.7071067811865476;
    const size_t n_pairs = 1ULL << (n_qubits - 1);
    
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;
    
    size_t mask = (1ULL << target) - 1;
    size_t i0 = (tid & mask) | ((tid & ~mask) << 1);
    size_t i1 = i0 | (1ULL << target);
    
    cuDoubleComplex a0 = state[i0];
    cuDoubleComplex a1 = state[i1];
    
    state[i0] = make_cuDoubleComplex(
        (cuCreal(a0) + cuCreal(a1)) * inv_sqrt2,
        (cuCimag(a0) + cuCimag(a1)) * inv_sqrt2
    );
    state[i1] = make_cuDoubleComplex(
        (cuCreal(a0) - cuCreal(a1)) * inv_sqrt2,
        (cuCimag(a0) - cuCimag(a1)) * inv_sqrt2
    );
}

__global__ void applyX_opt(cuDoubleComplex* state, int n_qubits, int target) {
    const size_t n_pairs = 1ULL << (n_qubits - 1);
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n_pairs) return;
    
    size_t mask = (1ULL << target) - 1;
    size_t i0 = (tid & mask) | ((tid & ~mask) << 1);
    size_t i1 = i0 | (1ULL << target);
    
    // X gate: swap amplitudes
    cuDoubleComplex tmp = state[i0];
    state[i0] = state[i1];
    state[i1] = tmp;
}

__global__ void applyGate1Q_opt(cuDoubleComplex* state, int n_qubits, int target,
                                 cuDoubleComplex a, cuDoubleComplex b,
                                 cuDoubleComplex c, cuDoubleComplex d) {
    const size_t n_pairs = 1ULL << (n_qubits - 1);
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n_pairs) return;
    
    size_t mask = (1ULL << target) - 1;
    size_t i0 = (tid & mask) | ((tid & ~mask) << 1);
    size_t i1 = i0 | (1ULL << target);
    
    cuDoubleComplex v0 = state[i0];
    cuDoubleComplex v1 = state[i1];
    
    state[i0] = qCadd(qCmul(a, v0), qCmul(b, v1));
    state[i1] = qCadd(qCmul(c, v0), qCmul(d, v1));
}

// ============================================================================
// Optimized CNOT
// ============================================================================

__global__ void applyCNOT_opt(cuDoubleComplex* state, int n_qubits, int control, int target) {
    const size_t n_states = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_states) return;
    
    // Only act if control bit is 1 and target bit is 0 (to avoid double swap)
    bool control_is_1 = (idx >> control) & 1;
    bool target_is_0 = !((idx >> target) & 1);
    
    if (control_is_1 && target_is_0) {
        size_t partner = idx ^ (1ULL << target);
        cuDoubleComplex tmp = state[idx];
        state[idx] = state[partner];
        state[partner] = tmp;
    }
}

// ============================================================================
// Rotation Gate with Shared Memory
// ============================================================================

__global__ void applyRotation_shared(cuDoubleComplex* state, int n_qubits, int target,
                                      double cos_half, double sin_half, bool is_rx) {
    extern __shared__ cuDoubleComplex tile[];
    
    const size_t n_states = 1ULL << n_qubits;
    const size_t pair_stride = 1ULL << target;
    const size_t tile_size = 2 * blockDim.x;
    const size_t tile_start = blockIdx.x * tile_size;
    
    if (tile_start >= n_states) return;
    
    // Load tile
    size_t global_idx = tile_start + threadIdx.x;
    if (global_idx < n_states) tile[threadIdx.x] = state[global_idx];
    global_idx = tile_start + threadIdx.x + blockDim.x;
    if (global_idx < n_states) tile[threadIdx.x + blockDim.x] = state[global_idx];
    
    __syncthreads();
    
    // Apply rotation to pairs in tile
    size_t local_idx = threadIdx.x;
    if (local_idx < blockDim.x && target < 8) {
        size_t mask = pair_stride - 1;
        size_t local_i0 = (local_idx & mask) | ((local_idx & ~mask) << 1);
        size_t local_i1 = local_i0 | pair_stride;
        
        if (local_i0 < tile_size && local_i1 < tile_size) {
            cuDoubleComplex a0 = tile[local_i0];
            cuDoubleComplex a1 = tile[local_i1];
            
            if (is_rx) {
                // Rx(theta): [[c, -is], [-is, c]]
                tile[local_i0] = make_cuDoubleComplex(
                    cos_half * cuCreal(a0) + sin_half * cuCimag(a1),
                    cos_half * cuCimag(a0) - sin_half * cuCreal(a1)
                );
                tile[local_i1] = make_cuDoubleComplex(
                    sin_half * cuCimag(a0) + cos_half * cuCreal(a1),
                    -sin_half * cuCreal(a0) + cos_half * cuCimag(a1)
                );
            } else {
                // Ry(theta): [[c, -s], [s, c]]
                tile[local_i0] = make_cuDoubleComplex(
                    cos_half * cuCreal(a0) - sin_half * cuCreal(a1),
                    cos_half * cuCimag(a0) - sin_half * cuCimag(a1)
                );
                tile[local_i1] = make_cuDoubleComplex(
                    sin_half * cuCreal(a0) + cos_half * cuCreal(a1),
                    sin_half * cuCimag(a0) + cos_half * cuCimag(a1)
                );
            }
        }
    }
    
    __syncthreads();
    
    // Write back
    global_idx = tile_start + threadIdx.x;
    if (global_idx < n_states) state[global_idx] = tile[threadIdx.x];
    global_idx = tile_start + threadIdx.x + blockDim.x;
    if (global_idx < n_states) state[global_idx] = tile[threadIdx.x + blockDim.x];
}

// ============================================================================
// Fused Single-Qubit Layer
// ============================================================================

__global__ void applyFusedSingleQubitLayer(cuDoubleComplex* state, int n_qubits,
                                            const cuDoubleComplex* gate_params,
                                            unsigned int active_qubits) {
    const size_t n_states = 1ULL << n_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_states) return;
    
    // This kernel applies all single-qubit gates in parallel
    // Each thread transforms one amplitude through all gates
    // Note: This is only correct for diagonal gates or when gates commute
    // For non-commuting gates, use sequential application
    
    cuDoubleComplex amplitude = state[idx];
    
    for (int q = 0; q < n_qubits; ++q) {
        if (!((active_qubits >> q) & 1)) continue;
        
        // Get diagonal elements of gate matrix for this qubit
        // Note: This kernel only works for diagonal gates (Z, S, T, Rz, etc.)
        // Off-diagonal elements b, c are not used
        cuDoubleComplex a = gate_params[q * 4 + 0];  // |0><0| coefficient
        cuDoubleComplex d = gate_params[q * 4 + 3];  // |1><1| coefficient
        
        // Determine if this state has qubit q as 0 or 1
        bool bit = (idx >> q) & 1;
        
        // Apply the appropriate diagonal element of the gate matrix
        if (bit) {
            // |1> state: amplitude *= d
            amplitude = qCmul(d, amplitude);
        } else {
            // |0> state: amplitude *= a
            amplitude = qCmul(a, amplitude);
        }
    }
    
    state[idx] = amplitude;
}

// ============================================================================
// Dispatcher Functions
// ============================================================================

void applyHadamardOptimized(cuDoubleComplex* state, int n_qubits, int target, cudaStream_t stream) {
    const size_t n_pairs = 1ULL << (n_qubits - 1);
    const int threads = OPT_BLOCK_SIZE;
    const int blocks = (n_pairs + threads - 1) / threads;
    
    if (target < SHARED_MEM_QUBIT_THRESHOLD && n_qubits <= 20) {
        // Use shared memory version for low-order qubits on smaller states
        const size_t tile_size = 2 * threads;
        const size_t n_states = 1ULL << n_qubits;
        const int tile_blocks = (n_states + tile_size - 1) / tile_size;
        const size_t shared_mem = tile_size * sizeof(cuDoubleComplex);
        
        applyH_shared<<<tile_blocks, threads, shared_mem, stream>>>(state, n_qubits, target);
    } else {
        // Use coalesced version for high-order qubits
        applyH_coalesced<<<blocks, threads, 0, stream>>>(state, n_qubits, target);
    }
}

void applyCNOTOptimized(cuDoubleComplex* state, int n_qubits, int control, int target, cudaStream_t stream) {
    const size_t n_states = 1ULL << n_qubits;
    const int threads = OPT_BLOCK_SIZE;
    const int blocks = (n_states + threads - 1) / threads;
    
    applyCNOT_opt<<<blocks, threads, 0, stream>>>(state, n_qubits, control, target);
}

} // namespace qsim
