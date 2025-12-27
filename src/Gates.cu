#include "Gates.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace qsim {

// ============================================================================
// Helper: Index calculation for gate application
// ============================================================================

// For single-qubit gates on qubit 'target':
// Given a pair index 'idx', compute the two state indices where bit 'target' is 0 and 1.
// The pair index goes from 0 to 2^(n-1)-1
__device__ inline void getPairIndices(size_t idx, int target, size_t& i0, size_t& i1) {
    // Insert 0 at bit position 'target'
    // Lower bits stay the same, upper bits shift left by 1
    size_t mask = (1ULL << target) - 1;
    i0 = (idx & mask) | ((idx & ~mask) << 1);
    i1 = i0 | (1ULL << target);
}

// ============================================================================
// Single-Qubit Gates
// ============================================================================

__global__ void applyX(cuDoubleComplex* state, int n_qubits, int target) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    if (idx < n_pairs) {
        size_t i0, i1;
        getPairIndices(idx, target, i0, i1);
        
        // X gate: swap amplitudes
        cuDoubleComplex tmp = state[i0];
        state[i0] = state[i1];
        state[i1] = tmp;
    }
}

__global__ void applyY(cuDoubleComplex* state, int n_qubits, int target) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    if (idx < n_pairs) {
        size_t i0, i1;
        getPairIndices(idx, target, i0, i1);
        
        // Y = [[0, -i], [i, 0]]
        cuDoubleComplex a0 = state[i0];
        cuDoubleComplex a1 = state[i1];
        
        // new_a0 = -i * a1 = (imag(a1), -real(a1))
        // new_a1 = i * a0 = (-imag(a0), real(a0))
        state[i0] = make_cuDoubleComplex(cuCimag(a1), -cuCreal(a1));
        state[i1] = make_cuDoubleComplex(-cuCimag(a0), cuCreal(a0));
    }
}

__global__ void applyZ(cuDoubleComplex* state, int n_qubits, int target) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    if (idx < n_pairs) {
        size_t i0, i1;
        getPairIndices(idx, target, i0, i1);
        
        // Z = [[1, 0], [0, -1]]
        // Only negate the |1⟩ amplitude
        state[i1] = make_cuDoubleComplex(-cuCreal(state[i1]), -cuCimag(state[i1]));
    }
}

__global__ void applyH(cuDoubleComplex* state, int n_qubits, int target) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    if (idx < n_pairs) {
        size_t i0, i1;
        getPairIndices(idx, target, i0, i1);
        
        // H = (1/sqrt(2)) * [[1, 1], [1, -1]]
        const double inv_sqrt2 = 0.7071067811865476;
        
        cuDoubleComplex a0 = state[i0];
        cuDoubleComplex a1 = state[i1];
        
        // new_a0 = (a0 + a1) / sqrt(2)
        // new_a1 = (a0 - a1) / sqrt(2)
        state[i0] = make_cuDoubleComplex(
            (cuCreal(a0) + cuCreal(a1)) * inv_sqrt2,
            (cuCimag(a0) + cuCimag(a1)) * inv_sqrt2
        );
        state[i1] = make_cuDoubleComplex(
            (cuCreal(a0) - cuCreal(a1)) * inv_sqrt2,
            (cuCimag(a0) - cuCimag(a1)) * inv_sqrt2
        );
    }
}

__global__ void applyS(cuDoubleComplex* state, int n_qubits, int target) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    if (idx < n_pairs) {
        size_t i0, i1;
        getPairIndices(idx, target, i0, i1);
        
        // S = [[1, 0], [0, i]]
        // Multiply |1⟩ by i: (a + bi) * i = (-b + ai)
        cuDoubleComplex a1 = state[i1];
        state[i1] = make_cuDoubleComplex(-cuCimag(a1), cuCreal(a1));
    }
}

__global__ void applyT(cuDoubleComplex* state, int n_qubits, int target) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    if (idx < n_pairs) {
        size_t i0, i1;
        getPairIndices(idx, target, i0, i1);
        
        // T = [[1, 0], [0, e^(i*pi/4)]]
        // e^(i*pi/4) = cos(pi/4) + i*sin(pi/4) = (1 + i)/sqrt(2)
        const double inv_sqrt2 = 0.7071067811865476;
        
        cuDoubleComplex a1 = state[i1];
        // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        // where c = d = 1/sqrt(2)
        state[i1] = make_cuDoubleComplex(
            (cuCreal(a1) - cuCimag(a1)) * inv_sqrt2,
            (cuCreal(a1) + cuCimag(a1)) * inv_sqrt2
        );
    }
}

__global__ void applySdag(cuDoubleComplex* state, int n_qubits, int target) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    if (idx < n_pairs) {
        size_t i0, i1;
        getPairIndices(idx, target, i0, i1);
        
        // S† = [[1, 0], [0, -i]]
        cuDoubleComplex a1 = state[i1];
        state[i1] = make_cuDoubleComplex(cuCimag(a1), -cuCreal(a1));
    }
}

__global__ void applyTdag(cuDoubleComplex* state, int n_qubits, int target) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    if (idx < n_pairs) {
        size_t i0, i1;
        getPairIndices(idx, target, i0, i1);
        
        // T† = [[1, 0], [0, e^(-i*pi/4)]]
        const double inv_sqrt2 = 0.7071067811865476;
        
        cuDoubleComplex a1 = state[i1];
        // e^(-i*pi/4) = (1 - i)/sqrt(2)
        state[i1] = make_cuDoubleComplex(
            (cuCreal(a1) + cuCimag(a1)) * inv_sqrt2,
            (-cuCreal(a1) + cuCimag(a1)) * inv_sqrt2
        );
    }
}

__global__ void applyRx(cuDoubleComplex* state, int n_qubits, int target, double theta) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    if (idx < n_pairs) {
        size_t i0, i1;
        getPairIndices(idx, target, i0, i1);
        
        // Rx(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
        double c = cos(theta / 2.0);
        double s = sin(theta / 2.0);
        
        cuDoubleComplex a0 = state[i0];
        cuDoubleComplex a1 = state[i1];
        
        // new_a0 = c*a0 - i*s*a1
        // new_a1 = -i*s*a0 + c*a1
        state[i0] = make_cuDoubleComplex(
            c * cuCreal(a0) + s * cuCimag(a1),
            c * cuCimag(a0) - s * cuCreal(a1)
        );
        state[i1] = make_cuDoubleComplex(
            s * cuCimag(a0) + c * cuCreal(a1),
            -s * cuCreal(a0) + c * cuCimag(a1)
        );
    }
}

__global__ void applyRy(cuDoubleComplex* state, int n_qubits, int target, double theta) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    if (idx < n_pairs) {
        size_t i0, i1;
        getPairIndices(idx, target, i0, i1);
        
        // Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
        double c = cos(theta / 2.0);
        double s = sin(theta / 2.0);
        
        cuDoubleComplex a0 = state[i0];
        cuDoubleComplex a1 = state[i1];
        
        state[i0] = make_cuDoubleComplex(
            c * cuCreal(a0) - s * cuCreal(a1),
            c * cuCimag(a0) - s * cuCimag(a1)
        );
        state[i1] = make_cuDoubleComplex(
            s * cuCreal(a0) + c * cuCreal(a1),
            s * cuCimag(a0) + c * cuCimag(a1)
        );
    }
}

__global__ void applyRz(cuDoubleComplex* state, int n_qubits, int target, double theta) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    if (idx < n_pairs) {
        size_t i0, i1;
        getPairIndices(idx, target, i0, i1);
        
        // Rz(θ) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
        double half_theta = theta / 2.0;
        double c = cos(half_theta);
        double s = sin(half_theta);
        
        cuDoubleComplex a0 = state[i0];
        cuDoubleComplex a1 = state[i1];
        
        // e^(-iθ/2) = cos(θ/2) - i*sin(θ/2)
        state[i0] = make_cuDoubleComplex(
            c * cuCreal(a0) + s * cuCimag(a0),
            c * cuCimag(a0) - s * cuCreal(a0)
        );
        // e^(iθ/2) = cos(θ/2) + i*sin(θ/2)
        state[i1] = make_cuDoubleComplex(
            c * cuCreal(a1) - s * cuCimag(a1),
            c * cuCimag(a1) + s * cuCreal(a1)
        );
    }
}

// ============================================================================
// Two-Qubit Gates
// ============================================================================

__global__ void applyCNOT(cuDoubleComplex* state, int n_qubits, int control, int target) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_states = 1ULL << n_qubits;
    
    if (idx < n_states) {
        // Only act if control bit is 1 and target bit is 0
        // (to avoid swapping twice)
        bool control_is_1 = (idx >> control) & 1;
        bool target_is_0 = !((idx >> target) & 1);
        
        if (control_is_1 && target_is_0) {
            size_t partner = idx ^ (1ULL << target);
            cuDoubleComplex tmp = state[idx];
            state[idx] = state[partner];
            state[partner] = tmp;
        }
    }
}

__global__ void applyCZ(cuDoubleComplex* state, int n_qubits, int control, int target) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_states = 1ULL << n_qubits;
    
    if (idx < n_states) {
        // Apply -1 phase when both control and target are 1
        bool control_is_1 = (idx >> control) & 1;
        bool target_is_1 = (idx >> target) & 1;
        
        if (control_is_1 && target_is_1) {
            state[idx] = make_cuDoubleComplex(-cuCreal(state[idx]), -cuCimag(state[idx]));
        }
    }
}

__global__ void applySWAP(cuDoubleComplex* state, int n_qubits, int qubit1, int qubit2) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_states = 1ULL << n_qubits;
    
    if (idx < n_states) {
        // Swap when qubits differ (01 <-> 10)
        bool q1_bit = (idx >> qubit1) & 1;
        bool q2_bit = (idx >> qubit2) & 1;
        
        // Only swap when q1=0, q2=1 (to avoid swapping twice)
        if (!q1_bit && q2_bit) {
            // Partner has bits swapped
            size_t partner = idx ^ (1ULL << qubit1) ^ (1ULL << qubit2);
            cuDoubleComplex tmp = state[idx];
            state[idx] = state[partner];
            state[partner] = tmp;
        }
    }
}

// ============================================================================
// Controlled Rotation Gates
// ============================================================================

__global__ void applyCRY(cuDoubleComplex* state, int n_qubits, int control, int target, double theta) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_states = 1ULL << n_qubits;
    
    if (idx < n_states) {
        // Only act when control is 1 and target is 0 (to avoid processing pairs twice)
        bool control_is_1 = (idx >> control) & 1;
        bool target_is_0 = !((idx >> target) & 1);
        
        if (control_is_1 && target_is_0) {
            size_t partner = idx ^ (1ULL << target);
            
            // Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
            double c = cos(theta / 2.0);
            double s = sin(theta / 2.0);
            
            cuDoubleComplex a0 = state[idx];      // target=0
            cuDoubleComplex a1 = state[partner];  // target=1
            
            state[idx] = make_cuDoubleComplex(
                c * cuCreal(a0) - s * cuCreal(a1),
                c * cuCimag(a0) - s * cuCimag(a1)
            );
            state[partner] = make_cuDoubleComplex(
                s * cuCreal(a0) + c * cuCreal(a1),
                s * cuCimag(a0) + c * cuCimag(a1)
            );
        }
    }
}

__global__ void applyCRZ(cuDoubleComplex* state, int n_qubits, int control, int target, double theta) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_states = 1ULL << n_qubits;
    
    if (idx < n_states) {
        // Only act when control is 1
        bool control_is_1 = (idx >> control) & 1;
        
        if (control_is_1) {
            bool target_is_1 = (idx >> target) & 1;
            
            // Rz(θ) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
            double half_theta = theta / 2.0;
            double c = cos(half_theta);
            double s = sin(half_theta);
            
            cuDoubleComplex a = state[idx];
            
            if (target_is_1) {
                // e^(iθ/2) = cos(θ/2) + i*sin(θ/2)
                state[idx] = make_cuDoubleComplex(
                    c * cuCreal(a) - s * cuCimag(a),
                    c * cuCimag(a) + s * cuCreal(a)
                );
            } else {
                // e^(-iθ/2) = cos(θ/2) - i*sin(θ/2)
                state[idx] = make_cuDoubleComplex(
                    c * cuCreal(a) + s * cuCimag(a),
                    c * cuCimag(a) - s * cuCreal(a)
                );
            }
        }
    }
}

// ============================================================================
// Three-Qubit Gates
// ============================================================================

__global__ void applyToffoli(cuDoubleComplex* state, int n_qubits,
                              int control1, int control2, int target) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_states = 1ULL << n_qubits;
    
    if (idx < n_states) {
        // Only act when both controls are 1 and target is 0 (to avoid swapping twice)
        bool c1_is_1 = (idx >> control1) & 1;
        bool c2_is_1 = (idx >> control2) & 1;
        bool target_is_0 = !((idx >> target) & 1);
        
        if (c1_is_1 && c2_is_1 && target_is_0) {
            size_t partner = idx ^ (1ULL << target);
            cuDoubleComplex tmp = state[idx];
            state[idx] = state[partner];
            state[partner] = tmp;
        }
    }
}

} // namespace qsim
