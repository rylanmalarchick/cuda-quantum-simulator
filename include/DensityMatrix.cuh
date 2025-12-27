/**
 * @file DensityMatrix.cuh
 * @brief Density matrix representation for mixed quantum state simulation
 * @author Rylan Malarchick
 * @date 2024
 *
 * Implements GPU-accelerated density matrix operations for simulating open
 * quantum systems and noisy quantum circuits. The density matrix formalism
 * allows exact representation of mixed states and decoherence effects that
 * cannot be captured by pure state vectors.
 *
 * The density matrix rho for n qubits is a 2^n x 2^n Hermitian, positive
 * semi-definite matrix with unit trace. Gate operations are applied as:
 *   rho' = U rho U^dagger
 *
 * Noise channels are applied using Kraus operators {K_k}:
 *   rho' = sum_k K_k rho K_k^dagger
 *
 * @note Memory scales as O(4^n), limiting practical simulation to ~14 qubits
 *       on an 8GB GPU versus ~27 qubits for state vector simulation.
 *
 * @see NoiseModel.cuh for noise channel implementations
 *
 * @references
 * - Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and Quantum
 *   Information (10th Anniversary Edition). Cambridge University Press.
 *   Chapter 8: Quantum noise and quantum operations.
 * - Preskill, J. (1998). Lecture Notes for Physics 229: Quantum Information
 *   and Computation. California Institute of Technology.
 */
#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <complex>
#include <stdexcept>
#include "Circuit.hpp"
#include "NoiseModel.cuh"

namespace qsim {

/**
 * DensityMatrix - GPU-resident density matrix for mixed state simulation
 * 
 * The density matrix ρ is a 2^n × 2^n Hermitian matrix stored in row-major order.
 * For n qubits, this requires 2^(2n) complex numbers - significantly more memory
 * than state vector simulation but enables exact noise simulation.
 * 
 * Memory requirements:
 *   n qubits -> 2^(2n) * 16 bytes
 *   5 qubits: 16 KB
 *   10 qubits: 16 MB
 *   12 qubits: 256 MB
 *   14 qubits: 4 GB (practical limit for 8GB GPU)
 * 
 * Gate application: ρ' = U ρ U†
 * Noise channels use Kraus operators: ρ' = Σ_k K_k ρ K_k†
 */
class DensityMatrix {
public:
    /**
     * Create a density matrix for n qubits, initialized to |0...0⟩⟨0...0|
     */
    explicit DensityMatrix(int n_qubits);
    
    /**
     * Create a density matrix from a pure state vector
     * ρ = |ψ⟩⟨ψ|
     */
    DensityMatrix(int n_qubits, const std::vector<std::complex<double>>& pure_state);
    
    ~DensityMatrix() noexcept;
    
    // Prevent copying (GPU resource)
    DensityMatrix(const DensityMatrix&) = delete;
    DensityMatrix& operator=(const DensityMatrix&) = delete;
    
    // Allow moving
    DensityMatrix(DensityMatrix&& other) noexcept;
    DensityMatrix& operator=(DensityMatrix&& other) noexcept;
    
    /**
     * Reset to |0...0⟩⟨0...0| state
     */
    void reset();
    
    /**
     * Initialize from pure state |ψ⟩ -> ρ = |ψ⟩⟨ψ|
     */
    void initFromPureState(const std::vector<std::complex<double>>& state);
    
    /**
     * Initialize to maximally mixed state: ρ = I/2^n
     */
    void initMaximallyMixed();
    
    // Accessors
    int getNumQubits() const { return n_qubits_; }
    size_t getDimension() const { return dim_; }
    size_t getNumElements() const { return dim_ * dim_; }
    size_t getMemoryBytes() const { return getNumElements() * sizeof(cuDoubleComplex); }
    
    /**
     * Get diagonal elements (probabilities for computational basis states)
     */
    std::vector<double> getProbabilities() const;
    
    /**
     * Get the full density matrix (copied from GPU)
     */
    std::vector<std::complex<double>> getMatrix() const;
    
    /**
     * Calculate trace (should be 1 for valid density matrix)
     */
    double trace() const;
    
    /**
     * Calculate purity: Tr(ρ²)
     * Purity = 1 for pure states, 1/2^n for maximally mixed
     */
    double purity() const;
    
    /**
     * Check if state is valid density matrix
     * - Trace = 1
     * - Hermitian
     * - Positive semi-definite (checked via purity bounds)
     */
    bool isValid(double tolerance = 1e-10) const;
    
    /**
     * Get raw device pointer (for kernel access)
     */
    cuDoubleComplex* getDevicePtr() { return d_rho_; }
    const cuDoubleComplex* getDevicePtr() const { return d_rho_; }
    
private:
    int n_qubits_;
    size_t dim_;         // 2^n
    cuDoubleComplex* d_rho_;  // Device memory for density matrix
};


/**
 * DensityMatrixSimulator - Exact simulation with noise using density matrices
 * 
 * Applies unitary gates as: ρ' = U ρ U†
 * Applies noise channels using Kraus operators: ρ' = Σ_k K_k ρ K_k†
 * 
 * This gives exact results for noise simulation (unlike Monte Carlo)
 * but is limited to ~14 qubits due to memory requirements.
 */
class DensityMatrixSimulator {
public:
    /**
     * Create simulator with optional noise model
     */
    explicit DensityMatrixSimulator(int n_qubits, const NoiseModel& noise = NoiseModel());
    
    ~DensityMatrixSimulator() noexcept;
    
    /**
     * Reset to |0...0⟩⟨0...0|
     */
    void reset();
    
    /**
     * Run a circuit on the density matrix
     */
    void run(const Circuit& circuit);
    
    /**
     * Apply a single gate
     */
    void applyGate(const GateOp& gate);
    
    /**
     * Get measurement probabilities
     */
    std::vector<double> getProbabilities() const;
    
    /**
     * Get the full density matrix
     */
    std::vector<std::complex<double>> getDensityMatrix() const;
    
    /**
     * Get purity of current state
     */
    double getPurity() const;
    
    /**
     * Get trace (should be 1)
     */
    double getTrace() const;
    
    /**
     * Measure a qubit (partial trace and state update)
     * Returns 0 or 1, updates density matrix
     */
    int measureQubit(int qubit);
    
    // Accessors
    int getNumQubits() const { return n_qubits_; }
    
private:
    void applyNoiseChannel(int target);
    void applyDepolarizing(int target, double p);
    void applyAmplitudeDamping(int target, double gamma);
    void applyPhaseDamping(int target, double gamma);
    void applyBitFlip(int target, double p);
    void applyPhaseFlip(int target, double p);
    void applyBitPhaseFlip(int target, double p);
    
    int n_qubits_;
    DensityMatrix rho_;
    NoiseModel noise_model_;
    cuDoubleComplex* d_scratch_;  // Scratch space for gate application
};


// ============================================================================
// Density Matrix Gate Kernels
// ============================================================================

// Single-qubit gate application: ρ' = U ρ U†
// For density matrix, we need to apply U to rows and U† to columns
__global__ void dmApplyX(cuDoubleComplex* rho, int n_qubits, int target);
__global__ void dmApplyY(cuDoubleComplex* rho, int n_qubits, int target);
__global__ void dmApplyZ(cuDoubleComplex* rho, int n_qubits, int target);
__global__ void dmApplyH(cuDoubleComplex* rho, int n_qubits, int target);
__global__ void dmApplyS(cuDoubleComplex* rho, int n_qubits, int target);
__global__ void dmApplyT(cuDoubleComplex* rho, int n_qubits, int target);
__global__ void dmApplySdag(cuDoubleComplex* rho, int n_qubits, int target);
__global__ void dmApplyTdag(cuDoubleComplex* rho, int n_qubits, int target);
__global__ void dmApplyRx(cuDoubleComplex* rho, int n_qubits, int target, double theta);
__global__ void dmApplyRy(cuDoubleComplex* rho, int n_qubits, int target, double theta);
__global__ void dmApplyRz(cuDoubleComplex* rho, int n_qubits, int target, double theta);

// Two-qubit gates
__global__ void dmApplyCNOT(cuDoubleComplex* rho, int n_qubits, int control, int target);
__global__ void dmApplyCZ(cuDoubleComplex* rho, int n_qubits, int control, int target);
__global__ void dmApplySWAP(cuDoubleComplex* rho, int n_qubits, int qubit1, int qubit2);

// Noise channel Kraus operators
// Depolarizing: ρ' = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
__global__ void dmApplyDepolarizing(cuDoubleComplex* rho, int n_qubits, int target, double p);

// Amplitude damping: K0 = [[1,0],[0,√(1-γ)]], K1 = [[0,√γ],[0,0]]
__global__ void dmApplyAmplitudeDamping(cuDoubleComplex* rho, int n_qubits, int target, double gamma);

// Phase damping: K0 = [[1,0],[0,√(1-γ)]], K1 = [[0,0],[0,√γ]]
__global__ void dmApplyPhaseDamping(cuDoubleComplex* rho, int n_qubits, int target, double gamma);

// Bit flip: ρ' = (1-p)ρ + p*XρX
__global__ void dmApplyBitFlip(cuDoubleComplex* rho, int n_qubits, int target, double p);

// Phase flip: ρ' = (1-p)ρ + p*ZρZ
__global__ void dmApplyPhaseFlip(cuDoubleComplex* rho, int n_qubits, int target, double p);

// Helper kernels
__global__ void dmComputeDiagonal(const cuDoubleComplex* rho, double* diag, size_t dim);
__global__ void dmComputeTrace(const cuDoubleComplex* rho, double* trace, size_t dim);
__global__ void dmInitPure(cuDoubleComplex* rho, const cuDoubleComplex* state, size_t dim);
__global__ void dmInitMaxMixed(cuDoubleComplex* rho, size_t dim, double val);
__global__ void dmCollapseMeasurement(cuDoubleComplex* rho, int n_qubits, int target, 
                                       int result, double norm_factor);

} // namespace qsim
