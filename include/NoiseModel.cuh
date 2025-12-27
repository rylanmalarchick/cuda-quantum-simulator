// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Rylan Malarchick

/**
 * @file NoiseModel.cuh
 * @brief Noise models and noisy simulation for quantum circuits
 * @author Rylan Malarchick
 * @date 2024
 * 
 * Implements realistic noise models for quantum circuit simulation using the
 * Monte Carlo wavefunction method (quantum trajectories). This approach
 * maintains state vector representation while applying noise probabilistically.
 * 
 * Noise channels implemented:
 * - Depolarizing: ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
 * - Amplitude damping (T1): Models energy relaxation to ground state
 * - Phase damping (T2): Models loss of phase coherence
 * - Bit flip, Phase flip, Bit-phase flip: Probabilistic Pauli errors
 * 
 * @references
 * - Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and Quantum
 *   Information, Chapter 8: Quantum noise and quantum operations.
 * - Dalibard, J., Castin, Y., & Mølmer, K. (1992). Wave-function approach to
 *   dissipative processes in quantum optics. Physical Review Letters, 68(5), 580.
 * - Carmichael, H. J. (1993). An Open Systems Approach to Quantum Optics.
 *   Lecture Notes in Physics, Vol. 18. Springer-Verlag.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <curand_kernel.h>
#include <vector>
#include <complex>
#include <memory>
#include <random>
#include <algorithm>
#include "CudaMemory.cuh"

namespace qsim {

/**
 * NoiseChannel - Types of noise that can be applied
 */
enum class NoiseType {
    Depolarizing,      // Random Pauli error with probability p
    AmplitudeDamping,  // T1 decay (energy relaxation)
    PhaseDamping,      // T2 dephasing (pure dephasing)
    BitFlip,           // X error with probability p
    PhaseFlip,         // Z error with probability p
    BitPhaseFlip       // Y error with probability p
};

/**
 * NoiseChannel - Represents a single noise channel applied to specific qubits
 */
struct NoiseChannel {
    NoiseType type;
    std::vector<int> qubits;  // Which qubits this noise applies to
    double probability;        // Error probability (interpretation depends on type)
    
    NoiseChannel(NoiseType t, std::vector<int> q, double p)
        : type(t), qubits(q), probability(p) {}
};

/**
 * NoiseModel - Collection of noise channels applied during simulation
 * 
 * Noise can be applied:
 * - After each gate (gate-level noise)
 * - On idle qubits (during a gate on other qubits)
 * - At specific points in the circuit
 * 
 * For state vector simulation, we use the Monte Carlo approach:
 * - For each noise channel, we probabilistically apply one of the Kraus operators
 * - This is equivalent to tracing out the environment in the density matrix formalism
 * 
 * Usage:
 *   NoiseModel noise;
 *   noise.addDepolarizing({0, 1, 2, 3}, 0.01);  // 1% depolarizing on all qubits
 *   noise.addAmplitudeDamping({0}, 0.02);       // T1 decay on qubit 0
 *   
 *   NoisySimulator sim(4, noise);
 *   sim.run(circuit);
 */
class NoiseModel {
public:
    NoiseModel() = default;
    
    // Add noise channels
    void addDepolarizing(const std::vector<int>& qubits, double probability);
    void addAmplitudeDamping(const std::vector<int>& qubits, double gamma);
    void addPhaseDamping(const std::vector<int>& qubits, double gamma);
    void addBitFlip(const std::vector<int>& qubits, double probability);
    void addPhaseFlip(const std::vector<int>& qubits, double probability);
    void addBitPhaseFlip(const std::vector<int>& qubits, double probability);
    
    // Convenience: add global noise (applies to all qubits automatically)
    // These store with empty qubits vector meaning "apply to all"
    void addDepolarizing(double probability);
    void addAmplitudeDamping(double gamma);
    void addPhaseDamping(double gamma);
    void addBitFlip(double probability);
    void addPhaseFlip(double probability);
    void addBitPhaseFlip(double probability);
    
    // Convenience: add same noise to all qubits
    void addDepolarizingAll(int num_qubits, double probability);
    void addAmplitudeDampingAll(int num_qubits, double gamma);
    void addPhaseDampingAll(int num_qubits, double gamma);
    
    // Access channels
    const std::vector<NoiseChannel>& getChannels() const { return channels_; }
    bool hasNoise() const { return !channels_.empty(); }
    void clear() { channels_.clear(); }
    
    // Check if a channel applies to a specific qubit
    bool channelAppliesToQubit(const NoiseChannel& channel, int qubit) const {
        return channel.qubits.empty() || 
               std::find(channel.qubits.begin(), channel.qubits.end(), qubit) != channel.qubits.end();
    }
    
private:
    std::vector<NoiseChannel> channels_;
};

/**
 * NoisySimulator - Simulator with noise support
 * 
 * Uses Monte Carlo wavefunction method:
 * - State remains a pure state (not density matrix)
 * - Noise is applied probabilistically
 * - Multiple runs give the ensemble average
 * 
 * This is memory efficient (O(2^n) vs O(4^n) for density matrix)
 * but requires multiple shots for expectation values.
 */
class NoisySimulator {
public:
    NoisySimulator(int num_qubits, const NoiseModel& noise_model);
    NoisySimulator(int num_qubits);  // No noise (equivalent to regular Simulator)
    ~NoisySimulator() noexcept = default;
    
    // Disable copy, enable move
    NoisySimulator(const NoisySimulator&) = delete;
    NoisySimulator& operator=(const NoisySimulator&) = delete;
    NoisySimulator(NoisySimulator&&) noexcept = default;
    NoisySimulator& operator=(NoisySimulator&&) noexcept = default;
    
    // Set/update noise model
    void setNoiseModel(const NoiseModel& noise_model);
    const NoiseModel& getNoiseModel() const { return noise_model_; }
    
    // Set random seed for reproducibility
    void setSeed(unsigned int seed);
    
    // Reset state to |0...0⟩
    void reset();
    
    // Run circuit with noise applied after each gate
    void run(const class Circuit& circuit);
    
    // Apply single gate (with noise)
    void applyGate(const struct GateOp& gate);
    
    // Apply noise channel directly
    void applyNoise(const NoiseChannel& channel);
    void applyNoiseToQubit(NoiseType type, int qubit, double probability);
    
    // State inspection (same as regular Simulator)
    std::vector<std::complex<double>> getStateVector() const;
    std::vector<double> getProbabilities() const;
    
    // Measurement
    std::vector<int> sample(int n_shots);
    int measureQubit(int qubit);
    
    // Info
    int getNumQubits() const { return num_qubits_; }
    size_t getStateSize() const { return 1ULL << num_qubits_; }
    
private:
    int num_qubits_;
    size_t size_;
    CudaMemory<cuDoubleComplex> d_state_;
    NoiseModel noise_model_;
    
    // Random number generation
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_;
    
    // CUDA random states for GPU-based noise
    CudaMemory<curandState> d_rng_states_;
    bool rng_initialized_;
    
    void initializeRNG(unsigned int seed);
    
    // Apply specific noise types
    void applyDepolarizing(int qubit, double p);
    void applyAmplitudeDamping(int qubit, double gamma);
    void applyPhaseDamping(int qubit, double gamma);
    void applyBitFlip(int qubit, double p);
    void applyPhaseFlip(int qubit, double p);
    void applyBitPhaseFlip(int qubit, double p);
    
    // Apply all noise channels in the model
    void applyAllNoiseChannels();
    
    // Gate application (reuse from regular Simulator)
    void launchSingleQubitGate(int gate_type, int target, double param = 0.0);
    void launchTwoQubitGate(int gate_type, int qubit1, int qubit2, double param = 0.0);
    void launchThreeQubitGate(int gate_type, int qubit1, int qubit2, int qubit3);
};

/**
 * BatchedSimulator - Run multiple simulations in parallel
 * 
 * Useful for:
 * - Monte Carlo sampling with noise
 * - Parameter sweep
 * - Variational algorithms
 * 
 * Each "trajectory" is an independent simulation with its own state vector.
 * All trajectories run the same circuit but with independent noise realizations.
 * 
 * Memory: batch_size * 2^n * 16 bytes
 *   - 100 trajectories, 20 qubits = 1.6 GB
 *   - 1000 trajectories, 16 qubits = 1 GB
 */
class BatchedSimulator {
public:
    BatchedSimulator(int num_qubits, int batch_size);
    BatchedSimulator(int num_qubits, int batch_size, const NoiseModel& noise_model);
    ~BatchedSimulator() noexcept = default;
    
    // Disable copy, enable move
    BatchedSimulator(const BatchedSimulator&) = delete;
    BatchedSimulator& operator=(const BatchedSimulator&) = delete;
    BatchedSimulator(BatchedSimulator&&) noexcept = default;
    BatchedSimulator& operator=(BatchedSimulator&&) noexcept = default;
    
    // Set noise model for all trajectories
    void setNoiseModel(const NoiseModel& noise_model);
    
    // Set random seed
    void setSeed(unsigned int seed);
    
    // Reset all trajectories to |0...0⟩
    void reset();
    
    // Run circuit on all trajectories
    void run(const class Circuit& circuit);
    
    // Get results
    // Returns probabilities averaged over all trajectories
    std::vector<double> getAverageProbabilities() const;
    
    // Get probabilities for a specific trajectory
    std::vector<double> getProbabilities(int trajectory_idx) const;
    
    // Sample from all trajectories (returns batch_size samples per shot)
    // result[shot][trajectory] = measurement outcome
    std::vector<std::vector<int>> sample(int n_shots);
    
    // Get histogram of measurement outcomes across all trajectories
    std::vector<int> getHistogram(int n_shots);
    
    // Info
    int getNumQubits() const { return num_qubits_; }
    int getBatchSize() const { return batch_size_; }
    size_t getTotalMemoryBytes() const { return batch_size_ * (1ULL << num_qubits_) * sizeof(cuDoubleComplex); }
    
private:
    int num_qubits_;
    int batch_size_;
    size_t state_size_;
    
    // Batched state: [batch_size * 2^n] contiguous array
    // Layout: trajectory 0 states, trajectory 1 states, ...
    CudaMemory<cuDoubleComplex> d_states_;
    
    NoiseModel noise_model_;
    
    // RNG
    std::mt19937 rng_;
    CudaMemory<curandState> d_rng_states_;
    
    void initializeRNG(unsigned int seed);
    
    // Batched gate application
    void launchBatchedSingleQubitGate(int gate_type, int target, double param = 0.0);
    void launchBatchedTwoQubitGate(int gate_type, int qubit1, int qubit2, double param = 0.0);
    
    // Batched noise application
    void applyBatchedNoise();
};

} // namespace qsim
