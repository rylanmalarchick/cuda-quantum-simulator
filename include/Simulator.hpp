/**
 * @file Simulator.hpp
 * @brief Main quantum circuit simulator orchestrating GPU execution
 * @author Rylan Malarchick
 * @date 2024
 *
 * Provides the primary interface for quantum circuit simulation. The Simulator
 * manages a GPU-resident state vector and dispatches gate operations to
 * optimized CUDA kernels.
 *
 * Key capabilities:
 * - Circuit execution with automatic kernel dispatch
 * - State vector inspection and probability extraction
 * - Sampling (without collapse) and measurement (with collapse)
 * - Support for up to 27 qubits on 8GB GPU (state vector: 2^n * 16 bytes)
 *
 * Performance is achieved through:
 * - GPU-accelerated gate kernels with coalesced memory access
 * - Shared memory optimization for gates on low-order qubits
 * - Efficient reduction kernels for probability computation
 *
 * @see StateVector.cuh for GPU memory management
 * @see Gates.cuh for kernel implementations
 *
 * @references
 * - Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and Quantum
 *   Information (10th Anniversary Edition). Cambridge University Press.
 */
#pragma once

#include "StateVector.cuh"
#include "Circuit.hpp"
#include <vector>
#include <complex>
#include <memory>

namespace qsim {

/**
 * Simulator - Orchestrates quantum circuit simulation on GPU
 * 
 * Usage:
 *   Simulator sim(4);  // 4 qubits
 *   Circuit circuit(4);
 *   circuit.h(0).cnot(0, 1).cnot(1, 2).cnot(2, 3);
 *   sim.run(circuit);
 *   auto probs = sim.getProbabilities();
 *   auto samples = sim.sample(1000);
 */
class Simulator {
public:
    explicit Simulator(int num_qubits);
    
    // Reset state to |0...0⟩
    void reset();
    
    // Run a circuit (applies all gates in sequence)
    void run(const Circuit& circuit);
    
    // Apply a single gate
    void applyGate(const GateOp& gate);
    
    // State inspection
    std::vector<std::complex<double>> getStateVector() const;
    std::vector<double> getProbabilities() const;
    
    // Measurement
    std::vector<int> sample(int n_shots);  // Sample without state collapse
    int measureQubit(int qubit);           // Measure and collapse
    
    // Info
    int getNumQubits() const { return state_.getNumQubits(); }
    size_t getStateSize() const { return state_.getSize(); }
    
private:
    StateVector state_;
    
    // Kernel launch helpers
    void launchSingleQubitGate(GateType type, int target, double param = 0.0);
    void launchTwoQubitGate(GateType type, int qubit1, int qubit2, double param = 0.0);
    void launchThreeQubitGate(GateType type, int qubit1, int qubit2, int qubit3);
};

// ============================================================================
// CPU Simulator (for benchmarking comparison)
// ============================================================================

class CPUSimulator {
public:
    explicit CPUSimulator(int num_qubits);
    
    void reset();
    void run(const Circuit& circuit);
    void applyGate(const GateOp& gate);
    
    std::vector<std::complex<double>> getStateVector() const { return state_; }
    std::vector<double> getProbabilities() const;
    std::vector<int> sample(int n_shots);
    
    int getNumQubits() const { return num_qubits_; }
    
private:
    int num_qubits_;
    size_t size_;
    std::vector<std::complex<double>> state_;
    
    void applySingleQubitGate(GateType type, int target, double param = 0.0);
    void applyTwoQubitGate(GateType type, int qubit1, int qubit2);
};

} // namespace qsim
