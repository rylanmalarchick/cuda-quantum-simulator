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
    void launchTwoQubitGate(GateType type, int qubit1, int qubit2);
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
