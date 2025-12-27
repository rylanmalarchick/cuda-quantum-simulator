#pragma once

#include <vector>
#include <string>
#include <cmath>

namespace qsim {

/**
 * Supported gate types
 */
enum class GateType {
    // Single-qubit gates
    X, Y, Z,        // Pauli gates
    H,              // Hadamard
    S, T,           // Phase gates
    Sdag, Tdag,     // Adjoint phase gates
    Rx, Ry, Rz,     // Rotation gates (parameterized)
    
    // Two-qubit gates
    CNOT,           // Controlled-X
    CZ,             // Controlled-Z
    SWAP            // Swap
};

/**
 * Gate operation in a circuit
 */
struct GateOp {
    GateType type;
    std::vector<int> qubits;  // Target qubit(s): [target] or [control, target]
    double parameter;          // For Rx, Ry, Rz (angle in radians)
    
    // Constructors for convenience
    GateOp(GateType t, int qubit) 
        : type(t), qubits{qubit}, parameter(0.0) {}
    
    GateOp(GateType t, int qubit, double param) 
        : type(t), qubits{qubit}, parameter(param) {}
    
    GateOp(GateType t, int qubit1, int qubit2) 
        : type(t), qubits{qubit1, qubit2}, parameter(0.0) {}
};

/**
 * Circuit - A sequence of quantum gate operations
 */
class Circuit {
public:
    explicit Circuit(int num_qubits);
    
    // Gate addition (fluent interface for chaining)
    Circuit& x(int qubit);
    Circuit& y(int qubit);
    Circuit& z(int qubit);
    Circuit& h(int qubit);
    Circuit& s(int qubit);
    Circuit& t(int qubit);
    Circuit& sdag(int qubit);
    Circuit& tdag(int qubit);
    Circuit& rx(int qubit, double theta);
    Circuit& ry(int qubit, double theta);
    Circuit& rz(int qubit, double theta);
    Circuit& cnot(int control, int target);
    Circuit& cx(int control, int target) { return cnot(control, target); }
    Circuit& cz(int control, int target);
    Circuit& swap(int qubit1, int qubit2);
    
    // Access
    int getNumQubits() const { return num_qubits_; }
    const std::vector<GateOp>& getGates() const { return gates_; }
    size_t getDepth() const;  // Circuit depth (max gates on any qubit)
    size_t getGateCount() const { return gates_.size(); }
    
    // Utility
    void clear() { gates_.clear(); }
    std::string toString() const;  // Human-readable representation
    
private:
    int num_qubits_;
    std::vector<GateOp> gates_;
    
    void validateQubit(int qubit) const;
    void validateQubitPair(int q1, int q2) const;
};

// ============================================================================
// Common circuit patterns (factory functions)
// ============================================================================

// Bell state: |00⟩ → (|00⟩ + |11⟩)/√2
Circuit createBellCircuit();

// GHZ state: |0...0⟩ → (|0...0⟩ + |1...1⟩)/√2
Circuit createGHZCircuit(int num_qubits);

// Random circuit (for benchmarking)
Circuit createRandomCircuit(int num_qubits, int depth, unsigned int seed = 42);

} // namespace qsim
