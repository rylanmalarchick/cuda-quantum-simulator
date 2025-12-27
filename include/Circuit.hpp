// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Rylan Malarchick

/**
 * @file Circuit.hpp
 * @brief Quantum circuit representation with fluent gate API
 * @author Rylan Malarchick
 * @date 2024
 *
 * Provides a high-level interface for constructing quantum circuits using
 * method chaining (fluent API). Circuits are represented as ordered sequences
 * of gate operations that can be executed by the Simulator.
 *
 * Supported gate set includes:
 * - Single-qubit: Pauli (X, Y, Z), Hadamard (H), Phase (S, T), Rotations (Rx, Ry, Rz)
 * - Two-qubit: CNOT, CZ, SWAP, controlled rotations (CRY, CRZ)
 * - Three-qubit: Toffoli (CCX)
 *
 * This gate set is universal for quantum computation. The Clifford+T subset
 * (H, S, CNOT, T) forms an approximately universal gate set.
 *
 * @see Simulator.hpp for circuit execution
 *
 * @references
 * - Barenco, A., et al. (1995). Elementary gates for quantum computation.
 *   Physical Review A, 52(5), 3457.
 * - Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and Quantum
 *   Information (10th Anniversary Edition). Cambridge University Press.
 *   Chapter 4: Quantum circuits.
 */
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
    CRY,            // Controlled-Ry (parameterized)
    CRZ,            // Controlled-Rz (parameterized)
    SWAP,           // Swap
    
    // Three-qubit gates
    Toffoli         // CCX (Controlled-Controlled-X)
};

/**
 * Gate operation in a circuit
 */
struct GateOp {
    GateType type;
    std::vector<int> qubits;  // Target qubit(s): [target], [control, target], or [c1, c2, target]
    double parameter;          // For Rx, Ry, Rz, CRY, CRZ (angle in radians)
    
    // Constructors for convenience
    GateOp(GateType t, int qubit) 
        : type(t), qubits{qubit}, parameter(0.0) {}
    
    GateOp(GateType t, int qubit, double param) 
        : type(t), qubits{qubit}, parameter(param) {}
    
    GateOp(GateType t, int qubit1, int qubit2) 
        : type(t), qubits{qubit1, qubit2}, parameter(0.0) {}
    
    GateOp(GateType t, int qubit1, int qubit2, double param)
        : type(t), qubits{qubit1, qubit2}, parameter(param) {}
    
    GateOp(GateType t, int qubit1, int qubit2, int qubit3)
        : type(t), qubits{qubit1, qubit2, qubit3}, parameter(0.0) {}
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
    Circuit& cry(int control, int target, double theta);
    Circuit& crz(int control, int target, double theta);
    Circuit& swap(int qubit1, int qubit2);
    Circuit& toffoli(int control1, int control2, int target);
    Circuit& ccx(int control1, int control2, int target) { return toffoli(control1, control2, target); }
    
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
    void validateQubitTriple(int q1, int q2, int q3) const;
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
