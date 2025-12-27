// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Rylan Malarchick

/**
 * Boundary Condition Tests
 * 
 * Tests for edge cases and boundary conditions:
 * - Minimum qubit count (1 qubit)
 * - Maximum practical qubit count (limited by GPU memory)
 * - Invalid inputs (should throw appropriate exceptions)
 * - State normalization verification
 */

#include <gtest/gtest.h>
#include "Simulator.hpp"
#include "StateVector.cuh"
#include <cmath>
#include <stdexcept>

using namespace qsim;

namespace {
constexpr double TOLERANCE = 1e-12;
}

// ============================================================================
// Qubit count boundaries
// ============================================================================

TEST(BoundaryTest, SingleQubit) {
    // Minimum valid case: 1 qubit
    Simulator sim(1);
    
    EXPECT_EQ(sim.getNumQubits(), 1);
    EXPECT_EQ(sim.getStateSize(), 2u);
    
    Circuit c(1);
    c.h(0).t(0).h(0);
    sim.run(c);
    
    auto probs = sim.getProbabilities();
    EXPECT_EQ(probs.size(), 2u);
    
    // Probabilities should sum to 1
    double total = probs[0] + probs[1];
    EXPECT_NEAR(total, 1.0, TOLERANCE);
}

TEST(BoundaryTest, TwoQubits) {
    Simulator sim(2);
    
    EXPECT_EQ(sim.getNumQubits(), 2);
    EXPECT_EQ(sim.getStateSize(), 4u);
    
    // Bell state
    sim.run(createBellCircuit());
    
    auto probs = sim.getProbabilities();
    EXPECT_NEAR(probs[0] + probs[1] + probs[2] + probs[3], 1.0, TOLERANCE);
}

TEST(BoundaryTest, MediumQubitCount_16) {
    // 16 qubits = 2^16 = 65536 states = 1MB memory
    Simulator sim(16);
    
    EXPECT_EQ(sim.getNumQubits(), 16);
    EXPECT_EQ(sim.getStateSize(), 65536u);
    
    // Apply some gates
    Circuit c(16);
    for (int i = 0; i < 16; ++i) c.h(i);
    sim.run(c);
    
    auto probs = sim.getProbabilities();
    EXPECT_EQ(probs.size(), 65536u);
    
    // Each probability should be 1/65536
    double expected = 1.0 / 65536.0;
    for (size_t i = 0; i < 16; ++i) {  // Sample first 16
        EXPECT_NEAR(probs[i], expected, TOLERANCE);
    }
}

TEST(BoundaryTest, LargerQubitCount_20) {
    // 20 qubits = 2^20 = ~1M states = 16MB memory
    Simulator sim(20);
    
    EXPECT_EQ(sim.getNumQubits(), 20);
    EXPECT_EQ(sim.getStateSize(), 1u << 20);
    
    // GHZ state on 20 qubits
    sim.run(createGHZCircuit(20));
    
    auto probs = sim.getProbabilities();
    
    // Only first and last should be non-zero
    EXPECT_NEAR(probs[0], 0.5, TOLERANCE);
    EXPECT_NEAR(probs[(1u << 20) - 1], 0.5, TOLERANCE);
    
    // Check a few intermediate states are zero
    EXPECT_NEAR(probs[1], 0.0, TOLERANCE);
    EXPECT_NEAR(probs[1000], 0.0, TOLERANCE);
    EXPECT_NEAR(probs[500000], 0.0, TOLERANCE);
}

// ============================================================================
// Invalid input tests
// ============================================================================

TEST(BoundaryTest, ZeroQubits_ShouldThrow) {
    EXPECT_THROW(Simulator sim(0), std::invalid_argument);
    EXPECT_THROW(StateVector sv(0), std::invalid_argument);
    EXPECT_THROW(Circuit c(0), std::invalid_argument);
}

TEST(BoundaryTest, NegativeQubits_ShouldThrow) {
    EXPECT_THROW(Simulator sim(-1), std::invalid_argument);
    EXPECT_THROW(StateVector sv(-5), std::invalid_argument);
}

TEST(BoundaryTest, TooManyQubits_ShouldThrow) {
    // 31+ qubits would require >32GB, definitely too many
    EXPECT_THROW(StateVector sv(31), std::invalid_argument);
    EXPECT_THROW(StateVector sv(40), std::invalid_argument);
}

TEST(BoundaryTest, InvalidQubitIndex_ShouldThrow) {
    Circuit c(4);
    
    EXPECT_THROW(c.h(-1), std::out_of_range);
    EXPECT_THROW(c.h(4), std::out_of_range);
    EXPECT_THROW(c.h(100), std::out_of_range);
    
    EXPECT_THROW(c.cnot(0, 4), std::out_of_range);
    EXPECT_THROW(c.cnot(-1, 0), std::out_of_range);
}

TEST(BoundaryTest, SameQubitForTwoQubitGate_ShouldThrow) {
    Circuit c(4);
    
    EXPECT_THROW(c.cnot(0, 0), std::invalid_argument);
    EXPECT_THROW(c.cz(2, 2), std::invalid_argument);
    EXPECT_THROW(c.swap(1, 1), std::invalid_argument);
}

TEST(BoundaryTest, CircuitQubitMismatch_ShouldThrow) {
    Simulator sim(4);
    Circuit c(3);  // Different qubit count
    c.h(0);
    
    EXPECT_THROW(sim.run(c), std::invalid_argument);
}

TEST(BoundaryTest, InvalidBasisIndex_ShouldThrow) {
    StateVector sv(3);  // 8 states, indices 0-7
    
    EXPECT_THROW(sv.initializeBasis(8), std::invalid_argument);
    EXPECT_THROW(sv.initializeBasis(100), std::invalid_argument);
    
    // Valid indices should not throw
    EXPECT_NO_THROW(sv.initializeBasis(0));
    EXPECT_NO_THROW(sv.initializeBasis(7));
}

// ============================================================================
// Normalization tests
// ============================================================================

TEST(BoundaryTest, Normalization_AfterInitialization) {
    for (int n = 1; n <= 10; ++n) {
        StateVector sv(n);
        EXPECT_NEAR(sv.getTotalProbability(), 1.0, TOLERANCE)
            << "Normalization failed for " << n << " qubits after init";
    }
}

TEST(BoundaryTest, Normalization_AfterGates) {
    Simulator sim(6);
    
    // Apply various gates
    Circuit c(6);
    for (int i = 0; i < 6; ++i) c.h(i);
    c.cnot(0, 1).cnot(2, 3).cnot(4, 5);
    c.cz(0, 2).cz(1, 3);
    c.rx(0, 1.23).ry(2, 0.45).rz(4, 2.34);
    
    sim.run(c);
    
    auto probs = sim.getProbabilities();
    double total = 0.0;
    for (double p : probs) total += p;
    
    EXPECT_NEAR(total, 1.0, TOLERANCE)
        << "Normalization violated after gate sequence";
}

TEST(BoundaryTest, Normalization_DeepCircuit) {
    // Test that normalization is preserved after many gates
    // (catching potential accumulated floating point error)
    Simulator sim(4);
    
    Circuit c = createRandomCircuit(4, 1000, 42);
    sim.run(c);
    
    auto probs = sim.getProbabilities();
    double total = 0.0;
    for (double p : probs) total += p;
    
    // Allow slightly more tolerance for deep circuits
    EXPECT_NEAR(total, 1.0, 1e-10)
        << "Normalization violated after 1000 gates";
}

// ============================================================================
// State initialization tests
// ============================================================================

TEST(BoundaryTest, BasisStateInitialization) {
    StateVector sv(4);  // 16 states
    
    for (size_t basis = 0; basis < 16; ++basis) {
        sv.initializeBasis(basis);
        auto state = sv.toHost();
        
        for (size_t i = 0; i < 16; ++i) {
            if (i == basis) {
                EXPECT_NEAR(std::abs(state[i]), 1.0, TOLERANCE)
                    << "Basis " << basis << " failed at index " << i;
            } else {
                EXPECT_NEAR(std::abs(state[i]), 0.0, TOLERANCE)
                    << "Basis " << basis << " non-zero at index " << i;
            }
        }
    }
}

TEST(BoundaryTest, ResetAfterCircuit) {
    Simulator sim(4);
    
    // Run complex circuit
    Circuit c = createRandomCircuit(4, 50, 123);
    sim.run(c);
    
    // Verify state is not |0000⟩
    auto probs_after = sim.getProbabilities();
    EXPECT_FALSE(std::abs(probs_after[0] - 1.0) < TOLERANCE)
        << "Circuit should have changed the state";
    
    // Reset
    sim.reset();
    
    // Verify back to |0000⟩
    auto probs_reset = sim.getProbabilities();
    EXPECT_NEAR(probs_reset[0], 1.0, TOLERANCE)
        << "Reset should restore |0000⟩";
    
    for (size_t i = 1; i < probs_reset.size(); ++i) {
        EXPECT_NEAR(probs_reset[i], 0.0, TOLERANCE)
            << "Reset failed: non-zero probability at index " << i;
    }
}

// ============================================================================
// Memory and resource tests
// ============================================================================

TEST(BoundaryTest, MoveSemantics) {
    StateVector sv1(10);
    sv1.initializeBasis(42);
    
    // Move construct
    StateVector sv2(std::move(sv1));
    EXPECT_EQ(sv2.getNumQubits(), 10);
    EXPECT_EQ(sv2.getSize(), 1024u);
    
    auto state = sv2.toHost();
    EXPECT_NEAR(std::abs(state[42]), 1.0, TOLERANCE);
    
    // Move assign
    StateVector sv3(5);
    sv3 = std::move(sv2);
    EXPECT_EQ(sv3.getNumQubits(), 10);
    
    state = sv3.toHost();
    EXPECT_NEAR(std::abs(state[42]), 1.0, TOLERANCE);
}

TEST(BoundaryTest, MultipleSimulatorsCoexist) {
    // Create multiple simulators simultaneously
    Simulator sim1(4);
    Simulator sim2(6);
    Simulator sim3(8);
    
    Circuit c1 = createGHZCircuit(4);
    Circuit c2 = createGHZCircuit(6);
    Circuit c3 = createGHZCircuit(8);
    
    sim1.run(c1);
    sim2.run(c2);
    sim3.run(c3);
    
    // Verify each is independent
    auto p1 = sim1.getProbabilities();
    auto p2 = sim2.getProbabilities();
    auto p3 = sim3.getProbabilities();
    
    EXPECT_NEAR(p1[0], 0.5, TOLERANCE);
    EXPECT_NEAR(p1[15], 0.5, TOLERANCE);  // 2^4 - 1
    
    EXPECT_NEAR(p2[0], 0.5, TOLERANCE);
    EXPECT_NEAR(p2[63], 0.5, TOLERANCE);  // 2^6 - 1
    
    EXPECT_NEAR(p3[0], 0.5, TOLERANCE);
    EXPECT_NEAR(p3[255], 0.5, TOLERANCE);  // 2^8 - 1
}
