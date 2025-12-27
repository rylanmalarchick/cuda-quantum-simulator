/**
 * Gate Tests
 * 
 * Tests for individual quantum gate kernels.
 * Each test applies a gate and verifies the resulting state.
 */

#include <gtest/gtest.h>
#include "Simulator.hpp"
#include <cmath>
#include <complex>

using namespace qsim;

const double EPSILON = 1e-10;
const double INV_SQRT2 = 0.7071067811865476;

class GateTest : public ::testing::Test {
protected:
    void expectStateEquals(const std::vector<std::complex<double>>& actual,
                          const std::vector<std::complex<double>>& expected) {
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i].real(), expected[i].real(), EPSILON)
                << "Real part mismatch at index " << i;
            EXPECT_NEAR(actual[i].imag(), expected[i].imag(), EPSILON)
                << "Imag part mismatch at index " << i;
        }
    }
};

// ============================================================================
// Single-Qubit Gates on 1-qubit system
// ============================================================================

TEST_F(GateTest, XGate) {
    Simulator sim(1);
    Circuit c(1);
    c.x(0);
    sim.run(c);
    
    auto state = sim.getStateVector();
    // X|0⟩ = |1⟩
    expectStateEquals(state, {{0, 0}, {1, 0}});
}

TEST_F(GateTest, HGate) {
    Simulator sim(1);
    Circuit c(1);
    c.h(0);
    sim.run(c);
    
    auto state = sim.getStateVector();
    // H|0⟩ = (|0⟩ + |1⟩)/√2
    expectStateEquals(state, {{INV_SQRT2, 0}, {INV_SQRT2, 0}});
}

TEST_F(GateTest, HH_Identity) {
    // H applied twice should return to original state
    Simulator sim(1);
    Circuit c(1);
    c.h(0).h(0);
    sim.run(c);
    
    auto state = sim.getStateVector();
    expectStateEquals(state, {{1, 0}, {0, 0}});
}

TEST_F(GateTest, ZGate) {
    // Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
    // Apply to |+⟩ = (|0⟩ + |1⟩)/√2
    Simulator sim(1);
    Circuit c(1);
    c.h(0).z(0);
    sim.run(c);
    
    auto state = sim.getStateVector();
    // Z|+⟩ = (|0⟩ - |1⟩)/√2 = |−⟩
    expectStateEquals(state, {{INV_SQRT2, 0}, {-INV_SQRT2, 0}});
}

TEST_F(GateTest, YGate) {
    Simulator sim(1);
    Circuit c(1);
    c.y(0);
    sim.run(c);
    
    auto state = sim.getStateVector();
    // Y|0⟩ = i|1⟩
    expectStateEquals(state, {{0, 0}, {0, 1}});
}

TEST_F(GateTest, SGate) {
    // S|1⟩ = i|1⟩
    Simulator sim(1);
    Circuit c(1);
    c.x(0).s(0);  // Create |1⟩, then apply S
    sim.run(c);
    
    auto state = sim.getStateVector();
    expectStateEquals(state, {{0, 0}, {0, 1}});
}

TEST_F(GateTest, TGate) {
    // T|1⟩ = e^(iπ/4)|1⟩ = (1+i)/√2 |1⟩
    Simulator sim(1);
    Circuit c(1);
    c.x(0).t(0);
    sim.run(c);
    
    auto state = sim.getStateVector();
    expectStateEquals(state, {{0, 0}, {INV_SQRT2, INV_SQRT2}});
}

TEST_F(GateTest, RzGate) {
    // Rz(π)|+⟩ should give |−⟩ (up to global phase)
    Simulator sim(1);
    Circuit c(1);
    c.h(0).rz(0, M_PI);
    sim.run(c);
    
    auto probs = sim.getProbabilities();
    EXPECT_NEAR(probs[0], 0.5, EPSILON);
    EXPECT_NEAR(probs[1], 0.5, EPSILON);
}

TEST_F(GateTest, RxGate) {
    // Rx(π)|0⟩ = -i|1⟩
    Simulator sim(1);
    Circuit c(1);
    c.rx(0, M_PI);
    sim.run(c);
    
    auto state = sim.getStateVector();
    EXPECT_NEAR(std::abs(state[0]), 0.0, EPSILON);
    EXPECT_NEAR(std::abs(state[1]), 1.0, EPSILON);
}

TEST_F(GateTest, RyGate) {
    // Ry(π)|0⟩ = |1⟩
    Simulator sim(1);
    Circuit c(1);
    c.ry(0, M_PI);
    sim.run(c);
    
    auto state = sim.getStateVector();
    EXPECT_NEAR(std::abs(state[0]), 0.0, EPSILON);
    EXPECT_NEAR(std::abs(state[1]), 1.0, EPSILON);
}

// ============================================================================
// Two-Qubit Gates
// ============================================================================

TEST_F(GateTest, CNOT_Control0) {
    // CNOT with control=0 in state |00⟩ should do nothing
    Simulator sim(2);
    Circuit c(2);
    c.cnot(0, 1);
    sim.run(c);
    
    auto state = sim.getStateVector();
    // |00⟩ unchanged
    expectStateEquals(state, {{1,0}, {0,0}, {0,0}, {0,0}});
}

TEST_F(GateTest, CNOT_Control1) {
    // CNOT with control=1 should flip target
    // |10⟩ → |11⟩
    Simulator sim(2);
    Circuit c(2);
    c.x(0).cnot(0, 1);  // Create |10⟩ then CNOT
    sim.run(c);
    
    auto state = sim.getStateVector();
    // Should be |11⟩ = index 3
    expectStateEquals(state, {{0,0}, {0,0}, {0,0}, {1,0}});
}

TEST_F(GateTest, BellState) {
    // H|0⟩ then CNOT creates Bell state (|00⟩ + |11⟩)/√2
    Simulator sim(2);
    sim.run(createBellCircuit());
    
    auto state = sim.getStateVector();
    expectStateEquals(state, {
        {INV_SQRT2, 0},  // |00⟩
        {0, 0},          // |01⟩
        {0, 0},          // |10⟩
        {INV_SQRT2, 0}   // |11⟩
    });
}

TEST_F(GateTest, CZGate) {
    // CZ|11⟩ = -|11⟩
    Simulator sim(2);
    Circuit c(2);
    c.x(0).x(1).cz(0, 1);
    sim.run(c);
    
    auto state = sim.getStateVector();
    expectStateEquals(state, {{0,0}, {0,0}, {0,0}, {-1,0}});
}

TEST_F(GateTest, SWAPGate) {
    // SWAP|01⟩ = |10⟩
    // x(0) creates |01⟩ (index 1), SWAP gives |10⟩ (index 2)
    Simulator sim(2);
    Circuit c(2);
    c.x(0).swap(0, 1);
    sim.run(c);
    
    auto state = sim.getStateVector();
    // |10⟩ = index 2
    expectStateEquals(state, {{0,0}, {0,0}, {1,0}, {0,0}});
}

// ============================================================================
// Multi-qubit systems
// ============================================================================

TEST_F(GateTest, GHZState) {
    // 4-qubit GHZ: (|0000⟩ + |1111⟩)/√2
    Simulator sim(4);
    sim.run(createGHZCircuit(4));
    
    auto probs = sim.getProbabilities();
    EXPECT_NEAR(probs[0], 0.5, EPSILON);    // |0000⟩
    EXPECT_NEAR(probs[15], 0.5, EPSILON);   // |1111⟩
    
    // All other states should be 0
    for (int i = 1; i < 15; ++i) {
        EXPECT_NEAR(probs[i], 0.0, EPSILON);
    }
}

TEST_F(GateTest, HadamardAllQubits) {
    // H on all qubits creates equal superposition
    Simulator sim(4);
    Circuit c(4);
    c.h(0).h(1).h(2).h(3);
    sim.run(c);
    
    auto probs = sim.getProbabilities();
    double expected = 1.0 / 16.0;
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(probs[i], expected, EPSILON);
    }
}

// ============================================================================
// Advanced Gates (Toffoli, CRY, CRZ)
// ============================================================================

TEST_F(GateTest, ToffoliGate_BothControlsOn) {
    // Toffoli flips target when both controls are 1
    // For 3 qubits with x(0).x(1): creates state with q0=1, q1=1, q2=0
    // In our bit ordering: index = q0 + 2*q1 + 4*q2 = 1 + 2 = 3 for |110⟩
    // After Toffoli(0,1,2): target q2 flips → q2=1, index = 1 + 2 + 4 = 7
    Simulator sim(3);
    Circuit c(3);
    c.x(0).x(1).toffoli(0, 1, 2);
    sim.run(c);
    
    auto state = sim.getStateVector();
    // Result should be at index 7
    std::vector<std::complex<double>> expected(8, {0, 0});
    expected[7] = {1, 0};
    expectStateEquals(state, expected);
}

TEST_F(GateTest, ToffoliGate_OneControlOff) {
    // Toffoli does nothing when only one control is 1
    // x(0) creates q0=1, others 0 → index 1
    Simulator sim(3);
    Circuit c(3);
    c.x(0).toffoli(0, 1, 2);
    sim.run(c);
    
    auto state = sim.getStateVector();
    // State stays at index 1
    std::vector<std::complex<double>> expected(8, {0, 0});
    expected[1] = {1, 0};
    expectStateEquals(state, expected);
}

TEST_F(GateTest, ToffoliGate_NoControlsOn) {
    // Toffoli does nothing when no controls are 1
    Simulator sim(3);
    Circuit c(3);
    c.toffoli(0, 1, 2);  // |000⟩ stays |000⟩
    sim.run(c);
    
    auto state = sim.getStateVector();
    std::vector<std::complex<double>> expected(8, {0, 0});
    expected[0] = {1, 0};
    expectStateEquals(state, expected);
}

TEST_F(GateTest, ToffoliGate_SelfInverse) {
    // Toffoli applied twice is identity
    // x(0).x(1) → index 3, Toffoli → index 7, Toffoli → index 3
    Simulator sim(3);
    Circuit c(3);
    c.x(0).x(1).toffoli(0, 1, 2).toffoli(0, 1, 2);
    sim.run(c);
    
    auto state = sim.getStateVector();
    // Should be back to index 3
    std::vector<std::complex<double>> expected(8, {0, 0});
    expected[3] = {1, 0};
    expectStateEquals(state, expected);
}

TEST_F(GateTest, CRYGate_ControlOff) {
    // CRY does nothing when control is 0
    Simulator sim(2);
    Circuit c(2);
    c.cry(0, 1, M_PI);  // Control is 0, should do nothing
    sim.run(c);
    
    auto state = sim.getStateVector();
    expectStateEquals(state, {{1,0}, {0,0}, {0,0}, {0,0}});
}

TEST_F(GateTest, CRYGate_ControlOn) {
    // CRY(π) with control=1 should flip target like Ry(π)
    // x(0) sets q0=1 → index 1. CRY on q1 should flip q1.
    // index 1 → index 3 (q0=1, q1=1)
    Simulator sim(2);
    Circuit c(2);
    c.x(0).cry(0, 1, M_PI);
    sim.run(c);
    
    auto state = sim.getStateVector();
    // Should be at index 3
    EXPECT_NEAR(std::abs(state[3]), 1.0, EPSILON);
}

TEST_F(GateTest, CRYGate_Superposition) {
    // CRY(π/2) with control=1 creates superposition on target
    // x(0) → index 1 (q0=1, q1=0)
    // CRY(π/2) on q1: creates superposition between index 1 and 3
    Simulator sim(2);
    Circuit c(2);
    c.x(0).cry(0, 1, M_PI / 2);
    sim.run(c);
    
    auto probs = sim.getProbabilities();
    // Should have probability in both index 1 and 3
    EXPECT_NEAR(probs[1] + probs[3], 1.0, EPSILON);
    EXPECT_GT(probs[1], 0.1);
    EXPECT_GT(probs[3], 0.1);
}

TEST_F(GateTest, CRZGate_ControlOff) {
    // CRZ does nothing when control is 0
    // Start with |00⟩, apply H to q1 → superposition of index 0 and 2
    // Then CRZ with control q0=0 should do nothing
    Simulator sim(2);
    Circuit c(2);
    c.h(1).crz(0, 1, M_PI);  // Control q0 is 0
    sim.run(c);
    
    auto state = sim.getStateVector();
    // |0+⟩ = (|00⟩ + |01⟩)/√2 = (index 0 + index 2)/√2
    expectStateEquals(state, {{INV_SQRT2,0}, {0,0}, {INV_SQRT2,0}, {0,0}});
}

TEST_F(GateTest, CRZGate_ControlOn) {
    // CRZ(π) with control=1 applies Z-like phase to target
    // x(0).h(1) creates (|10⟩ + |11⟩)/√2 = (index 1 + index 3)/√2
    // CRZ(π) applies e^(±iπ/2) phases → (-i|10⟩ + i|11⟩)/√2
    // Probabilities should still be 0.5 each
    Simulator sim(2);
    Circuit c(2);
    c.x(0).h(1).crz(0, 1, M_PI);
    sim.run(c);
    
    auto probs = sim.getProbabilities();
    EXPECT_NEAR(probs[1], 0.5, EPSILON);  // index 1
    EXPECT_NEAR(probs[3], 0.5, EPSILON);  // index 3
}
