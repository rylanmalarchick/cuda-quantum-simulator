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
