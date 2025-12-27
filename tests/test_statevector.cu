/**
 * State Vector Tests
 * 
 * Tests for the StateVector class - GPU-resident quantum state.
 */

#include <gtest/gtest.h>
#include "StateVector.cuh"
#include <cmath>
#include <complex>

using namespace qsim;

TEST(StateVectorTest, Initialization) {
    StateVector sv(3);  // 3 qubits = 8 states
    
    EXPECT_EQ(sv.getNumQubits(), 3);
    EXPECT_EQ(sv.getSize(), 8u);
    
    auto state = sv.toHost();
    EXPECT_EQ(state.size(), 8u);
    
    // Should be initialized to |000⟩
    EXPECT_NEAR(std::abs(state[0]), 1.0, 1e-10);
    for (size_t i = 1; i < 8; ++i) {
        EXPECT_NEAR(std::abs(state[i]), 0.0, 1e-10);
    }
}

TEST(StateVectorTest, BasisInitialization) {
    StateVector sv(4);  // 4 qubits = 16 states
    
    // Initialize to |1010⟩ = index 10
    sv.initializeBasis(10);
    
    auto state = sv.toHost();
    
    for (size_t i = 0; i < 16; ++i) {
        if (i == 10) {
            EXPECT_NEAR(std::abs(state[i]), 1.0, 1e-10);
        } else {
            EXPECT_NEAR(std::abs(state[i]), 0.0, 1e-10);
        }
    }
}

TEST(StateVectorTest, TotalProbability) {
    StateVector sv(5);  // 5 qubits
    
    double total = sv.getTotalProbability();
    EXPECT_NEAR(total, 1.0, 1e-10) << "Total probability should be 1.0";
}

TEST(StateVectorTest, Probabilities) {
    StateVector sv(2);
    
    auto probs = sv.getProbabilities();
    EXPECT_EQ(probs.size(), 4u);
    
    // Initial state |00⟩ should have P(|00⟩) = 1
    EXPECT_NEAR(probs[0], 1.0, 1e-10);
    EXPECT_NEAR(probs[1], 0.0, 1e-10);
    EXPECT_NEAR(probs[2], 0.0, 1e-10);
    EXPECT_NEAR(probs[3], 0.0, 1e-10);
}

TEST(StateVectorTest, MoveSemantics) {
    StateVector sv1(4);
    sv1.initializeBasis(5);
    
    // Move constructor
    StateVector sv2(std::move(sv1));
    
    EXPECT_EQ(sv2.getNumQubits(), 4);
    EXPECT_EQ(sv2.getSize(), 16u);
    
    auto state = sv2.toHost();
    EXPECT_NEAR(std::abs(state[5]), 1.0, 1e-10);
}

TEST(StateVectorTest, LargerStateVector) {
    // Test with 20 qubits (1M states, 16MB)
    StateVector sv(20);
    
    EXPECT_EQ(sv.getSize(), 1u << 20);
    
    double total = sv.getTotalProbability();
    EXPECT_NEAR(total, 1.0, 1e-8);
}
