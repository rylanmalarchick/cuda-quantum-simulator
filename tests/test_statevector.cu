// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Rylan Malarchick

/**
 * State Vector Tests
 * 
 * Tests for the StateVector class - GPU-resident quantum state.
 */

#include <gtest/gtest.h>
#include "StateVector.cuh"
#include "Gates.cuh"
#include "Constants.hpp"
#include <cmath>
#include <complex>
#include <map>

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

// ============================================================================
// Measurement Tests
// ============================================================================

TEST(MeasurementTest, SampleBasisState) {
    // |00⟩ should always sample to 0
    StateVector sv(2);
    
    auto samples = sv.sample(100);
    EXPECT_EQ(samples.size(), 100u);
    
    for (int s : samples) {
        EXPECT_EQ(s, 0) << "Sampling |00⟩ should always yield 0";
    }
}

TEST(MeasurementTest, SampleBasisStateNonZero) {
    // Initialize to |11⟩ = index 3
    StateVector sv(2);
    sv.initializeBasis(3);
    
    auto samples = sv.sample(100);
    
    for (int s : samples) {
        EXPECT_EQ(s, 3) << "Sampling |11⟩ should always yield 3";
    }
}

TEST(MeasurementTest, MeasureZeroState) {
    // Measure qubit 0 of |00⟩ - should always get 0
    StateVector sv(2);
    
    int result = sv.measure(0);
    EXPECT_EQ(result, 0) << "Measuring qubit 0 of |00⟩ should yield 0";
    
    // State should still be normalized after collapse
    EXPECT_TRUE(sv.isNormalized(1e-10));
}

TEST(MeasurementTest, MeasureOneState) {
    // Initialize to |1⟩ for single qubit
    StateVector sv(1);
    sv.initializeBasis(1);  // |1⟩
    
    int result = sv.measure(0);
    EXPECT_EQ(result, 1) << "Measuring qubit 0 of |1⟩ should yield 1";
    
    EXPECT_TRUE(sv.isNormalized(1e-10));
}

TEST(MeasurementTest, MeasureCollapsesBellState) {
    // Create |Φ+⟩ = (|00⟩ + |11⟩) / √2
    StateVector sv(2);
    
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(sv.getSize(), threads);
    
    // Apply H to qubit 0
    applyH<<<blocks, threads>>>(sv.devicePtr(), 2, 0);
    cudaDeviceSynchronize();
    
    // Apply CNOT(0, 1)
    applyCNOT<<<blocks, threads>>>(sv.devicePtr(), 2, 0, 1);
    cudaDeviceSynchronize();
    
    // Measure qubit 0
    int result0 = sv.measure(0);
    EXPECT_TRUE(result0 == 0 || result0 == 1);
    
    // After measuring qubit 0, qubit 1 should be in same state (Bell entanglement)
    // Measure qubit 1 - should be correlated with qubit 0
    int result1 = sv.measure(1);
    EXPECT_EQ(result0, result1) << "Bell state qubits should be correlated";
    
    EXPECT_TRUE(sv.isNormalized(1e-10));
}

TEST(MeasurementTest, SampleSuperposition) {
    // Create |+⟩ = (|0⟩ + |1⟩) / √2 on single qubit
    StateVector sv(1);
    
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(sv.getSize(), threads);
    
    applyH<<<blocks, threads>>>(sv.devicePtr(), 1, 0);
    cudaDeviceSynchronize();
    
    // Sample many times - should get roughly 50/50 distribution
    const int n_shots = 10000;
    auto samples = sv.sample(n_shots);
    
    int count_0 = 0;
    int count_1 = 0;
    for (int s : samples) {
        if (s == 0) count_0++;
        else count_1++;
    }
    
    // Expect roughly 50/50, allow 5% tolerance
    double ratio = static_cast<double>(count_0) / n_shots;
    EXPECT_NEAR(ratio, 0.5, 0.05) << "Superposition should sample ~50/50";
}

TEST(MeasurementTest, SampleBellState) {
    // Create Bell state, sample many times
    StateVector sv(2);
    
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(sv.getSize(), threads);
    
    applyH<<<blocks, threads>>>(sv.devicePtr(), 2, 0);
    cudaDeviceSynchronize();
    applyCNOT<<<blocks, threads>>>(sv.devicePtr(), 2, 0, 1);
    cudaDeviceSynchronize();
    
    // Sample - should only get |00⟩ (index 0) or |11⟩ (index 3)
    const int n_shots = 1000;
    auto samples = sv.sample(n_shots);
    
    std::map<int, int> counts;
    for (int s : samples) {
        counts[s]++;
    }
    
    // Only states 0 and 3 should appear
    EXPECT_EQ(counts.size(), 2u);
    EXPECT_TRUE(counts.count(0) > 0);
    EXPECT_TRUE(counts.count(3) > 0);
    EXPECT_EQ(counts.count(1), 0u);
    EXPECT_EQ(counts.count(2), 0u);
}

TEST(MeasurementTest, InvalidQubitThrows) {
    StateVector sv(3);
    
    EXPECT_THROW(sv.measure(-1), std::invalid_argument);
    EXPECT_THROW(sv.measure(3), std::invalid_argument);
    EXPECT_THROW(sv.measure(100), std::invalid_argument);
}

TEST(MeasurementTest, InvalidShotsThrows) {
    StateVector sv(2);
    
    EXPECT_THROW(sv.sample(0), std::invalid_argument);
    EXPECT_THROW(sv.sample(-1), std::invalid_argument);
}
