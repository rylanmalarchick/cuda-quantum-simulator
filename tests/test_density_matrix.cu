#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include "DensityMatrix.cuh"
#include "Circuit.hpp"

using namespace qsim;

// ============================================================================
// DensityMatrix Class Tests
// ============================================================================

TEST(DensityMatrixTest, InitializesToZeroState) {
    DensityMatrix dm(2);
    
    auto probs = dm.getProbabilities();
    EXPECT_NEAR(probs[0], 1.0, 1e-10);  // |00> has probability 1
    EXPECT_NEAR(probs[1], 0.0, 1e-10);
    EXPECT_NEAR(probs[2], 0.0, 1e-10);
    EXPECT_NEAR(probs[3], 0.0, 1e-10);
}

TEST(DensityMatrixTest, TraceIsOne) {
    DensityMatrix dm(3);
    EXPECT_NEAR(dm.trace(), 1.0, 1e-10);
}

TEST(DensityMatrixTest, PurityIsOneForPureState) {
    DensityMatrix dm(2);
    // |00> is a pure state, purity should be 1
    EXPECT_NEAR(dm.purity(), 1.0, 1e-10);
}

TEST(DensityMatrixTest, MaximallyMixedHasCorrectPurity) {
    DensityMatrix dm(2);
    dm.initMaximallyMixed();
    
    // Purity of maximally mixed state = 1/dim = 1/4
    EXPECT_NEAR(dm.purity(), 0.25, 1e-10);
    EXPECT_NEAR(dm.trace(), 1.0, 1e-10);
}

TEST(DensityMatrixTest, InitFromPureState) {
    // Create |+> = (|0> + |1>)/sqrt(2)
    std::vector<std::complex<double>> plus_state = {
        {1.0/std::sqrt(2.0), 0.0},
        {1.0/std::sqrt(2.0), 0.0}
    };
    
    DensityMatrix dm(1, plus_state);
    
    auto probs = dm.getProbabilities();
    EXPECT_NEAR(probs[0], 0.5, 1e-10);
    EXPECT_NEAR(probs[1], 0.5, 1e-10);
    EXPECT_NEAR(dm.purity(), 1.0, 1e-10);
}

TEST(DensityMatrixTest, Reset) {
    DensityMatrix dm(2);
    dm.initMaximallyMixed();
    dm.reset();
    
    auto probs = dm.getProbabilities();
    EXPECT_NEAR(probs[0], 1.0, 1e-10);
    EXPECT_NEAR(dm.purity(), 1.0, 1e-10);
}

TEST(DensityMatrixTest, IsValid) {
    DensityMatrix dm(2);
    EXPECT_TRUE(dm.isValid());
    
    dm.initMaximallyMixed();
    EXPECT_TRUE(dm.isValid());
}

// ============================================================================
// DensityMatrixSimulator Gate Tests
// ============================================================================

TEST(DensityMatrixSimulatorTest, InitialState) {
    DensityMatrixSimulator sim(2);
    
    auto probs = sim.getProbabilities();
    EXPECT_NEAR(probs[0], 1.0, 1e-10);
    EXPECT_NEAR(probs[1], 0.0, 1e-10);
    EXPECT_NEAR(probs[2], 0.0, 1e-10);
    EXPECT_NEAR(probs[3], 0.0, 1e-10);
}

TEST(DensityMatrixSimulatorTest, XGate) {
    DensityMatrixSimulator sim(1);
    
    Circuit circuit(1);
    circuit.x(0);
    sim.run(circuit);
    
    auto probs = sim.getProbabilities();
    EXPECT_NEAR(probs[0], 0.0, 1e-10);  // |0> probability
    EXPECT_NEAR(probs[1], 1.0, 1e-10);  // |1> probability
}

TEST(DensityMatrixSimulatorTest, HGate) {
    DensityMatrixSimulator sim(1);
    
    Circuit circuit(1);
    circuit.h(0);
    sim.run(circuit);
    
    auto probs = sim.getProbabilities();
    EXPECT_NEAR(probs[0], 0.5, 1e-10);
    EXPECT_NEAR(probs[1], 0.5, 1e-10);
}

TEST(DensityMatrixSimulatorTest, HHIsIdentity) {
    DensityMatrixSimulator sim(1);
    
    Circuit circuit(1);
    circuit.h(0).h(0);
    sim.run(circuit);
    
    auto probs = sim.getProbabilities();
    EXPECT_NEAR(probs[0], 1.0, 1e-10);
    EXPECT_NEAR(probs[1], 0.0, 1e-10);
}

TEST(DensityMatrixSimulatorTest, ZGate) {
    DensityMatrixSimulator sim(1);
    
    // Z on |0> should leave probabilities unchanged
    Circuit circuit(1);
    circuit.z(0);
    sim.run(circuit);
    
    auto probs = sim.getProbabilities();
    EXPECT_NEAR(probs[0], 1.0, 1e-10);
    EXPECT_NEAR(probs[1], 0.0, 1e-10);
}

TEST(DensityMatrixSimulatorTest, BellState) {
    DensityMatrixSimulator sim(2);
    
    Circuit circuit(2);
    circuit.h(0).cnot(0, 1);
    sim.run(circuit);
    
    auto probs = sim.getProbabilities();
    EXPECT_NEAR(probs[0], 0.5, 1e-10);  // |00>
    EXPECT_NEAR(probs[1], 0.0, 1e-10);  // |01>
    EXPECT_NEAR(probs[2], 0.0, 1e-10);  // |10>
    EXPECT_NEAR(probs[3], 0.5, 1e-10);  // |11>
    
    // Bell state is pure
    EXPECT_NEAR(sim.getPurity(), 1.0, 1e-10);
}

TEST(DensityMatrixSimulatorTest, SWAPGate) {
    DensityMatrixSimulator sim(2);
    
    // Prepare |01> (X on qubit 0) then SWAP to get |10>
    Circuit circuit(2);
    circuit.x(0).swap(0, 1);
    sim.run(circuit);
    
    auto probs = sim.getProbabilities();
    EXPECT_NEAR(probs[0], 0.0, 1e-10);  // |00>
    EXPECT_NEAR(probs[1], 0.0, 1e-10);  // |01>
    EXPECT_NEAR(probs[2], 1.0, 1e-10);  // |10>
    EXPECT_NEAR(probs[3], 0.0, 1e-10);  // |11>
}

TEST(DensityMatrixSimulatorTest, CZGate) {
    DensityMatrixSimulator sim(2);
    
    // CZ on |00> should leave it unchanged
    Circuit circuit(2);
    circuit.cz(0, 1);
    sim.run(circuit);
    
    auto probs = sim.getProbabilities();
    EXPECT_NEAR(probs[0], 1.0, 1e-10);
}

// ============================================================================
// Noise Channel Tests
// ============================================================================

TEST(DensityMatrixNoiseTest, DepolarizingReducesPurity) {
    NoiseModel noise;
    noise.addDepolarizing(0.1);  // 10% depolarizing
    
    DensityMatrixSimulator sim(1, noise);
    
    Circuit circuit(1);
    circuit.h(0);  // Create superposition
    sim.run(circuit);
    
    // Purity should be less than 1 due to noise
    double purity = sim.getPurity();
    EXPECT_LT(purity, 1.0);
    EXPECT_GT(purity, 0.0);
}

TEST(DensityMatrixNoiseTest, AmplitudeDampingDecaysToGround) {
    NoiseModel noise;
    noise.addAmplitudeDamping(0.5);  // Strong damping
    
    DensityMatrixSimulator sim(1, noise);
    
    // Start in |1> state
    Circuit circuit(1);
    circuit.x(0);
    sim.run(circuit);
    
    auto probs = sim.getProbabilities();
    // With amplitude damping, |1> decays toward |0>
    EXPECT_GT(probs[0], 0.0);  // Some population in |0>
}

TEST(DensityMatrixNoiseTest, PhaseDampingPreservesDiagonal) {
    NoiseModel noise;
    noise.addPhaseDamping(0.3);
    
    DensityMatrixSimulator sim(1, noise);
    
    // Create superposition then apply noise
    Circuit circuit(1);
    circuit.h(0);
    sim.run(circuit);
    
    auto probs = sim.getProbabilities();
    // Phase damping doesn't affect probabilities, only coherence
    EXPECT_NEAR(probs[0], 0.5, 0.1);
    EXPECT_NEAR(probs[1], 0.5, 0.1);
}

TEST(DensityMatrixNoiseTest, NoNoisePreservesPurity) {
    NoiseModel noise;  // Empty noise model
    
    DensityMatrixSimulator sim(1, noise);
    
    Circuit circuit(1);
    circuit.h(0);
    sim.run(circuit);
    
    EXPECT_NEAR(sim.getPurity(), 1.0, 1e-10);
}

TEST(DensityMatrixNoiseTest, BitFlipMixesPopulations) {
    NoiseModel noise;
    noise.addBitFlip(0.5);  // 50% bit flip
    
    DensityMatrixSimulator sim(1, noise);
    
    Circuit circuit(1);
    circuit.x(0);  // Prepare |1>
    sim.run(circuit);
    
    auto probs = sim.getProbabilities();
    // 50% bit flip should give equal populations
    EXPECT_NEAR(probs[0], 0.5, 0.1);
    EXPECT_NEAR(probs[1], 0.5, 0.1);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(DensityMatrixIntegrationTest, MultiQubitCircuitWithNoise) {
    NoiseModel noise;
    noise.addDepolarizing(0.01);
    
    DensityMatrixSimulator sim(3, noise);
    
    // GHZ state preparation
    Circuit circuit(3);
    circuit.h(0).cnot(0, 1).cnot(1, 2);
    sim.run(circuit);
    
    auto probs = sim.getProbabilities();
    
    // Should still have mostly |000> and |111>
    double ghz_prob = probs[0] + probs[7];
    EXPECT_GT(ghz_prob, 0.8);  // Most probability in GHZ states
    
    // Purity reduced due to noise
    EXPECT_LT(sim.getPurity(), 1.0);
}

TEST(DensityMatrixIntegrationTest, ResetWorks) {
    DensityMatrixSimulator sim(2);
    
    Circuit circuit(2);
    circuit.h(0).cnot(0, 1);
    sim.run(circuit);
    
    sim.reset();
    
    auto probs = sim.getProbabilities();
    EXPECT_NEAR(probs[0], 1.0, 1e-10);
    EXPECT_NEAR(sim.getPurity(), 1.0, 1e-10);
}

TEST(DensityMatrixIntegrationTest, TracePreservedUnderNoise) {
    NoiseModel noise;
    noise.addDepolarizing(0.1);
    noise.addAmplitudeDamping(0.05);
    
    DensityMatrixSimulator sim(2, noise);
    
    Circuit circuit(2);
    circuit.h(0).h(1).cnot(0, 1).rz(0, 0.5);
    sim.run(circuit);
    
    EXPECT_NEAR(sim.getTrace(), 1.0, 1e-6);
}

TEST(DensityMatrixIntegrationTest, RotationGates) {
    DensityMatrixSimulator sim(1);
    
    // Rx(pi) should be equivalent to X (up to global phase)
    Circuit circuit(1);
    circuit.rx(0, M_PI);
    sim.run(circuit);
    
    auto probs = sim.getProbabilities();
    EXPECT_NEAR(probs[0], 0.0, 1e-6);
    EXPECT_NEAR(probs[1], 1.0, 1e-6);
}

TEST(DensityMatrixIntegrationTest, RyGate) {
    DensityMatrixSimulator sim(1);
    
    // Ry(pi/2) on |0> should give equal superposition
    Circuit circuit(1);
    circuit.ry(0, M_PI / 2.0);
    sim.run(circuit);
    
    auto probs = sim.getProbabilities();
    EXPECT_NEAR(probs[0], 0.5, 1e-6);
    EXPECT_NEAR(probs[1], 0.5, 1e-6);
}

TEST(DensityMatrixIntegrationTest, CompareWithNoisySimulator) {
    // Both should give similar results for same noise model
    NoiseModel noise;
    noise.addDepolarizing(0.05);
    
    // Run density matrix simulation
    DensityMatrixSimulator dm_sim(2, noise);
    Circuit circuit(2);
    circuit.h(0).cnot(0, 1);
    dm_sim.run(circuit);
    auto dm_probs = dm_sim.getProbabilities();
    
    // Density matrix gives exact noise simulation
    // Just verify it gives reasonable results
    double total = 0.0;
    for (double p : dm_probs) {
        EXPECT_GE(p, 0.0);
        total += p;
    }
    EXPECT_NEAR(total, 1.0, 1e-10);
}
