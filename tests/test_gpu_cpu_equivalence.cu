// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Rylan Malarchick

/**
 * GPU vs CPU Equivalence Tests
 * 
 * These tests verify that the GPU and CPU simulators produce identical
 * results for the same circuits. This is critical for:
 * 1. Validating GPU kernel correctness
 * 2. Ensuring the CPU baseline is trustworthy for benchmarking
 * 3. Catching subtle numerical or algorithmic differences
 * 
 * We test with various circuit structures and sizes.
 */

#include <gtest/gtest.h>
#include "Simulator.hpp"
#include <cmath>
#include <complex>
#include <random>

using namespace qsim;

namespace {

constexpr double TOLERANCE = 1e-12;
constexpr double PI = 3.14159265358979323846;

/**
 * Compare two state vectors element-wise
 */
::testing::AssertionResult statesEqual(
    const std::vector<std::complex<double>>& gpu_state,
    const std::vector<std::complex<double>>& cpu_state,
    double tol = TOLERANCE
) {
    if (gpu_state.size() != cpu_state.size()) {
        return ::testing::AssertionFailure() 
            << "Size mismatch: GPU=" << gpu_state.size() 
            << ", CPU=" << cpu_state.size();
    }
    
    for (size_t i = 0; i < gpu_state.size(); ++i) {
        double real_diff = std::abs(gpu_state[i].real() - cpu_state[i].real());
        double imag_diff = std::abs(gpu_state[i].imag() - cpu_state[i].imag());
        
        if (real_diff > tol || imag_diff > tol) {
            return ::testing::AssertionFailure()
                << "Mismatch at index " << i << ":\n"
                << "  GPU: (" << gpu_state[i].real() << ", " << gpu_state[i].imag() << ")\n"
                << "  CPU: (" << cpu_state[i].real() << ", " << cpu_state[i].imag() << ")\n"
                << "  Diff: real=" << real_diff << ", imag=" << imag_diff;
        }
    }
    
    return ::testing::AssertionSuccess();
}

/**
 * Compare probability distributions
 */
::testing::AssertionResult probabilitiesEqual(
    const std::vector<double>& gpu_probs,
    const std::vector<double>& cpu_probs,
    double tol = TOLERANCE
) {
    if (gpu_probs.size() != cpu_probs.size()) {
        return ::testing::AssertionFailure()
            << "Size mismatch: GPU=" << gpu_probs.size()
            << ", CPU=" << cpu_probs.size();
    }
    
    for (size_t i = 0; i < gpu_probs.size(); ++i) {
        double diff = std::abs(gpu_probs[i] - cpu_probs[i]);
        if (diff > tol) {
            return ::testing::AssertionFailure()
                << "Probability mismatch at index " << i << ":\n"
                << "  GPU: " << gpu_probs[i] << "\n"
                << "  CPU: " << cpu_probs[i] << "\n"
                << "  Diff: " << diff;
        }
    }
    
    return ::testing::AssertionSuccess();
}

}  // namespace

class GPUvsCPUTest : public ::testing::Test {
protected:
    void compareSimulators(const Circuit& circuit) {
        Simulator gpu_sim(circuit.getNumQubits());
        CPUSimulator cpu_sim(circuit.getNumQubits());
        
        gpu_sim.run(circuit);
        cpu_sim.run(circuit);
        
        auto gpu_state = gpu_sim.getStateVector();
        auto cpu_state = cpu_sim.getStateVector();
        
        EXPECT_TRUE(statesEqual(gpu_state, cpu_state));
    }
    
    void compareProbabilities(const Circuit& circuit) {
        Simulator gpu_sim(circuit.getNumQubits());
        CPUSimulator cpu_sim(circuit.getNumQubits());
        
        gpu_sim.run(circuit);
        cpu_sim.run(circuit);
        
        auto gpu_probs = gpu_sim.getProbabilities();
        auto cpu_probs = cpu_sim.getProbabilities();
        
        EXPECT_TRUE(probabilitiesEqual(gpu_probs, cpu_probs));
    }
};

// ============================================================================
// Single-qubit gates
// ============================================================================

TEST_F(GPUvsCPUTest, SingleQubitGates_AllTypes) {
    const int n = 3;
    
    // Test each gate type individually
    std::vector<std::function<void(Circuit&, int)>> gates = {
        [](Circuit& c, int q) { c.x(q); },
        [](Circuit& c, int q) { c.y(q); },
        [](Circuit& c, int q) { c.z(q); },
        [](Circuit& c, int q) { c.h(q); },
        [](Circuit& c, int q) { c.s(q); },
        [](Circuit& c, int q) { c.t(q); },
        [](Circuit& c, int q) { c.sdag(q); },
        [](Circuit& c, int q) { c.tdag(q); },
        [](Circuit& c, int q) { c.rx(q, PI/3); },
        [](Circuit& c, int q) { c.ry(q, PI/5); },
        [](Circuit& c, int q) { c.rz(q, PI/7); },
    };
    
    for (size_t g = 0; g < gates.size(); ++g) {
        for (int q = 0; q < n; ++q) {
            Circuit c(n);
            // First create non-trivial state
            c.h(0).h(1).h(2);
            // Then apply the gate being tested
            gates[g](c, q);
            
            SCOPED_TRACE("Gate index " + std::to_string(g) + ", qubit " + std::to_string(q));
            compareSimulators(c);
        }
    }
}

// ============================================================================
// Two-qubit gates
// ============================================================================

TEST_F(GPUvsCPUTest, CNOT_AllQubitPairs) {
    const int n = 4;
    
    for (int control = 0; control < n; ++control) {
        for (int target = 0; target < n; ++target) {
            if (control == target) continue;
            
            Circuit c(n);
            c.h(0).h(1).h(2).h(3);  // Superposition
            c.cnot(control, target);
            
            SCOPED_TRACE("CNOT(" + std::to_string(control) + ", " + std::to_string(target) + ")");
            compareSimulators(c);
        }
    }
}

TEST_F(GPUvsCPUTest, CZ_AllQubitPairs) {
    const int n = 4;
    
    for (int q1 = 0; q1 < n; ++q1) {
        for (int q2 = 0; q2 < n; ++q2) {
            if (q1 == q2) continue;
            
            Circuit c(n);
            c.h(0).h(1).h(2).h(3);
            c.cz(q1, q2);
            
            SCOPED_TRACE("CZ(" + std::to_string(q1) + ", " + std::to_string(q2) + ")");
            compareSimulators(c);
        }
    }
}

TEST_F(GPUvsCPUTest, SWAP_AllQubitPairs) {
    const int n = 4;
    
    for (int q1 = 0; q1 < n; ++q1) {
        for (int q2 = q1 + 1; q2 < n; ++q2) {
            Circuit c(n);
            // Create asymmetric state
            c.h(0).t(1).s(2).x(3);
            c.swap(q1, q2);
            
            SCOPED_TRACE("SWAP(" + std::to_string(q1) + ", " + std::to_string(q2) + ")");
            compareSimulators(c);
        }
    }
}

// ============================================================================
// Standard circuit patterns
// ============================================================================

TEST_F(GPUvsCPUTest, BellState) {
    compareSimulators(createBellCircuit());
}

TEST_F(GPUvsCPUTest, GHZState_Various_Sizes) {
    for (int n = 2; n <= 8; ++n) {
        SCOPED_TRACE("GHZ with " + std::to_string(n) + " qubits");
        compareSimulators(createGHZCircuit(n));
    }
}

// ============================================================================
// Random circuits
// ============================================================================

TEST_F(GPUvsCPUTest, RandomCircuits_Small) {
    for (unsigned int seed = 0; seed < 20; ++seed) {
        int n_qubits = 3 + (seed % 3);  // 3-5 qubits
        int depth = 10 + (seed % 20);   // 10-29 gates
        
        Circuit c = createRandomCircuit(n_qubits, depth, seed);
        
        SCOPED_TRACE("Random circuit: n=" + std::to_string(n_qubits) + 
                     ", depth=" + std::to_string(depth) + ", seed=" + std::to_string(seed));
        compareSimulators(c);
    }
}

TEST_F(GPUvsCPUTest, RandomCircuits_Medium) {
    for (unsigned int seed = 0; seed < 10; ++seed) {
        int n_qubits = 8 + (seed % 4);  // 8-11 qubits
        int depth = 50 + (seed % 50);   // 50-99 gates
        
        Circuit c = createRandomCircuit(n_qubits, depth, seed);
        
        SCOPED_TRACE("Random circuit: n=" + std::to_string(n_qubits) +
                     ", depth=" + std::to_string(depth) + ", seed=" + std::to_string(seed));
        compareSimulators(c);
    }
}

TEST_F(GPUvsCPUTest, RandomCircuits_Deep) {
    // Test deep circuits to check for accumulated numerical error
    for (unsigned int seed = 0; seed < 5; ++seed) {
        int n_qubits = 4;
        int depth = 500;  // Very deep
        
        Circuit c = createRandomCircuit(n_qubits, depth, seed);
        
        SCOPED_TRACE("Deep random circuit: seed=" + std::to_string(seed));
        // Use slightly looser tolerance for deep circuits
        Simulator gpu_sim(n_qubits);
        CPUSimulator cpu_sim(n_qubits);
        
        gpu_sim.run(c);
        cpu_sim.run(c);
        
        auto gpu_probs = gpu_sim.getProbabilities();
        auto cpu_probs = cpu_sim.getProbabilities();
        
        // For deep circuits, allow slightly more numerical drift
        EXPECT_TRUE(probabilitiesEqual(gpu_probs, cpu_probs, 1e-10));
    }
}

// ============================================================================
// Rotation gates with various angles
// ============================================================================

TEST_F(GPUvsCPUTest, RotationGates_VariousAngles) {
    std::vector<double> angles = {
        0.0, PI/8, PI/4, PI/3, PI/2, 2*PI/3, PI, 3*PI/2, 2*PI,
        0.1, 0.7, 1.23, 2.5, 4.0, 5.5
    };
    
    for (double theta : angles) {
        // Rx
        {
            Circuit c(2);
            c.h(0).h(1).rx(0, theta).cnot(0, 1);
            SCOPED_TRACE("Rx(" + std::to_string(theta) + ")");
            compareSimulators(c);
        }
        
        // Ry
        {
            Circuit c(2);
            c.h(0).h(1).ry(0, theta).cnot(0, 1);
            SCOPED_TRACE("Ry(" + std::to_string(theta) + ")");
            compareSimulators(c);
        }
        
        // Rz
        {
            Circuit c(2);
            c.h(0).h(1).rz(0, theta).cnot(0, 1);
            SCOPED_TRACE("Rz(" + std::to_string(theta) + ")");
            compareSimulators(c);
        }
    }
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_F(GPUvsCPUTest, EmptyCircuit) {
    Circuit c(4);  // No gates
    compareSimulators(c);
}

TEST_F(GPUvsCPUTest, SingleGateCircuits) {
    for (int n = 1; n <= 5; ++n) {
        Circuit c(n);
        c.h(0);
        SCOPED_TRACE("Single H on " + std::to_string(n) + " qubits");
        compareSimulators(c);
    }
}

TEST_F(GPUvsCPUTest, IdentitySequence) {
    // Circuit that should return to |0...0⟩
    Circuit c(3);
    c.h(0).h(1).h(2);
    c.h(0).h(1).h(2);  // H² = I
    
    compareSimulators(c);
}
