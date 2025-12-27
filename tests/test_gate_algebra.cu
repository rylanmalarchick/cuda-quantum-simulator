/**
 * Gate Algebra Tests
 * 
 * Tests for mathematical identities that quantum gates must satisfy.
 * These are fundamental correctness tests - if any fail, the simulator
 * is producing physically incorrect results.
 * 
 * Key identities tested:
 * - Involutions: X² = Y² = Z² = H² = I
 * - Phase gates: S² = Z, T² = S, T⁸ = I
 * - Adjoints: S†S = I, T†T = I
 * - Rotation periodicity: Rx(2π) = Ry(2π) = Rz(2π) = I (up to global phase)
 * - Controlled gate properties: CNOT² = I, CZ² = I
 * - Decompositions: CNOT = H₂·CZ·H₂ (where H₂ is H on target)
 */

#include <gtest/gtest.h>
#include "Simulator.hpp"
#include <cmath>
#include <complex>
#include <random>

using namespace qsim;

namespace {

// Tolerance for floating point comparisons
// Double precision gives ~15-16 significant digits
// We use 1e-12 to allow for accumulated floating point error
constexpr double TOLERANCE = 1e-12;

// Mathematical constants
constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;

/**
 * Helper: Check if two state vectors are equal (up to global phase)
 * 
 * Two states |ψ⟩ and |φ⟩ are equivalent if |ψ⟩ = e^(iθ)|φ⟩ for some θ.
 * We check this by finding the first non-zero amplitude and computing
 * the phase difference, then verifying all other amplitudes match.
 */
bool statesEqualUpToGlobalPhase(
    const std::vector<std::complex<double>>& a,
    const std::vector<std::complex<double>>& b,
    double tol = TOLERANCE
) {
    if (a.size() != b.size()) return false;
    
    // Find first non-negligible amplitude in 'a'
    std::complex<double> phase(1.0, 0.0);
    bool found_phase = false;
    
    for (size_t i = 0; i < a.size(); ++i) {
        double mag_a = std::abs(a[i]);
        double mag_b = std::abs(b[i]);
        
        if (mag_a > tol || mag_b > tol) {
            if (mag_a < tol || mag_b < tol) {
                // One is zero, other isn't
                return false;
            }
            if (!found_phase) {
                // Compute phase: phase * a[i] = b[i] => phase = b[i] / a[i]
                phase = b[i] / a[i];
                found_phase = true;
            }
            // Check: phase * a[i] ≈ b[i]
            std::complex<double> expected = phase * a[i];
            if (std::abs(expected - b[i]) > tol) {
                return false;
            }
        }
    }
    
    return true;
}

/**
 * Helper: Check if state equals |0...0⟩ (up to global phase)
 */
bool isZeroState(const std::vector<std::complex<double>>& state, double tol = TOLERANCE) {
    if (state.empty()) return false;
    
    // First amplitude should have magnitude 1
    if (std::abs(std::abs(state[0]) - 1.0) > tol) return false;
    
    // All other amplitudes should be 0
    for (size_t i = 1; i < state.size(); ++i) {
        if (std::abs(state[i]) > tol) return false;
    }
    
    return true;
}

/**
 * Helper: Create a random initial state for testing
 */
void applyRandomState(Simulator& sim, int num_qubits, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> angle_dist(0.0, TWO_PI);
    
    Circuit c(num_qubits);
    for (int q = 0; q < num_qubits; ++q) {
        c.ry(q, angle_dist(rng));
        c.rz(q, angle_dist(rng));
    }
    sim.run(c);
}

}  // namespace

class GateAlgebraTest : public ::testing::Test {
protected:
    // Run a circuit and return final state
    std::vector<std::complex<double>> runCircuit(const Circuit& c) {
        Simulator sim(c.getNumQubits());
        sim.run(c);
        return sim.getStateVector();
    }
    
    // Get identity state (|0...0⟩) for n qubits
    std::vector<std::complex<double>> identityState(int n_qubits) {
        Simulator sim(n_qubits);
        return sim.getStateVector();
    }
};

// ============================================================================
// Involutions: G² = I
// ============================================================================

TEST_F(GateAlgebraTest, X_Squared_Is_Identity) {
    for (int n = 1; n <= 4; ++n) {
        for (int q = 0; q < n; ++q) {
            Circuit c(n);
            c.x(q).x(q);
            EXPECT_TRUE(isZeroState(runCircuit(c)))
                << "X² ≠ I for qubit " << q << " in " << n << "-qubit system";
        }
    }
}

TEST_F(GateAlgebraTest, Y_Squared_Is_Identity) {
    for (int n = 1; n <= 4; ++n) {
        for (int q = 0; q < n; ++q) {
            Circuit c(n);
            c.y(q).y(q);
            // Y² = I, but Y|0⟩ = i|1⟩, so Y²|0⟩ = -|0⟩
            // This is still |0⟩ up to global phase
            EXPECT_TRUE(isZeroState(runCircuit(c)))
                << "Y² ≠ I (up to phase) for qubit " << q;
        }
    }
}

TEST_F(GateAlgebraTest, Z_Squared_Is_Identity) {
    for (int n = 1; n <= 4; ++n) {
        for (int q = 0; q < n; ++q) {
            Circuit c(n);
            c.z(q).z(q);
            EXPECT_TRUE(isZeroState(runCircuit(c)))
                << "Z² ≠ I for qubit " << q;
        }
    }
}

TEST_F(GateAlgebraTest, H_Squared_Is_Identity) {
    for (int n = 1; n <= 4; ++n) {
        for (int q = 0; q < n; ++q) {
            Circuit c(n);
            c.h(q).h(q);
            EXPECT_TRUE(isZeroState(runCircuit(c)))
                << "H² ≠ I for qubit " << q;
        }
    }
}

// ============================================================================
// Phase gate identities
// ============================================================================

TEST_F(GateAlgebraTest, S_Squared_Equals_Z) {
    // S² = Z
    // Apply S² to |+⟩ and Z to |+⟩, compare results
    for (int n = 1; n <= 3; ++n) {
        for (int q = 0; q < n; ++q) {
            Circuit c1(n);
            c1.h(q).s(q).s(q);
            
            Circuit c2(n);
            c2.h(q).z(q);
            
            EXPECT_TRUE(statesEqualUpToGlobalPhase(runCircuit(c1), runCircuit(c2)))
                << "S² ≠ Z for qubit " << q;
        }
    }
}

TEST_F(GateAlgebraTest, T_Squared_Equals_S) {
    // T² = S
    for (int n = 1; n <= 3; ++n) {
        for (int q = 0; q < n; ++q) {
            Circuit c1(n);
            c1.h(q).t(q).t(q);
            
            Circuit c2(n);
            c2.h(q).s(q);
            
            EXPECT_TRUE(statesEqualUpToGlobalPhase(runCircuit(c1), runCircuit(c2)))
                << "T² ≠ S for qubit " << q;
        }
    }
}

TEST_F(GateAlgebraTest, T_To_Eighth_Is_Identity) {
    // T⁸ = I (since T = e^(iπ/4), T⁸ = e^(i2π) = 1)
    for (int n = 1; n <= 3; ++n) {
        for (int q = 0; q < n; ++q) {
            Circuit c(n);
            c.h(q);  // Create superposition to see phase effects
            for (int i = 0; i < 8; ++i) c.t(q);
            c.h(q);  // Interfere to reveal phase
            
            EXPECT_TRUE(isZeroState(runCircuit(c)))
                << "T⁸ ≠ I for qubit " << q;
        }
    }
}

// ============================================================================
// Adjoint identities: G†G = I
// ============================================================================

TEST_F(GateAlgebraTest, S_Adjoint_Cancels_S) {
    for (int n = 1; n <= 3; ++n) {
        for (int q = 0; q < n; ++q) {
            Circuit c(n);
            c.h(q).s(q).sdag(q).h(q);
            
            EXPECT_TRUE(isZeroState(runCircuit(c)))
                << "S†S ≠ I for qubit " << q;
        }
    }
}

TEST_F(GateAlgebraTest, T_Adjoint_Cancels_T) {
    for (int n = 1; n <= 3; ++n) {
        for (int q = 0; q < n; ++q) {
            Circuit c(n);
            c.h(q).t(q).tdag(q).h(q);
            
            EXPECT_TRUE(isZeroState(runCircuit(c)))
                << "T†T ≠ I for qubit " << q;
        }
    }
}

TEST_F(GateAlgebraTest, Sdag_Sdag_Equals_Z) {
    // S†S† = S⁻² = Z⁻¹ = Z (since Z² = I)
    for (int q = 0; q < 2; ++q) {
        Circuit c1(2);
        c1.h(q).sdag(q).sdag(q);
        
        Circuit c2(2);
        c2.h(q).z(q);
        
        EXPECT_TRUE(statesEqualUpToGlobalPhase(runCircuit(c1), runCircuit(c2)))
            << "S†S† ≠ Z for qubit " << q;
    }
}

// ============================================================================
// Rotation gate periodicity
// ============================================================================

TEST_F(GateAlgebraTest, Rx_2Pi_Is_Identity_UpToPhase) {
    // Rx(2π) = -I (global phase of -1)
    Circuit c(1);
    c.rx(0, TWO_PI);
    
    auto state = runCircuit(c);
    // Should be -|0⟩, which equals |0⟩ up to global phase
    EXPECT_TRUE(isZeroState(state)) << "Rx(2π) ≠ ±I";
}

TEST_F(GateAlgebraTest, Ry_2Pi_Is_Identity_UpToPhase) {
    Circuit c(1);
    c.ry(0, TWO_PI);
    
    EXPECT_TRUE(isZeroState(runCircuit(c))) << "Ry(2π) ≠ ±I";
}

TEST_F(GateAlgebraTest, Rz_2Pi_Is_Identity_UpToPhase) {
    Circuit c(1);
    c.h(0).rz(0, TWO_PI).h(0);
    
    EXPECT_TRUE(isZeroState(runCircuit(c))) << "Rz(2π) ≠ ±I";
}

TEST_F(GateAlgebraTest, Rx_Pi_Equals_iX) {
    // Rx(π) = -iX
    // Rx(π)|0⟩ = -i|1⟩
    Circuit c1(1);
    c1.rx(0, PI);
    
    Circuit c2(1);
    c2.x(0);  // X|0⟩ = |1⟩
    
    auto s1 = runCircuit(c1);
    auto s2 = runCircuit(c2);
    
    // s1 = -i * s2 (up to global phase, both should give |1⟩)
    EXPECT_TRUE(statesEqualUpToGlobalPhase(s1, s2))
        << "Rx(π) and X don't produce same state (up to phase)";
}

TEST_F(GateAlgebraTest, Rz_Pi_Equals_Z_UpToPhase) {
    // Rz(π) = -iZ (up to global phase)
    Circuit c1(1);
    c1.h(0).rz(0, PI);
    
    Circuit c2(1);
    c2.h(0).z(0);
    
    EXPECT_TRUE(statesEqualUpToGlobalPhase(runCircuit(c1), runCircuit(c2)))
        << "Rz(π) ≠ Z (up to phase)";
}

// ============================================================================
// Two-qubit gate identities
// ============================================================================

TEST_F(GateAlgebraTest, CNOT_Squared_Is_Identity) {
    Circuit c(2);
    c.h(0);  // Create superposition
    c.cnot(0, 1).cnot(0, 1);
    c.h(0);
    
    EXPECT_TRUE(isZeroState(runCircuit(c))) << "CNOT² ≠ I";
}

TEST_F(GateAlgebraTest, CZ_Squared_Is_Identity) {
    Circuit c(2);
    c.h(0).h(1);  // Create superposition on both
    c.cz(0, 1).cz(0, 1);
    c.h(0).h(1);
    
    EXPECT_TRUE(isZeroState(runCircuit(c))) << "CZ² ≠ I";
}

TEST_F(GateAlgebraTest, SWAP_Squared_Is_Identity) {
    Circuit c(2);
    c.h(0);  // Asymmetric state
    c.swap(0, 1).swap(0, 1);
    c.h(0);
    
    EXPECT_TRUE(isZeroState(runCircuit(c))) << "SWAP² ≠ I";
}

TEST_F(GateAlgebraTest, CZ_Is_Symmetric) {
    // CZ(a, b) = CZ(b, a)
    Circuit c1(2);
    c1.h(0).h(1).cz(0, 1);
    
    Circuit c2(2);
    c2.h(0).h(1).cz(1, 0);
    
    EXPECT_TRUE(statesEqualUpToGlobalPhase(runCircuit(c1), runCircuit(c2)))
        << "CZ is not symmetric";
}

// ============================================================================
// Gate decompositions
// ============================================================================

TEST_F(GateAlgebraTest, CNOT_Decomposition_HCZ_H) {
    // CNOT = (I⊗H) · CZ · (I⊗H)
    // i.e., CNOT(c, t) = H(t) · CZ(c, t) · H(t)
    Circuit c1(2);
    c1.h(0);  // Create interesting state
    c1.cnot(0, 1);
    
    Circuit c2(2);
    c2.h(0);
    c2.h(1).cz(0, 1).h(1);  // Decomposed CNOT
    
    EXPECT_TRUE(statesEqualUpToGlobalPhase(runCircuit(c1), runCircuit(c2)))
        << "CNOT ≠ H·CZ·H decomposition";
}

TEST_F(GateAlgebraTest, SWAP_Decomposition_Three_CNOTs) {
    // SWAP = CNOT(a,b) · CNOT(b,a) · CNOT(a,b)
    Circuit c1(2);
    c1.h(0).t(1);  // Create asymmetric state
    c1.swap(0, 1);
    
    Circuit c2(2);
    c2.h(0).t(1);
    c2.cnot(0, 1).cnot(1, 0).cnot(0, 1);  // Decomposed SWAP
    
    EXPECT_TRUE(statesEqualUpToGlobalPhase(runCircuit(c1), runCircuit(c2)))
        << "SWAP ≠ CNOT³ decomposition";
}

// ============================================================================
// Commutation relations
// ============================================================================

TEST_F(GateAlgebraTest, X_Z_Anticommute) {
    // XZ = -ZX
    // XZ|+⟩ and -ZX|+⟩ should be equal
    Circuit c1(1);
    c1.h(0).x(0).z(0);
    
    Circuit c2(1);
    c2.h(0).z(0).x(0);
    
    auto s1 = runCircuit(c1);
    auto s2 = runCircuit(c2);
    
    // s1 = -s2
    for (size_t i = 0; i < s1.size(); ++i) {
        EXPECT_NEAR(s1[i].real(), -s2[i].real(), TOLERANCE);
        EXPECT_NEAR(s1[i].imag(), -s2[i].imag(), TOLERANCE);
    }
}

// ============================================================================
// Random state preservation tests
// These verify that identities hold for arbitrary input states
// ============================================================================

TEST_F(GateAlgebraTest, H_Squared_RandomState) {
    for (unsigned int seed = 0; seed < 10; ++seed) {
        Simulator sim(3);
        applyRandomState(sim, 3, seed);
        auto before = sim.getStateVector();
        
        Circuit c(3);
        for (int q = 0; q < 3; ++q) c.h(q).h(q);
        sim.run(c);
        auto after = sim.getStateVector();
        
        EXPECT_TRUE(statesEqualUpToGlobalPhase(before, after))
            << "H² ≠ I for random state (seed=" << seed << ")";
    }
}

TEST_F(GateAlgebraTest, CNOT_Squared_RandomState) {
    for (unsigned int seed = 0; seed < 10; ++seed) {
        Simulator sim(2);
        applyRandomState(sim, 2, seed);
        auto before = sim.getStateVector();
        
        Circuit c(2);
        c.cnot(0, 1).cnot(0, 1);
        sim.run(c);
        auto after = sim.getStateVector();
        
        EXPECT_TRUE(statesEqualUpToGlobalPhase(before, after))
            << "CNOT² ≠ I for random state (seed=" << seed << ")";
    }
}
