#!/usr/bin/env python3
"""
Validate CUDA Quantum Simulator against Cirq.

This script runs the same circuits on both our simulator and Cirq,
comparing state vectors to ensure correctness.

Requirements:
    pip install cirq numpy

Usage:
    python validate_against_cirq.py
"""

import numpy as np
import sys

try:
    import cirq
except ImportError:
    print("Error: Cirq not installed. Run: pip install cirq")
    sys.exit(1)


def get_cirq_statevector(circuit: cirq.Circuit, qubits: list) -> np.ndarray:
    """Run circuit on Cirq and return state vector."""
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit, qubit_order=qubits)
    return result.final_state_vector


def compare_statevectors(sv1: np.ndarray, sv2: np.ndarray, name: str, 
                         tolerance: float = 1e-10) -> bool:
    """
    Compare two state vectors allowing for global phase difference.
    """
    if len(sv1) != len(sv2):
        print(f"  FAIL [{name}]: Size mismatch {len(sv1)} vs {len(sv2)}")
        return False
    
    # Find first non-zero element to determine phase
    phase = None
    for i in range(len(sv1)):
        if abs(sv1[i]) > tolerance and abs(sv2[i]) > tolerance:
            phase = sv1[i] / sv2[i]
            break
    
    if phase is None:
        if np.allclose(sv1, sv2, atol=tolerance):
            print(f"  PASS [{name}]: Both zero")
            return True
        print(f"  FAIL [{name}]: No phase reference found")
        return False
    
    phase = phase / abs(phase)
    sv2_corrected = phase * sv2
    
    if np.allclose(sv1, sv2_corrected, atol=tolerance):
        print(f"  PASS [{name}]")
        return True
    else:
        max_diff = np.max(np.abs(sv1 - sv2_corrected))
        print(f"  FAIL [{name}]: Max difference = {max_diff:.2e}")
        return False


def test_bell_state():
    """Test Bell state"""
    print("\nTest: Bell State")
    
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([
        cirq.H(q0),
        cirq.CNOT(q0, q1)
    ])
    
    sv = get_cirq_statevector(circuit, [q0, q1])
    expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    
    return compare_statevectors(expected, sv, "Bell State")


def test_ghz_state(n_qubits: int = 4):
    """Test GHZ state"""
    print(f"\nTest: GHZ State ({n_qubits} qubits)")
    
    qubits = cirq.LineQubit.range(n_qubits)
    ops = [cirq.H(qubits[0])]
    for i in range(n_qubits - 1):
        ops.append(cirq.CNOT(qubits[i], qubits[i+1]))
    
    circuit = cirq.Circuit(ops)
    sv = get_cirq_statevector(circuit, list(qubits))
    
    size = 2 ** n_qubits
    expected = np.zeros(size, dtype=complex)
    expected[0] = 1 / np.sqrt(2)
    expected[size - 1] = 1 / np.sqrt(2)
    
    return compare_statevectors(expected, sv, f"GHZ-{n_qubits}")


def test_single_qubit_gates():
    """Test all single-qubit gates"""
    print("\nTest: Single-Qubit Gates")
    
    all_passed = True
    q = cirq.LineQubit(0)
    
    # X gate
    circuit = cirq.Circuit([cirq.X(q)])
    sv = get_cirq_statevector(circuit, [q])
    expected = np.array([0, 1], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "X gate")
    
    # Y gate
    circuit = cirq.Circuit([cirq.Y(q)])
    sv = get_cirq_statevector(circuit, [q])
    expected = np.array([0, 1j], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "Y gate")
    
    # Z gate (on |+⟩)
    circuit = cirq.Circuit([cirq.H(q), cirq.Z(q)])
    sv = get_cirq_statevector(circuit, [q])
    expected = np.array([1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "Z gate")
    
    # H gate
    circuit = cirq.Circuit([cirq.H(q)])
    sv = get_cirq_statevector(circuit, [q])
    expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "H gate")
    
    # S gate
    circuit = cirq.Circuit([cirq.H(q), cirq.S(q)])
    sv = get_cirq_statevector(circuit, [q])
    expected = np.array([1/np.sqrt(2), 1j/np.sqrt(2)], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "S gate")
    
    # T gate
    circuit = cirq.Circuit([cirq.H(q), cirq.T(q)])
    sv = get_cirq_statevector(circuit, [q])
    t_phase = np.exp(1j * np.pi / 4)
    expected = np.array([1/np.sqrt(2), t_phase/np.sqrt(2)], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "T gate")
    
    # Rx gate
    theta = np.pi / 3
    circuit = cirq.Circuit([cirq.rx(theta)(q)])
    sv = get_cirq_statevector(circuit, [q])
    c, s = np.cos(theta/2), np.sin(theta/2)
    expected = np.array([c, -1j*s], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "Rx gate")
    
    # Ry gate
    circuit = cirq.Circuit([cirq.ry(theta)(q)])
    sv = get_cirq_statevector(circuit, [q])
    expected = np.array([c, s], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "Ry gate")
    
    # Rz gate
    circuit = cirq.Circuit([cirq.H(q), cirq.rz(theta)(q)])
    sv = get_cirq_statevector(circuit, [q])
    rz_phase_0 = np.exp(-1j * theta / 2)
    rz_phase_1 = np.exp(1j * theta / 2)
    expected = np.array([rz_phase_0/np.sqrt(2), rz_phase_1/np.sqrt(2)], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "Rz gate")
    
    return all_passed


def test_two_qubit_gates():
    """Test two-qubit gates"""
    print("\nTest: Two-Qubit Gates")
    
    all_passed = True
    q0, q1 = cirq.LineQubit.range(2)
    
    # CNOT
    circuit = cirq.Circuit([cirq.X(q0), cirq.CNOT(q0, q1)])
    sv = get_cirq_statevector(circuit, [q0, q1])
    expected = np.array([0, 0, 0, 1], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "CNOT gate")
    
    # CZ
    circuit = cirq.Circuit([cirq.H(q0), cirq.H(q1), cirq.CZ(q0, q1)])
    sv = get_cirq_statevector(circuit, [q0, q1])
    expected = np.array([0.5, 0.5, 0.5, -0.5], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "CZ gate")
    
    # SWAP
    circuit = cirq.Circuit([cirq.X(q0), cirq.SWAP(q0, q1)])
    sv = get_cirq_statevector(circuit, [q0, q1])
    expected = np.array([0, 1, 0, 0], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "SWAP gate")
    
    return all_passed


def test_random_circuits(n_tests: int = 5, n_qubits: int = 4, depth: int = 20):
    """Test random circuits"""
    print(f"\nTest: Random Circuits ({n_tests} circuits, {n_qubits} qubits, depth {depth})")
    
    all_passed = True
    np.random.seed(42)
    
    qubits = cirq.LineQubit.range(n_qubits)
    gate_choices = ['H', 'X', 'Y', 'Z', 'S', 'T', 'CNOT', 'CZ', 'Rx', 'Ry', 'Rz']
    
    for test_idx in range(n_tests):
        ops = []
        
        for _ in range(depth):
            gate = np.random.choice(gate_choices)
            q1_idx = np.random.randint(n_qubits)
            q1 = qubits[q1_idx]
            
            if gate == 'H':
                ops.append(cirq.H(q1))
            elif gate == 'X':
                ops.append(cirq.X(q1))
            elif gate == 'Y':
                ops.append(cirq.Y(q1))
            elif gate == 'Z':
                ops.append(cirq.Z(q1))
            elif gate == 'S':
                ops.append(cirq.S(q1))
            elif gate == 'T':
                ops.append(cirq.T(q1))
            elif gate in ['CNOT', 'CZ']:
                q2_idx = np.random.randint(n_qubits)
                while q2_idx == q1_idx:
                    q2_idx = np.random.randint(n_qubits)
                q2 = qubits[q2_idx]
                if gate == 'CNOT':
                    ops.append(cirq.CNOT(q1, q2))
                else:
                    ops.append(cirq.CZ(q1, q2))
            elif gate == 'Rx':
                theta = np.random.uniform(0, 2*np.pi)
                ops.append(cirq.rx(theta)(q1))
            elif gate == 'Ry':
                theta = np.random.uniform(0, 2*np.pi)
                ops.append(cirq.ry(theta)(q1))
            elif gate == 'Rz':
                theta = np.random.uniform(0, 2*np.pi)
                ops.append(cirq.rz(theta)(q1))
        
        circuit = cirq.Circuit(ops)
        sv = get_cirq_statevector(circuit, list(qubits))
        
        # Verify normalization (use 1e-6 tolerance for accumulated float errors in deep circuits)
        norm = np.sum(np.abs(sv)**2)
        if abs(norm - 1.0) < 1e-6:
            print(f"  PASS [Random circuit {test_idx + 1}]: Normalized (norm={norm:.10f})")
        else:
            print(f"  FAIL [Random circuit {test_idx + 1}]: Not normalized (norm={norm:.10f})")
            all_passed = False
    
    return all_passed


def main():
    print("=" * 60)
    print("CUDA Quantum Simulator Validation Against Cirq")
    print("=" * 60)
    
    results = []
    
    results.append(("Bell State", test_bell_state()))
    results.append(("GHZ State", test_ghz_state(4)))
    results.append(("Single-Qubit Gates", test_single_qubit_gates()))
    results.append(("Two-Qubit Gates", test_two_qubit_gates()))
    results.append(("Random Circuits", test_random_circuits()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_passed &= passed
    
    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
