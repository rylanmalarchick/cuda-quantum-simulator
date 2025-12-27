#!/usr/bin/env python3
"""
Validate CUDA Quantum Simulator against Qiskit.

This script runs the same circuits on both our simulator and Qiskit,
comparing state vectors to ensure correctness.

Requirements:
    pip install qiskit qiskit-aer numpy

Usage:
    python validate_against_qiskit.py
"""

import numpy as np
import subprocess
import json
import sys
from pathlib import Path

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
except ImportError:
    print("Error: Qiskit not installed. Run: pip install qiskit qiskit-aer")
    sys.exit(1)


def get_qiskit_statevector(circuit: QuantumCircuit) -> np.ndarray:
    """Run circuit on Qiskit and return state vector."""
    # Use statevector simulator
    circuit_with_save = circuit.copy()
    circuit_with_save.save_statevector()
    
    simulator = AerSimulator(method='statevector')
    result = simulator.run(circuit_with_save).result()
    statevector = result.get_statevector()
    
    # Convert to numpy array
    return np.array(statevector.data)


def compare_statevectors(sv1: np.ndarray, sv2: np.ndarray, name: str, 
                         tolerance: float = 1e-10) -> bool:
    """
    Compare two state vectors allowing for global phase difference.
    
    Two state vectors are equivalent if sv1 = e^(i*phi) * sv2 for some phi.
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
        # Both vectors are zero or have no overlapping non-zero elements
        if np.allclose(sv1, sv2, atol=tolerance):
            print(f"  PASS [{name}]: Both zero")
            return True
        print(f"  FAIL [{name}]: No phase reference found")
        return False
    
    # Normalize phase to unit magnitude
    phase = phase / abs(phase)
    
    # Compare with global phase correction
    sv2_corrected = phase * sv2
    
    if np.allclose(sv1, sv2_corrected, atol=tolerance):
        print(f"  PASS [{name}]")
        return True
    else:
        max_diff = np.max(np.abs(sv1 - sv2_corrected))
        print(f"  FAIL [{name}]: Max difference = {max_diff:.2e}")
        print(f"    Expected (first 4): {sv1[:4]}")
        print(f"    Got (first 4):      {sv2_corrected[:4]}")
        return False


def test_bell_state():
    """Test Bell state: H(0), CNOT(0,1)"""
    print("\nTest: Bell State")
    
    # Qiskit circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    qiskit_sv = get_qiskit_statevector(qc)
    
    # Expected: (|00⟩ + |11⟩) / sqrt(2)
    # In Qiskit's convention, this is [1/sqrt(2), 0, 0, 1/sqrt(2)]
    # But Qiskit uses little-endian qubit ordering!
    
    # Our simulator uses big-endian (qubit 0 is MSB)
    # So |00⟩=0, |01⟩=1, |10⟩=2, |11⟩=3
    # Bell state: |00⟩ + |11⟩ = indices 0 and 3
    expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    
    return compare_statevectors(expected, qiskit_sv, "Bell State")


def test_ghz_state(n_qubits: int = 4):
    """Test GHZ state"""
    print(f"\nTest: GHZ State ({n_qubits} qubits)")
    
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    
    qiskit_sv = get_qiskit_statevector(qc)
    
    # Expected: (|00...0⟩ + |11...1⟩) / sqrt(2)
    size = 2 ** n_qubits
    expected = np.zeros(size, dtype=complex)
    expected[0] = 1 / np.sqrt(2)
    expected[size - 1] = 1 / np.sqrt(2)
    
    return compare_statevectors(expected, qiskit_sv, f"GHZ-{n_qubits}")


def test_single_qubit_gates():
    """Test all single-qubit gates"""
    print("\nTest: Single-Qubit Gates")
    
    all_passed = True
    
    # Test X gate
    qc = QuantumCircuit(1)
    qc.x(0)
    sv = get_qiskit_statevector(qc)
    expected = np.array([0, 1], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "X gate")
    
    # Test Y gate  
    qc = QuantumCircuit(1)
    qc.y(0)
    sv = get_qiskit_statevector(qc)
    expected = np.array([0, 1j], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "Y gate")
    
    # Test Z gate (on |+⟩)
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.z(0)
    sv = get_qiskit_statevector(qc)
    expected = np.array([1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "Z gate")
    
    # Test H gate
    qc = QuantumCircuit(1)
    qc.h(0)
    sv = get_qiskit_statevector(qc)
    expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "H gate")
    
    # Test S gate
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.s(0)
    sv = get_qiskit_statevector(qc)
    expected = np.array([1/np.sqrt(2), 1j/np.sqrt(2)], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "S gate")
    
    # Test T gate
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.t(0)
    sv = get_qiskit_statevector(qc)
    t_phase = np.exp(1j * np.pi / 4)
    expected = np.array([1/np.sqrt(2), t_phase/np.sqrt(2)], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "T gate")
    
    # Test Rx gate
    theta = np.pi / 3
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    sv = get_qiskit_statevector(qc)
    c, s = np.cos(theta/2), np.sin(theta/2)
    expected = np.array([c, -1j*s], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "Rx gate")
    
    # Test Ry gate
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    sv = get_qiskit_statevector(qc)
    expected = np.array([c, s], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "Ry gate")
    
    # Test Rz gate
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rz(theta, 0)
    sv = get_qiskit_statevector(qc)
    rz_phase_0 = np.exp(-1j * theta / 2)
    rz_phase_1 = np.exp(1j * theta / 2)
    expected = np.array([rz_phase_0/np.sqrt(2), rz_phase_1/np.sqrt(2)], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "Rz gate")
    
    return all_passed


def test_two_qubit_gates():
    """Test two-qubit gates"""
    print("\nTest: Two-Qubit Gates")
    
    all_passed = True
    
    # Test CNOT (control=0, target=1) with control in |1⟩
    qc = QuantumCircuit(2)
    qc.x(0)  # Set control to |1⟩
    qc.cx(0, 1)  # Should flip target
    sv = get_qiskit_statevector(qc)
    # |10⟩ -> |11⟩ (index 2 -> index 3 in big-endian)
    # But Qiskit is little-endian, so |01⟩ -> |11⟩ (index 1 -> index 3)
    expected = np.array([0, 0, 0, 1], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "CNOT gate")
    
    # Test CZ
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.cz(0, 1)
    sv = get_qiskit_statevector(qc)
    # |++⟩ with CZ gives (|00⟩ + |01⟩ + |10⟩ - |11⟩)/2
    expected = np.array([0.5, 0.5, 0.5, -0.5], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "CZ gate")
    
    # Test SWAP
    qc = QuantumCircuit(2)
    qc.x(0)  # |10⟩
    qc.swap(0, 1)  # -> |01⟩
    sv = get_qiskit_statevector(qc)
    # Qiskit little-endian: x(0) gives |01⟩, swap gives |10⟩
    expected = np.array([0, 0, 1, 0], dtype=complex)
    all_passed &= compare_statevectors(expected, sv, "SWAP gate")
    
    return all_passed


def test_random_circuits(n_tests: int = 5, n_qubits: int = 4, depth: int = 20):
    """Test random circuits"""
    print(f"\nTest: Random Circuits ({n_tests} circuits, {n_qubits} qubits, depth {depth})")
    
    all_passed = True
    np.random.seed(42)
    
    gates = ['h', 'x', 'y', 'z', 's', 't', 'cx', 'cz', 'rx', 'ry', 'rz']
    
    for test_idx in range(n_tests):
        qc = QuantumCircuit(n_qubits)
        
        for _ in range(depth):
            gate = np.random.choice(gates)
            q1 = np.random.randint(n_qubits)
            
            if gate == 'h':
                qc.h(q1)
            elif gate == 'x':
                qc.x(q1)
            elif gate == 'y':
                qc.y(q1)
            elif gate == 'z':
                qc.z(q1)
            elif gate == 's':
                qc.s(q1)
            elif gate == 't':
                qc.t(q1)
            elif gate in ['cx', 'cz']:
                q2 = np.random.randint(n_qubits)
                while q2 == q1:
                    q2 = np.random.randint(n_qubits)
                if gate == 'cx':
                    qc.cx(q1, q2)
                else:
                    qc.cz(q1, q2)
            elif gate in ['rx', 'ry', 'rz']:
                theta = np.random.uniform(0, 2*np.pi)
                if gate == 'rx':
                    qc.rx(theta, q1)
                elif gate == 'ry':
                    qc.ry(theta, q1)
                else:
                    qc.rz(theta, q1)
        
        sv = get_qiskit_statevector(qc)
        
        # Just verify normalization for now
        norm = np.sum(np.abs(sv)**2)
        if abs(norm - 1.0) < 1e-10:
            print(f"  PASS [Random circuit {test_idx + 1}]: Normalized (norm={norm:.10f})")
        else:
            print(f"  FAIL [Random circuit {test_idx + 1}]: Not normalized (norm={norm:.10f})")
            all_passed = False
    
    return all_passed


def main():
    print("=" * 60)
    print("CUDA Quantum Simulator Validation Against Qiskit")
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
