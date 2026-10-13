"""Quantum NLP: encode word-sense ambiguity as a 2-qubit state.

Treat two ambiguous words as two qubits:
  - qubit 0: bank (0=financial, 1=river)
  - qubit 1: bat  (0=baseball, 1=animal)

The amplitude vector [a00, a01, a10, a11] describes how likely each
meaning pair is. A strongly correlated pair (financial+bat, river+animal)
produces an entangled state and can violate the CHSH inequality.

This script:
  1. Builds three semantic states (correlated, uniform, anti-correlated)
  2. Runs each through a real Qiskit Aer quantum circuit via state initialization
  3. Reports the CHSH S-value and whether it exceeds the classical bound |S|=2
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qstk.qc import semantic_circuit
from qstk.qc.operators import alice_operators, bob_operators


CASES = {
    "correlated": {
        "amplitudes": [0.7, 0.05, 0.05, 0.7],
        "description": "financial+bat and river+animal meanings go together",
    },
    "uniform": {
        "amplitudes": [0.5, 0.5, 0.5, 0.5],
        "description": "all four meaning pairs are equally likely",
    },
    "anti-correlated": {
        "amplitudes": [0.05, 0.7, 0.7, 0.05],
        "description": "financial+animal and river+baseball meanings go together",
    },
}


def _build_semantic_chsh_circuits(amplitudes, a0, a1, b0, b1):
    """Build 4 Qiskit CHSH circuits initialized from a semantic state."""
    state = np.array(amplitudes, dtype=complex)
    state = state / np.linalg.norm(state)

    settings = {
        "A_B": (a0, b0),
        "A_B_prime": (a0, b1),
        "A_prime_B": (a1, b0),
        "A_prime_B_prime": (a1, b1),
    }

    circuits = {}
    for label, (alice_angle, bob_angle) in settings.items():
        qc = QuantumCircuit(2, 2)
        qc.initialize(state, [0, 1])

        if abs(alice_angle) > 1e-12:
            qc.ry(-alice_angle, 0)
        if abs(bob_angle) > 1e-12:
            qc.ry(-bob_angle, 1)

        qc.measure(0, 0)
        qc.measure(1, 1)
        circuits[label] = qc

    return circuits


def _s_from_counts(all_counts):
    ev = {}
    for label, counts in all_counts.items():
        agree = counts.get("00", 0) + counts.get("11", 0)
        disagree = counts.get("01", 0) + counts.get("10", 0)
        total = agree + disagree
        ev[label] = (agree - disagree) / total if total else 0.0
    return ev["A_B"] - ev["A_B_prime"] + ev["A_prime_B"] + ev["A_prime_B_prime"]


def run_qiskit_semantic(amplitudes, n_shots=8192, seed=42):
    A0, A1 = alice_operators()
    B0, B1 = bob_operators()
    a0, a1 = 0.0, np.pi / 2
    b0, b1 = np.pi / 4, 3 * np.pi / 4

    circuits = _build_semantic_chsh_circuits(amplitudes, a0, a1, b0, b1)
    simulator = AerSimulator(seed_simulator=seed)

    all_counts = {}
    for label, qc in circuits.items():
        transpiled = transpile(qc, simulator)
        job = simulator.run(transpiled, shots=n_shots)
        counts = job.result().get_counts(transpiled)
        all_counts[label] = {k: int(v) for k, v in counts.items()}

    return _s_from_counts(all_counts)


def main():
    print("=" * 70)
    print("Quantum NLP: word-sense correlations as 2-qubit states")
    print("=" * 70)
    print()

    for label, info in CASES.items():
        amps = info["amplitudes"]
        description = info["description"]
        np_result = semantic_circuit(amps, n_shots=0)
        qiskit_s = run_qiskit_semantic(amps)

        print(f"{label.upper()} — {description}")
        print(f"  amplitudes: {amps}")
        print(f"  concurrence: {np_result['concurrence']:.3f}")
        print(f"  entanglement entropy: {np_result['entanglement_entropy']:.3f} bits")
        print(f"  exact S (numpy):     {np_result['s_value']:+.4f}")
        print(f"  Qiskit Aer S:        {qiskit_s:+.4f}")
        print(f"  violates |S|>2:      {abs(qiskit_s) > 2.0}")
        print()

    print("=" * 70)
    print("Conclusion")
    print("=" * 70)
    print("Strongly correlated word senses (|00> + |11>) produce entanglement")
    print("and CHSH violation with the default measurement bases. Anti-correlated")
    print("senses (|01> + |10>) are also entangled but need rotated bases to")
    print("violate. Uniform senses are separable and stay classical.")


if __name__ == "__main__":
    main()
