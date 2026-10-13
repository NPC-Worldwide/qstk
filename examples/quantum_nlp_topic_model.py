"""Quantum NLP: topic correlation as a 2-qubit mixed state.

Imagine two topics, each represented by a qubit:
  - qubit 0: finance vs. nature
  - qubit 1: sports vs. biology

Given a document, estimate the joint topic probabilities:
  p00 = P(finance, sports)
  p01 = P(finance, biology)
  p10 = P(nature, sports)
  p11 = P(nature, biology)

These probabilities define a density matrix ρ. We then run the CHSH
Bell operator on it through a Qiskit circuit and compare the measured
S-value to the classical bound |S|=2.

This is a quantum-style topic-modeling primitive: if the topics are
strongly enough correlated, the state is entangled and violates CHSH.
"""

import numpy as np
from qstk.qc import chsh_s_value
from qstk.qc.measure import density_matrix
from qstk.qc.operators import alice_operators, bob_operators


TOPIC_STATES = {
    "sports-science": {
        "label": "sports and biology correlated",
        "probs": [0.45, 0.05, 0.05, 0.45],  # |00> and |11> dominate
    },
    "uniform": {
        "label": "uncorrelated topics",
        "probs": [0.25, 0.25, 0.25, 0.25],
    },
    "mixed-noise": {
        "label": "weakly correlated with noise",
        "probs": [0.35, 0.15, 0.15, 0.35],
    },
}


def build_density_matrix(probs):
    """Build a 4x4 diagonal density matrix from topic joint probabilities."""
    return np.diag(probs).astype(complex)


def main():
    print("=" * 70)
    print("Quantum NLP: topic correlation as a mixed quantum state")
    print("=" * 70)
    print()

    A0, A1 = alice_operators()
    B0, B1 = bob_operators()

    for name, info in TOPIC_STATES.items():
        probs = info["probs"]
        rho = build_density_matrix(probs)

        s = chsh_s_value(rho, A0, A1, B0, B1, is_density_matrix=True)
        viol = abs(s) > 2.0

        print(f"{name}: {info['label']}")
        print(f"  probabilities: {probs}")
        print(f"  CHSH S = {s:+.4f}")
        print(f"  violates |S|>2: {viol}")
        print()

    print("=" * 70)
    print("Next step: diagonal ρ values can come from an LLM or TF-IDF model;")
    print("then submit the corresponding state-preparation circuit to Qiskit Aer.")


if __name__ == "__main__":
    main()
