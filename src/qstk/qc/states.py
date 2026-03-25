"""Quantum state preparation for Bell/CHSH experiments.

All states are returned as numpy column vectors (shape (N, 1) or (N,)).
2-qubit states live in C^4, single-qubit in C^2.
"""

import numpy as np
from typing import Optional, List, Tuple


# Computational basis states
_ket0 = np.array([1, 0], dtype=complex)
_ket1 = np.array([0, 1], dtype=complex)


def computational_basis(n_qubits: int = 2) -> dict:
    """Return all computational basis states for n qubits.

    Returns
    -------
    dict mapping binary string labels to state vectors.
    e.g. {"00": array, "01": array, "10": array, "11": array}
    """
    dim = 2 ** n_qubits
    states = {}
    for i in range(dim):
        label = format(i, f"0{n_qubits}b")
        vec = np.zeros(dim, dtype=complex)
        vec[i] = 1.0
        states[label] = vec
    return states


def bell_state(which: str = "phi_plus") -> np.ndarray:
    """Prepare one of the four maximally entangled Bell states.

    Parameters
    ----------
    which : str
        One of "phi_plus", "phi_minus", "psi_plus", "psi_minus".
        Also accepts shorthand: "Φ+", "Φ-", "Ψ+", "Ψ-".

    Returns
    -------
    Normalized 4-component state vector.
    """
    ket00 = np.kron(_ket0, _ket0)
    ket01 = np.kron(_ket0, _ket1)
    ket10 = np.kron(_ket1, _ket0)
    ket11 = np.kron(_ket1, _ket1)

    mapping = {
        "phi_plus": (ket00 + ket11) / np.sqrt(2),
        "phi_minus": (ket00 - ket11) / np.sqrt(2),
        "psi_plus": (ket01 + ket10) / np.sqrt(2),
        "psi_minus": (ket01 - ket10) / np.sqrt(2),
        "Φ+": (ket00 + ket11) / np.sqrt(2),
        "Φ-": (ket00 - ket11) / np.sqrt(2),
        "Ψ+": (ket01 + ket10) / np.sqrt(2),
        "Ψ-": (ket01 - ket10) / np.sqrt(2),
    }

    if which not in mapping:
        raise ValueError(
            f"Unknown Bell state '{which}'. "
            f"Use one of: {list(mapping.keys())}"
        )
    return mapping[which]


def semantic_state(
    amplitudes: List[complex],
    labels: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Create a semantic superposition state from amplitudes.

    Maps word sense probabilities into qubit amplitudes.
    For 2 senses, uses 1 qubit. For 4 senses (2 words x 2 meanings), uses 2 qubits.

    Parameters
    ----------
    amplitudes : list of complex
        Raw amplitudes (will be normalized).
        For word sense disambiguation with 2 words each having 2 meanings:
        [α_AA, α_AB, α_BA, α_BB] where A/B are the two meanings.
    labels : list of str, optional
        Human-readable labels for each basis state.

    Returns
    -------
    (state_vector, labels) tuple.

    Example
    -------
    >>> # "bank" biased toward financial (A), "bat" toward animal (B)
    >>> state, labels = semantic_state(
    ...     [0.8, 0.1, 0.3, 0.7],
    ...     labels=["financial_baseball", "financial_animal", "river_baseball", "river_animal"]
    ... )
    """
    amps = np.array(amplitudes, dtype=complex)
    norm = np.linalg.norm(amps)
    if norm == 0:
        raise ValueError("Amplitudes cannot all be zero")
    state = amps / norm
    return state, labels


def werner_state(
    bell_state_vec: np.ndarray,
    p: float = 1.0,
) -> np.ndarray:
    """Create a Werner state (mixture of Bell state and maximally mixed state).

    ρ = p |ψ⟩⟨ψ| + (1-p) I/4

    Parameters
    ----------
    bell_state_vec : array
        The pure Bell state vector.
    p : float
        Entanglement strength parameter in [0, 1].
        p=1 is pure Bell state, p=0 is maximally mixed.

    Returns
    -------
    4x4 density matrix.
    """
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}")
    pure = np.outer(bell_state_vec, bell_state_vec.conj())
    mixed = np.eye(4, dtype=complex) / 4
    return p * pure + (1 - p) * mixed


def parameterized_entangled_state(
    theta: float,
    phi: float = 0.0,
) -> np.ndarray:
    """Create a parameterized 2-qubit entangled state.

    |ψ(θ, φ)⟩ = cos(θ)|00⟩ + e^{iφ} sin(θ)|11⟩

    Parameters
    ----------
    theta : float
        Controls entanglement degree. θ=π/4 gives maximally entangled.
    phi : float
        Relative phase between components.

    Returns
    -------
    4-component state vector.
    """
    ket00 = np.kron(_ket0, _ket0)
    ket11 = np.kron(_ket1, _ket1)
    return np.cos(theta) * ket00 + np.exp(1j * phi) * np.sin(theta) * ket11
