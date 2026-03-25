"""Quantum measurement routines for CHSH experiments.

Provides expectation value computation, CHSH S-value from quantum states,
state measurement/sampling, density matrices, entanglement measures.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple

from .operators import alice_operators, bob_operators


def expectation_value(
    state: np.ndarray,
    operator: np.ndarray,
) -> float:
    """Compute ⟨ψ|O|ψ⟩ for a pure state.

    Parameters
    ----------
    state : array
        State vector (will be conjugated on the left).
    operator : array
        Hermitian operator matrix.

    Returns
    -------
    Real-valued expectation value.
    """
    return float(np.real(state.conj() @ operator @ state))


def expectation_value_density(
    rho: np.ndarray,
    operator: np.ndarray,
) -> float:
    """Compute Tr(ρO) for a density matrix.

    Parameters
    ----------
    rho : array
        Density matrix.
    operator : array
        Hermitian operator.

    Returns
    -------
    Real-valued expectation value.
    """
    return float(np.real(np.trace(rho @ operator)))


def chsh_expectation_values(
    state: np.ndarray,
    A0: Optional[np.ndarray] = None,
    A1: Optional[np.ndarray] = None,
    B0: Optional[np.ndarray] = None,
    B1: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute all four CHSH expectation values for a quantum state.

    E(Ai, Bj) = ⟨ψ|Ai⊗Bj|ψ⟩  (operators are already in tensor product form)

    Parameters
    ----------
    state : array
        2-qubit state vector (length 4).
    A0, A1 : array, optional
        Alice's operators (4x4). Default: standard CHSH optimal.
    B0, B1 : array, optional
        Bob's operators (4x4). Default: standard CHSH optimal.

    Returns
    -------
    dict with keys "A_B", "A_B_prime", "A_prime_B", "A_prime_B_prime".
    """
    if A0 is None or A1 is None:
        A0, A1 = alice_operators()
    if B0 is None or B1 is None:
        B0, B1 = bob_operators()

    # The operators are already in 4x4 tensor product form,
    # but CHSH correlations need A_i * B_j products
    # Since A_i = a_i ⊗ I and B_j = I ⊗ b_j, their product is a_i ⊗ b_j
    return {
        "A_B": expectation_value(state, A0 @ B0),
        "A_B_prime": expectation_value(state, A0 @ B1),
        "A_prime_B": expectation_value(state, A1 @ B0),
        "A_prime_B_prime": expectation_value(state, A1 @ B1),
    }


def chsh_expectation_values_density(
    rho: np.ndarray,
    A0: Optional[np.ndarray] = None,
    A1: Optional[np.ndarray] = None,
    B0: Optional[np.ndarray] = None,
    B1: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute CHSH expectation values for a density matrix.

    Same as chsh_expectation_values but for mixed states.
    """
    if A0 is None or A1 is None:
        A0, A1 = alice_operators()
    if B0 is None or B1 is None:
        B0, B1 = bob_operators()

    return {
        "A_B": expectation_value_density(rho, A0 @ B0),
        "A_B_prime": expectation_value_density(rho, A0 @ B1),
        "A_prime_B": expectation_value_density(rho, A1 @ B0),
        "A_prime_B_prime": expectation_value_density(rho, A1 @ B1),
    }


def chsh_s_value(
    state: np.ndarray,
    A0: Optional[np.ndarray] = None,
    A1: Optional[np.ndarray] = None,
    B0: Optional[np.ndarray] = None,
    B1: Optional[np.ndarray] = None,
    is_density_matrix: bool = False,
) -> float:
    """Compute the CHSH S-value directly from a quantum state or density matrix.

    S = E(A0,B0) - E(A0,B1) + E(A1,B0) + E(A1,B1)

    Parameters
    ----------
    state : array
        State vector (length 4) or density matrix (4x4).
    A0, A1, B0, B1 : array, optional
        Measurement operators.
    is_density_matrix : bool
        If True, treat `state` as a density matrix.

    Returns
    -------
    S-value (float). Classical bound: |S| <= 2. Tsirelson bound: |S| <= 2√2.
    """
    if is_density_matrix:
        ev = chsh_expectation_values_density(state, A0, A1, B0, B1)
    else:
        ev = chsh_expectation_values(state, A0, A1, B0, B1)

    return ev["A_B"] - ev["A_B_prime"] + ev["A_prime_B"] + ev["A_prime_B_prime"]


def measure_state(
    state: np.ndarray,
    n_shots: int = 1024,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """Simulate measuring a state in the computational basis.

    Parameters
    ----------
    state : array
        State vector.
    n_shots : int
        Number of measurement shots.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict of outcome label -> count (e.g. {"00": 512, "11": 512}).
    """
    probs = np.abs(state) ** 2
    probs = probs / probs.sum()  # normalize

    n_qubits = int(np.log2(len(state)))
    rng = np.random.default_rng(seed)
    samples = rng.choice(len(state), size=n_shots, p=probs)

    counts = {}
    for s in samples:
        label = format(s, f"0{n_qubits}b")
        counts[label] = counts.get(label, 0) + 1
    return counts


def density_matrix(state: np.ndarray) -> np.ndarray:
    """Construct the density matrix |ψ⟩⟨ψ| from a pure state vector."""
    return np.outer(state, state.conj())


def reduced_density_matrix(
    rho: np.ndarray,
    trace_out: int = 1,
    dims: Tuple[int, int] = (2, 2),
) -> np.ndarray:
    """Compute the reduced density matrix by tracing out one subsystem.

    Parameters
    ----------
    rho : array
        Full density matrix of shape (d1*d2, d1*d2).
    trace_out : int
        Which subsystem to trace out (0 or 1).
    dims : tuple
        Dimensions of the two subsystems.

    Returns
    -------
    Reduced density matrix.
    """
    d1, d2 = dims
    rho_reshaped = rho.reshape(d1, d2, d1, d2)

    if trace_out == 1:
        # Trace over second subsystem
        return np.trace(rho_reshaped, axis1=1, axis2=3)
    else:
        # Trace over first subsystem
        return np.trace(rho_reshaped, axis1=0, axis2=2)


def von_neumann_entropy(rho: np.ndarray) -> float:
    """Compute the von Neumann entropy S = -Tr(ρ log₂ ρ).

    Parameters
    ----------
    rho : array
        Density matrix.

    Returns
    -------
    Entropy in bits. 0 for pure states, log₂(d) for maximally mixed.
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def entanglement_entropy(
    state: np.ndarray,
    dims: Tuple[int, int] = (2, 2),
) -> float:
    """Compute the entanglement entropy of a bipartite pure state.

    Traces out the second subsystem and computes von Neumann entropy
    of the reduced state.

    Parameters
    ----------
    state : array
        Pure state vector of the composite system.
    dims : tuple
        Dimensions of the two subsystems.

    Returns
    -------
    Entanglement entropy in bits. 0=separable, 1=maximally entangled (for qubits).
    """
    rho = density_matrix(state)
    rho_A = reduced_density_matrix(rho, trace_out=1, dims=dims)
    return von_neumann_entropy(rho_A)


def concurrence(state: np.ndarray) -> float:
    """Compute the concurrence of a 2-qubit pure state.

    C = 2|α₀₀ α₁₁ - α₀₁ α₁₀|

    Parameters
    ----------
    state : array
        4-component state vector [α₀₀, α₀₁, α₁₀, α₁₁].

    Returns
    -------
    Concurrence in [0, 1]. 0=separable, 1=maximally entangled.
    """
    if len(state) != 4:
        raise ValueError("Concurrence only defined for 2-qubit states")
    return float(2 * abs(state[0] * state[3] - state[1] * state[2]))


def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Compute the fidelity |⟨ψ₁|ψ₂⟩|² between two pure states."""
    return float(abs(np.dot(state1.conj(), state2)) ** 2)
