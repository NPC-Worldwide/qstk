"""Quantum operators for CHSH Bell test measurements.

Provides Pauli matrices, standard Alice/Bob CHSH measurement operators,
and utilities for constructing custom measurement bases.
"""

import numpy as np
from typing import Tuple, Optional


# Pauli matrices
def pauli_x() -> np.ndarray:
    """Pauli X (σ_x) matrix."""
    return np.array([[0, 1], [1, 0]], dtype=complex)


def pauli_y() -> np.ndarray:
    """Pauli Y (σ_y) matrix."""
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def pauli_z() -> np.ndarray:
    """Pauli Z (σ_z) matrix."""
    return np.array([[1, 0], [0, -1]], dtype=complex)


def pauli_i() -> np.ndarray:
    """2x2 Identity matrix."""
    return np.eye(2, dtype=complex)


def rotation_y(theta: float) -> np.ndarray:
    """Single-qubit Y-rotation matrix RY(θ)."""
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)],
    ], dtype=complex)


def rotation_z(phi: float) -> np.ndarray:
    """Single-qubit Z-rotation matrix RZ(φ)."""
    return np.array([
        [np.exp(-1j * phi / 2), 0],
        [0, np.exp(1j * phi / 2)],
    ], dtype=complex)


def hadamard() -> np.ndarray:
    """Hadamard gate."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def cnot() -> np.ndarray:
    """CNOT (controlled-X) gate for 2 qubits."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=complex)


def alice_operators(
    a0_angle: float = 0.0,
    a1_angle: float = np.pi / 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct Alice's two measurement operators for CHSH.

    Default angles give the standard optimal CHSH operators:
    - A₀ = σ_z ⊗ I  (angle 0)
    - A₁ = σ_x ⊗ I  (angle π/2)

    Parameters
    ----------
    a0_angle : float
        Measurement angle for A₀ in the Z-X plane.
        0 gives σ_z, π/2 gives σ_x.
    a1_angle : float
        Measurement angle for A₁.

    Returns
    -------
    (A0, A1) tuple of 4x4 Hermitian matrices.
    """
    I2 = pauli_i()

    def _angle_op(angle):
        return np.cos(angle) * pauli_z() + np.sin(angle) * pauli_x()

    A0 = np.kron(_angle_op(a0_angle), I2)
    A1 = np.kron(_angle_op(a1_angle), I2)
    return A0, A1


def bob_operators(
    b0_angle: float = np.pi / 4,
    b1_angle: float = 3 * np.pi / 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct Bob's two measurement operators for CHSH.

    Default angles give the standard optimal CHSH operators:
    - B₀ = I ⊗ (σ_z + σ_x)/√2  (angle π/4)
    - B₁ = I ⊗ (-σ_z + σ_x)/√2  (angle 3π/4)

    Parameters
    ----------
    b0_angle : float
        Measurement angle for B₀ in the Z-X plane.
    b1_angle : float
        Measurement angle for B₁.

    Returns
    -------
    (B0, B1) tuple of 4x4 Hermitian matrices.
    """
    I2 = pauli_i()

    def _angle_op(angle):
        return np.cos(angle) * pauli_z() + np.sin(angle) * pauli_x()

    B0 = np.kron(I2, _angle_op(b0_angle))
    B1 = np.kron(I2, _angle_op(b1_angle))
    return B0, B1


def chsh_operator(
    A0: Optional[np.ndarray] = None,
    A1: Optional[np.ndarray] = None,
    B0: Optional[np.ndarray] = None,
    B1: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Construct the CHSH Bell operator.

    B_CHSH = A₀⊗B₀ - A₀⊗B₁ + A₁⊗B₀ + A₁⊗B₁

    If operators are None, uses the standard optimal CHSH operators.

    Returns
    -------
    4x4 Hermitian matrix (the CHSH operator).
    """
    if A0 is None or A1 is None:
        A0, A1 = alice_operators()
    if B0 is None or B1 is None:
        B0, B1 = bob_operators()

    return A0 @ B0 - A0 @ B1 + A1 @ B0 + A1 @ B1


def measurement_operator(
    angle: float,
    qubit: int = 0,
    n_qubits: int = 2,
) -> np.ndarray:
    """Construct a single-qubit measurement operator in the Z-X plane.

    M(θ) = cos(θ)σ_z + sin(θ)σ_x

    Embedded in the full n-qubit Hilbert space on the specified qubit.

    Parameters
    ----------
    angle : float
        Measurement angle. 0=Z basis, π/2=X basis.
    qubit : int
        Which qubit (0-indexed) this operator acts on.
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    2^n × 2^n Hermitian matrix.
    """
    single_op = np.cos(angle) * pauli_z() + np.sin(angle) * pauli_x()
    I2 = pauli_i()

    ops = [I2] * n_qubits
    ops[qubit] = single_op

    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the commutator [A, B] = AB - BA."""
    return A @ B - B @ A


def commutator_norm(A: np.ndarray, B: np.ndarray) -> float:
    """Compute the Frobenius norm of the commutator [A, B]."""
    return float(np.linalg.norm(commutator(A, B), "fro"))


def random_hermitian(dim: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a random Hermitian matrix (e.g. for testing non-commutativity).

    Parameters
    ----------
    dim : int
        Matrix dimension.
    seed : int, optional
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return (M + M.conj().T) / 2
