"""Quantum circuit construction for Bell/CHSH experiments.

Provides backend-agnostic circuit descriptions that can be compiled
to Qiskit, Cirq, or executed directly via numpy statevector simulation.

Each circuit function returns a dict describing the circuit, plus a
numpy-simulated result. For real hardware execution, use the
backend-specific helpers (to_qiskit, to_cirq).
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from .states import bell_state, _ket0
from .operators import (
    hadamard, cnot, pauli_x, pauli_z, pauli_i,
    rotation_y, rotation_z,
    alice_operators, bob_operators,
)
from .measure import (
    expectation_value,
    chsh_expectation_values,
    chsh_s_value,
    measure_state,
)


def bell_circuit(
    which: str = "phi_plus",
    n_shots: int = 1024,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Construct and simulate a Bell state preparation circuit.

    Circuit:
        |0⟩ ─ H ─ ● ─    (+ optional phase gates for different Bell states)
        |0⟩ ───── X ─

    Parameters
    ----------
    which : str
        Bell state: "phi_plus", "phi_minus", "psi_plus", "psi_minus".
    n_shots : int
        Measurement shots for sampling.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict with keys:
        - "state": the prepared state vector
        - "counts": measurement counts
        - "circuit_ops": list of gate operations (for backend compilation)
        - "which": which Bell state
    """
    # Build via gate operations
    initial = np.kron(_ket0, _ket0)
    ops = []

    # Hadamard on qubit 0
    H_full = np.kron(hadamard(), pauli_i())
    state = H_full @ initial
    ops.append(("H", 0))

    # CNOT
    state = cnot() @ state
    ops.append(("CNOT", 0, 1))

    # Phase gates for different Bell states
    if which in ("phi_minus", "Φ-"):
        Z_full = np.kron(pauli_z(), pauli_i())
        state = Z_full @ state
        ops.append(("Z", 0))
    elif which in ("psi_plus", "Ψ+"):
        X_full = np.kron(pauli_i(), pauli_x())
        state = X_full @ state
        ops.append(("X", 1))
    elif which in ("psi_minus", "Ψ-"):
        X_full = np.kron(pauli_i(), pauli_x())
        Z_full = np.kron(pauli_z(), pauli_i())
        state = X_full @ Z_full @ state
        ops.append(("Z", 0))
        ops.append(("X", 1))

    counts = measure_state(state, n_shots, seed)

    return {
        "state": state,
        "counts": counts,
        "circuit_ops": ops,
        "which": which,
    }


def chsh_circuit(
    state_type: str = "phi_plus",
    a0_angle: float = 0.0,
    a1_angle: float = np.pi / 2,
    b0_angle: float = np.pi / 4,
    b1_angle: float = 3 * np.pi / 4,
    n_shots: int = 1024,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Construct and simulate a full CHSH experiment circuit.

    Prepares a Bell state and computes all four CHSH correlations
    with configurable measurement angles.

    Parameters
    ----------
    state_type : str
        Which Bell state to prepare.
    a0_angle, a1_angle : float
        Alice's measurement angles (radians). Default: optimal CHSH.
    b0_angle, b1_angle : float
        Bob's measurement angles (radians). Default: optimal CHSH.
    n_shots : int
        Measurement shots per setting.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict with keys:
        - "state": prepared state vector
        - "s_value": CHSH S-value
        - "expectation_values": dict of E(Ai,Bj)
        - "violation": whether |S| > 2
        - "tsirelson_fraction": |S| / 2√2
        - "measurement_counts": counts per setting
        - "operators": (A0, A1, B0, B1)
    """
    prep = bell_circuit(state_type, n_shots=0)
    state = prep["state"]

    A0, A1 = alice_operators(a0_angle, a1_angle)
    B0, B1 = bob_operators(b0_angle, b1_angle)

    ev = chsh_expectation_values(state, A0, A1, B0, B1)
    s = ev["A_B"] - ev["A_B_prime"] + ev["A_prime_B"] + ev["A_prime_B_prime"]

    # Sample measurements for each setting
    rng = np.random.default_rng(seed)
    measurement_counts = {}
    for label, op in [("A_B", A0 @ B0), ("A_B_prime", A0 @ B1),
                       ("A_prime_B", A1 @ B0), ("A_prime_B_prime", A1 @ B1)]:
        # Project state into eigenbasis of the operator and sample
        eigenvalues, eigenvectors = np.linalg.eigh(op)
        probs = np.abs(eigenvectors.conj().T @ state) ** 2
        probs = probs / probs.sum()
        samples = rng.choice(len(eigenvalues), size=n_shots, p=probs)
        outcomes = eigenvalues[samples]
        measurement_counts[label] = {
            "+1": int(np.sum(outcomes > 0)),
            "-1": int(np.sum(outcomes <= 0)),
            "mean": float(np.mean(outcomes)),
        }

    tsirelson = 2 * np.sqrt(2)

    return {
        "state": state,
        "s_value": s,
        "expectation_values": ev,
        "violation": abs(s) > 2.0,
        "tsirelson_fraction": abs(s) / tsirelson,
        "measurement_counts": measurement_counts,
        "operators": (A0, A1, B0, B1),
    }


def semantic_circuit(
    amplitudes: List[complex],
    a0_angle: float = 0.0,
    a1_angle: float = np.pi / 2,
    b0_angle: float = np.pi / 4,
    b1_angle: float = 3 * np.pi / 4,
    n_shots: int = 1024,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run CHSH experiment on a custom semantic state.

    Instead of a Bell state, uses an arbitrary 2-qubit state defined by
    amplitudes (e.g. from word sense probabilities).

    Parameters
    ----------
    amplitudes : list of complex
        4-component amplitudes [α₀₀, α₀₁, α₁₀, α₁₁].
    a0_angle, a1_angle, b0_angle, b1_angle : float
        Measurement angles.
    n_shots : int
        Measurement shots.
    seed : int, optional
        Random seed.

    Returns
    -------
    Same structure as chsh_circuit output.
    """
    state = np.array(amplitudes, dtype=complex)
    state = state / np.linalg.norm(state)

    from .measure import entanglement_entropy, concurrence
    ent_entropy = entanglement_entropy(state)
    conc = concurrence(state)

    A0, A1 = alice_operators(a0_angle, a1_angle)
    B0, B1 = bob_operators(b0_angle, b1_angle)

    ev = chsh_expectation_values(state, A0, A1, B0, B1)
    s = ev["A_B"] - ev["A_B_prime"] + ev["A_prime_B"] + ev["A_prime_B_prime"]

    counts = measure_state(state, n_shots, seed)
    tsirelson = 2 * np.sqrt(2)

    return {
        "state": state,
        "s_value": s,
        "expectation_values": ev,
        "violation": abs(s) > 2.0,
        "tsirelson_fraction": abs(s) / tsirelson,
        "entanglement_entropy": ent_entropy,
        "concurrence": conc,
        "measurement_counts": counts,
        "operators": (A0, A1, B0, B1),
    }


def to_qiskit(
    circuit_ops: List[tuple],
    n_qubits: int = 2,
    measure: bool = True,
):
    """Convert circuit_ops to a Qiskit QuantumCircuit.

    Uses the modern Qiskit API (circuit library gates + append pattern).

    Parameters
    ----------
    circuit_ops : list of tuples
        Gate operations from bell_circuit or similar.
        Each tuple is (gate_name, qubit_idx, ...) or (gate_name, ctrl, tgt)
        for two-qubit gates.
    n_qubits : int
        Number of qubits.
    measure : bool
        Whether to append measurement gates.

    Returns
    -------
    qiskit.QuantumCircuit (requires qiskit >= 1.0 to be installed).

    Example
    -------
    >>> prep = bell_circuit("phi_plus")
    >>> qc = to_qiskit(prep["circuit_ops"])
    >>> # Run on hardware via Qiskit Runtime:
    >>> # from qiskit.primitives import StatevectorSampler
    >>> # result = StatevectorSampler().run([qc]).result()
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import HGate, XGate, YGate, ZGate, CXGate, RYGate, RZGate

    qc = QuantumCircuit(n_qubits)

    gate_map = {
        "H": lambda op: qc.append(HGate(), [op[1]]),
        "X": lambda op: qc.append(XGate(), [op[1]]),
        "Y": lambda op: qc.append(YGate(), [op[1]]),
        "Z": lambda op: qc.append(ZGate(), [op[1]]),
        "CNOT": lambda op: qc.append(CXGate(), [op[1], op[2]]),
        "RY": lambda op: qc.append(RYGate(op[2]), [op[1]]),
        "RZ": lambda op: qc.append(RZGate(op[2]), [op[1]]),
    }

    for op in circuit_ops:
        handler = gate_map.get(op[0])
        if handler:
            handler(op)
        else:
            raise ValueError(f"Unknown gate: {op[0]}")

    if measure:
        qc.measure_all()

    return qc


def to_cirq(circuit_ops: List[tuple], n_qubits: int = 2):
    """Convert circuit_ops to a Cirq Circuit.

    Parameters
    ----------
    circuit_ops : list of tuples
        Gate operations from bell_circuit or similar.
    n_qubits : int
        Number of qubits.

    Returns
    -------
    cirq.Circuit (requires cirq to be installed).
    """
    import cirq

    qubits = cirq.LineQubit.range(n_qubits)
    moments = []
    for op in circuit_ops:
        gate = op[0]
        if gate == "H":
            moments.append(cirq.H(qubits[op[1]]))
        elif gate == "X":
            moments.append(cirq.X(qubits[op[1]]))
        elif gate == "Y":
            moments.append(cirq.Y(qubits[op[1]]))
        elif gate == "Z":
            moments.append(cirq.Z(qubits[op[1]]))
        elif gate == "CNOT":
            moments.append(cirq.CNOT(qubits[op[1]], qubits[op[2]]))
        elif gate == "RY":
            moments.append(cirq.ry(op[2])(qubits[op[1]]))
        elif gate == "RZ":
            moments.append(cirq.rz(op[2])(qubits[op[1]]))

    circuit = cirq.Circuit(moments)
    circuit.append(cirq.measure(*qubits, key="result"))
    return circuit
