"""Qiskit-based CHSH simulation for qstk.qc.

Uses Qiskit AerSimulator to execute CHSH measurement circuits. This is
not a numpy statevector simulator; the quantum circuit is built and run
through Qiskit, returning sampled bitstring counts.
"""

import time
from typing import Dict, Any, Optional

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

from .hardware import (
    HardwareResult,
    expectation_from_counts,
    s_value_from_counts,
    _build_qiskit_chsh_circuits,
)


def run_qiskit_simulator(
    state_type: str = "phi_plus",
    n_shots: int = 1024,
    a0_angle: float = 0.0,
    a1_angle: float = np.pi / 2,
    b0_angle: float = np.pi / 4,
    b1_angle: float = 3 * np.pi / 4,
    seed: Optional[int] = None,
) -> HardwareResult:
    """Run CHSH experiment through Qiskit AerSimulator.

    Builds 4 CHSH circuits using the same construction as real IBM hardware,
    executes them on AerSimulator, and computes S from sampled counts.

    Parameters
    ----------
    state_type : str
        Bell state to prepare.
    n_shots : int
        Measurement shots per circuit.
    a0_angle, a1_angle : float
        Alice measurement angles.
    b0_angle, b1_angle : float
        Bob measurement angles.
    seed : int, optional
        Seed for AerSimulator.

    Returns
    -------
    HardwareResult with Qiskit-simulated counts and S-value.
    """
    circuits = _build_qiskit_chsh_circuits(
        state_type, a0_angle, a1_angle, b0_angle, b1_angle
    )
    labels = list(circuits.keys())

    simulator = AerSimulator(seed_simulator=seed)

    t0 = time.time()
    all_counts: Dict[str, Dict[str, int]] = {}
    for label, qc in circuits.items():
        transpiled = transpile(qc, simulator)
        job = simulator.run(transpiled, shots=n_shots)
        result = job.result()
        counts = result.get_counts(transpiled)
        all_counts[label] = {k: int(v) for k, v in counts.items()}

    elapsed = time.time() - t0

    s_value, ev = s_value_from_counts(all_counts)

    return HardwareResult(
        backend="qiskit_aer",
        backend_type="simulator",
        counts=all_counts,
        n_shots=n_shots,
        s_value=s_value,
        expectation_values=ev,
        execution_time_s=elapsed,
        cost_estimate_usd=0.0,
        metadata={
            "state_type": state_type,
            "seed": seed,
            "angles": {
                "a0": a0_angle,
                "a1": a1_angle,
                "b0": b0_angle,
                "b1": b1_angle,
            },
        },
        timestamp=time.time(),
    )


def draw_chsh_circuit(
    state_type: str,
    a0_angle: float,
    a1_angle: float,
    b0_angle: float,
    b1_angle: float,
    setting: str,
    output_path: str,
) -> str:
    """Draw one CHSH Qiskit circuit with gate annotations and save to file.

    Parameters
    ----------
    state_type : str
        Bell state.
    a0_angle, a1_angle, b0_angle, b1_angle : float
        Measurement angles.
    setting : str
        One of "A_B", "A_B_prime", "A_prime_B", "A_prime_B_prime".
    output_path : str
        File path for the matplotlib figure.

    Returns
    -------
    output_path
    """
    circuits = _build_qiskit_chsh_circuits(
        state_type, a0_angle, a1_angle, b0_angle, b1_angle
    )
    qc = circuits[setting]

    angle_labels = {
        "A_B": (a0_angle, b0_angle),
        "A_B_prime": (a0_angle, b1_angle),
        "A_prime_B": (a1_angle, b0_angle),
        "A_prime_B_prime": (a1_angle, b1_angle),
    }
    alice_a, bob_b = angle_labels[setting]

    title = (
        f"{state_type} | {setting} | "
        f"Alice θ={alice_a:.3f}, Bob θ={bob_b:.3f}"
    )

    text_diagram = qc.draw(output="text", fold=60).single_string()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    ax.text(0.0, 0.95, title, fontsize=13, fontweight="bold", transform=ax.transAxes)
    ax.text(0.0, 0.55, text_diagram, fontsize=10, family="monospace",
            verticalalignment="center", transform=ax.transAxes)

    gate_note = (
        "Gate meaning:\n"
        "  H  = Hadamard\n"
        "  ●─X = CNOT (control ●, target X)\n"
        "  Z  = phase flip for Φ− / Ψ−\n"
        "  X  = bit flip for Ψ+ / Ψ−\n"
        "  Ry(−θ) = rotation to measurement basis\n"
        "  M  = measurement in computational (Z) basis"
    )
    ax.text(0.72, 0.55, gate_note, fontsize=9, verticalalignment="center",
            transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
