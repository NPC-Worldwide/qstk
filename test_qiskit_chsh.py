#!/usr/bin/env python3
"""Qiskit-based CHSH tests and visualizations for qstk.qc.

Uses Qiskit AerSimulator (not numpy) to execute CHSH circuits and produces:
- Annotated circuit diagrams for all 4 measurement settings.
- Comparison of exact S-values vs Qiskit-simulated S-values across Bell states.
- Shot-noise convergence plot.
- Werner noise curve from Qiskit simulation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from qstk.qc.states import bell_state, werner_state
from qstk.qc.operators import alice_operators, bob_operators
from qstk.qc.measure import chsh_s_value
from qstk.qc.qiskit_runner import run_qiskit_simulator, draw_chsh_circuit


OPTIMAL_ANGLES = {
    "phi_plus":   (0.0,       np.pi / 2,  np.pi / 4, 3 * np.pi / 4),
    "phi_minus":  (0.0,      -np.pi / 2,  np.pi / 4, 3 * np.pi / 4),
    "psi_plus":   (0.0,       np.pi / 2,  3 * np.pi / 4, np.pi / 4),
    "psi_minus":  (0.0,       np.pi / 2,  np.pi / 4, 3 * np.pi / 4),
}

SETTINGS = ["A_B", "A_B_prime", "A_prime_B", "A_prime_B_prime"]


def draw_all_circuits(out_dir: str):
    """Draw annotated CHSH circuits for phi_plus and save each setting."""
    os.makedirs(out_dir, exist_ok=True)
    a0, a1, b0, b1 = OPTIMAL_ANGLES["phi_plus"]
    paths = []
    for setting in SETTINGS:
        path = os.path.join(out_dir, f"chsh_circuit_phi_plus_{setting}.png")
        draw_chsh_circuit("phi_plus", a0, a1, b0, b1, setting, path)
        paths.append(path)
        print(f"  saved circuit: {path}")
    return paths


def bell_state_comparison_table(n_shots: int = 8192, seed: int = 42):
    """Compare exact S-values to Qiskit Aer simulation for all Bell states."""
    print("\n" + "=" * 70)
    print("QISKIT AER CHSH RESULTS (n_shots={} seed={})".format(n_shots, seed))
    print("=" * 70)
    print(f"{'State':<15} {'Exact |S|':>12} {'Qiskit |S|':>12} {'Delta':>10} {'Violation':>10}")
    print("-" * 70)

    results = {}
    for name in ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]:
        a0, a1, b0, b1 = OPTIMAL_ANGLES[name]
        A0, A1 = alice_operators(a0, a1)
        B0, B1 = bob_operators(b0, b1)
        exact = abs(chsh_s_value(bell_state(name), A0, A1, B0, B1))
        r = run_qiskit_simulator(name, n_shots=n_shots, a0_angle=a0, a1_angle=a1,
                                 b0_angle=b0, b1_angle=b1, seed=seed)
        sim = abs(r.s_value)
        delta = sim - exact
        violation = "YES" if sim > 2.0 else "no"
        print(f"{name:<15} {exact:>12.6f} {sim:>12.6f} {delta:>+10.6f} {violation:>10}")
        results[name] = {"exact": exact, "sim": sim, "counts": r.counts}
    return results


def plot_bell_state_comparison(results, out_path: str):
    """Bar chart: exact vs Qiskit S-values for all Bell states."""
    names = list(results.keys())
    exact = [results[n]["exact"] for n in names]
    sim = [results[n]["sim"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, exact, width, label="Exact", color="steelblue")
    ax.bar(x + width / 2, sim, width, label="Qiskit Aer", color="coral")
    ax.axhline(2 * np.sqrt(2), color="green", linestyle="--", label="Tsirelson 2√2")
    ax.axhline(2.0, color="red", linestyle="--", label="Classical bound |S|=2")
    ax.set_ylabel("|S|")
    ax.set_title("Exact vs Qiskit Aer CHSH S-values")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  saved plot: {out_path}")


def plot_shot_noise_convergence(state_type: str, out_path: str, seed: int = 42):
    """Show how Qiskit simulated |S| converges to exact as shots increase."""
    a0, a1, b0, b1 = OPTIMAL_ANGLES[state_type]
    A0, A1 = alice_operators(a0, a1)
    B0, B1 = bob_operators(b0, b1)
    exact = abs(chsh_s_value(bell_state(state_type), A0, A1, B0, B1))

    shot_counts = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    means = []
    stds = []

    for n in shot_counts:
        trials = []
        for trial_seed in range(10):
            r = run_qiskit_simulator(state_type, n_shots=n, a0_angle=a0, a1_angle=a1,
                                     b0_angle=b0, b1_angle=b1, seed=seed + trial_seed)
            trials.append(abs(r.s_value))
        means.append(np.mean(trials))
        stds.append(np.std(trials))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(shot_counts, means, yerr=stds, fmt="o-", capsize=5, color="coral",
                label=f"Qiskit Aer |S| (mean ± σ)")
    ax.axhline(exact, color="green", linestyle="--", label=f"Exact |S| = {exact:.4f}")
    ax.axhline(2.0, color="red", linestyle="--", label="Classical bound |S|=2")
    ax.set_xlabel("Shots per circuit")
    ax.set_ylabel("|S|")
    ax.set_title(f"Shot-Noise Convergence: {state_type}")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  saved plot: {out_path}")


def plot_werner_noise_qiskit(out_path: str, seed: int = 42):
    """Werner noise curve via Qiskit simulation of phi_plus."""
    a0, a1, b0, b1 = OPTIMAL_ANGLES["phi_plus"]
    p_values = np.linspace(0, 1, 21)
    exact_s = []
    qiskit_s = []

    for p in p_values:
        rho = werner_state(bell_state("phi_plus"), p)
        exact_s.append(abs(chsh_s_value(rho, is_density_matrix=True)))
        r = run_qiskit_simulator("phi_plus", n_shots=4096, a0_angle=a0, a1_angle=a1,
                                 b0_angle=b0, b1_angle=b1, seed=seed)
        qiskit_s.append(abs(r.s_value))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(p_values, exact_s, "o-", color="steelblue", label="Exact")
    ax.plot(p_values, qiskit_s, "s-", color="coral", label="Qiskit Aer (4096 shots)")
    ax.axhline(2 * np.sqrt(2), color="green", linestyle="--", label="Tsirelson bound")
    ax.axhline(2.0, color="red", linestyle="--", label="Classical bound |S|=2")
    ax.axvline(1 / np.sqrt(2), color="purple", linestyle=":",
               label=f"Critical p = 1/√2 ≈ {1 / np.sqrt(2):.3f}")
    ax.set_xlabel("Werner parameter p")
    ax.set_ylabel("|S|")
    ax.set_title("CHSH Violation vs Werner Noise (Qiskit Aer)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  saved plot: {out_path}")


if __name__ == "__main__":
    out_dir = "/home/caug/npcww/qstk/qstk/qiskit_chsh_outputs"
    os.makedirs(out_dir, exist_ok=True)

    print("\nDrawing annotated CHSH circuits...")
    draw_all_circuits(out_dir)

    results = bell_state_comparison_table(n_shots=8192, seed=42)

    plot_bell_state_comparison(results, os.path.join(out_dir, "bell_state_comparison.png"))
    plot_shot_noise_convergence("phi_plus", os.path.join(out_dir, "shot_noise_convergence.png"))
    plot_werner_noise_qiskit(os.path.join(out_dir, "werner_noise_qiskit.png"))

    print("\nAll Qiskit CHSH outputs saved to:", out_dir)
