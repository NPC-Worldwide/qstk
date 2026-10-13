"""Generate all README images headlessly. Run from repo root:
    python docs/generate_images.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/images"

# ═════════════════════════════════════════════════════════════════════
# QC: Bell states
# ═════════════════════════════════════════════════════════════════════

from qstk.qc import (
    bell_state, werner_state, parameterized_entangled_state,
    alice_operators, bob_operators,
    chsh_s_value, chsh_expectation_values, measure_state,
    entanglement_entropy, concurrence, fidelity,
    chsh_circuit, semantic_circuit,
    chsh_expectation_values_density,
)


def qc_bell_states():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("The Four Bell States — Measurement Probabilities (1000 shots)", fontsize=14, fontweight="bold")
    names = ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]
    symbols = ["|Φ+⟩", "|Φ-⟩", "|Ψ+⟩", "|Ψ-⟩"]
    colors = ["#2196F3", "#f44336", "#4CAF50", "#FF9800"]

    for ax, name, sym, col in zip(axes, names, symbols, colors):
        psi = bell_state(name)
        counts = measure_state(psi, n_shots=1000, seed=42)
        labels = sorted(counts.keys())
        vals = [counts.get(l, 0) for l in labels]
        ax.bar(labels, vals, color=col, edgecolor="k", linewidth=0.5)
        c = concurrence(psi)
        e = entanglement_entropy(psi)
        ax.set_title(f"{sym}\nC={c:.2f}, S_ent={e:.2f} bits", fontsize=11)
        ax.set_ylabel("Counts")
        ax.set_ylim(0, 1100)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(f"{OUT}/bell_states.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  bell_states.png")


def qc_chsh_angle_sweep():
    fig, ax = plt.subplots(figsize=(10, 5))
    psi = bell_state("phi_plus")
    A0, A1 = alice_operators(0, np.pi / 2)
    angles = np.linspace(0, 360, 360)
    s_vals = []
    for b1_deg in angles:
        B0, B1 = bob_operators(np.pi / 4, np.radians(b1_deg))
        s_vals.append(chsh_s_value(psi, A0, A1, B0, B1))

    ax.plot(angles, s_vals, "b-", linewidth=2)
    ax.axhline(y=2, color="red", linestyle="--", linewidth=2, label="Classical bound |S|=2")
    ax.axhline(y=-2, color="red", linestyle="--", linewidth=2)
    ax.axhline(y=2 * np.sqrt(2), color="green", linestyle=":", linewidth=2, label="Tsirelson bound 2√2")
    ax.axhline(y=-2 * np.sqrt(2), color="green", linestyle=":", linewidth=2)
    ax.fill_between(angles, 2, 2 * np.sqrt(2), alpha=0.08, color="green", label="Quantum advantage zone")
    ax.fill_between(angles, -2 * np.sqrt(2), -2, alpha=0.08, color="green")

    # Mark optimal
    opt_idx = np.argmax(s_vals)
    ax.plot(angles[opt_idx], s_vals[opt_idx], "r*", markersize=15,
            label=f"Max S={s_vals[opt_idx]:.3f} at b₁={angles[opt_idx]:.0f}°")

    ax.set_xlabel("Bob's b₁ angle (degrees)", fontsize=12)
    ax.set_ylabel("S-value", fontsize=12)
    ax.set_title("CHSH S-value vs Bob's Measurement Angle (|Φ+⟩, a₀=0°, a₁=90°, b₀=45°)",
                 fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)
    plt.tight_layout()
    fig.savefig(f"{OUT}/chsh_angle_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  chsh_angle_sweep.png")


def qc_werner_sweep():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Werner States: Noise Degrades Entanglement", fontsize=14, fontweight="bold")

    ps = np.linspace(0, 1, 200)
    s_vals = []
    for p in ps:
        rho = werner_state(bell_state("phi_plus"), p)
        s_vals.append(chsh_s_value(rho, is_density_matrix=True))
    s_vals = np.array(s_vals)

    # S vs p
    ax = axes[0]
    ax.plot(ps, s_vals, "b-", linewidth=2.5)
    ax.axhline(y=2, color="red", linestyle="--", linewidth=2, label="|S|=2")
    ax.axhline(y=2 * np.sqrt(2), color="green", linestyle=":", linewidth=2, label="2√2")
    ax.axvline(x=1 / np.sqrt(2), color="orange", linestyle="-.", linewidth=2,
               label=f"p=1/√2≈{1/np.sqrt(2):.3f}")
    ax.fill_between(ps, 2, s_vals, where=s_vals > 2, alpha=0.15, color="green")
    ax.set_xlabel("Werner parameter p", fontsize=11)
    ax.set_ylabel("S-value", fontsize=11)
    ax.set_title("S-value vs Noise")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Density matrices at p=1 and p=0.5
    for idx, (p_val, title) in enumerate([(1.0, "Pure Bell |Φ+⟩ (p=1)"), (0.5, "Werner (p=0.5)")]):
        ax = axes[idx + 1]
        rho = werner_state(bell_state("phi_plus"), p_val)
        im = ax.imshow(np.real(rho), cmap="RdBu_r", vmin=-0.5, vmax=0.5, extent=[0, 4, 4, 0])
        ax.set_xticks([0.5, 1.5, 2.5, 3.5])
        ax.set_xticklabels(["00", "01", "10", "11"])
        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_yticklabels(["00", "01", "10", "11"])
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(f"{OUT}/werner_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  werner_sweep.png")


def qc_s_heatmap():
    state = bell_state("phi_plus")
    A0, A1 = alice_operators(0, np.pi / 2)
    n = 120
    b0v = np.linspace(0, 2 * np.pi, n)
    b1v = np.linspace(0, 2 * np.pi, n)
    B0g, B1g = np.meshgrid(b0v, b1v)
    S = np.zeros_like(B0g)
    for i in range(n):
        for j in range(n):
            B0, B1 = bob_operators(B0g[i, j], B1g[i, j])
            S[i, j] = chsh_s_value(state, A0, A1, B0, B1)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.pcolormesh(np.degrees(B0g), np.degrees(B1g), S,
                        cmap="RdBu_r", vmin=-3, vmax=3, shading="auto")
    plt.colorbar(im, ax=ax, label="S-value")
    ax.contour(np.degrees(B0g), np.degrees(B1g), np.abs(S),
               levels=[2.0], colors=["red"], linewidths=[2], linestyles=["--"])
    ax.contour(np.degrees(B0g), np.degrees(B1g), np.abs(S),
               levels=[2 * np.sqrt(2) - 0.01], colors=["lime"], linewidths=[2], linestyles=[":"])
    ax.plot(45, 135, "w*", markersize=20, markeredgecolor="k", markeredgewidth=1.5,
            label="Optimal (45°, 135°)")
    ax.set_xlabel("Bob b₀ angle (degrees)", fontsize=12)
    ax.set_ylabel("Bob b₁ angle (degrees)", fontsize=12)
    ax.set_title("CHSH S-value Landscape for |Φ+⟩\nRed dashes: |S|=2 | Green: Tsirelson",
                 fontsize=12)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(f"{OUT}/s_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  s_heatmap.png")


def qc_entanglement_landscape():
    rng = np.random.default_rng(42)
    n = 2000
    concs, s_vals, entropies = [], [], []
    for _ in range(n):
        theta = rng.uniform(0, np.pi / 2)
        phi = rng.uniform(0, 2 * np.pi)
        psi = parameterized_entangled_state(theta, phi)
        concs.append(concurrence(psi))
        s_vals.append(abs(chsh_s_value(psi)))
        entropies.append(entanglement_entropy(psi))

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(concs, s_vals, c=entropies, cmap="plasma", s=8, alpha=0.6, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Entanglement Entropy (bits)")
    ax.axhline(y=2, color="red", linestyle="--", linewidth=2, label="Classical bound |S|=2")
    ax.axhline(y=2 * np.sqrt(2), color="green", linestyle=":", linewidth=2, label="Tsirelson bound")
    ax.set_xlabel("Concurrence", fontsize=12)
    ax.set_ylabel("|S| value", fontsize=12)
    ax.set_title("Entanglement Measures: 2000 Random 2-Qubit States", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{OUT}/entanglement_landscape.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  entanglement_landscape.png")


def qc_semantic_states():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Semantic CHSH: Word-Sense Entanglement", fontsize=14, fontweight="bold")

    cases = [
        ("Correlated", [0.7, 0.05, 0.05, 0.7]),
        ("Uniform", [0.5, 0.5, 0.5, 0.5]),
        ("Anti-correlated", [0.05, 0.7, 0.7, 0.05]),
    ]
    colors = ["#4CAF50", "#2196F3", "#f44336"]
    basis = ["|fin,bat⟩", "|fin,ani⟩", "|riv,bat⟩", "|riv,ani⟩"]

    for ax, (label, amps), col in zip(axes, cases, colors):
        result = semantic_circuit(amps, seed=42)
        state = result["state"]
        probs = np.abs(state) ** 2

        ax.bar(basis, probs, color=col, edgecolor="k", linewidth=0.5, alpha=0.8)
        s = result["s_value"]
        c = result["concurrence"]
        viol = "VIOLATION" if result["violation"] else "classical"
        ax.set_title(f"{label}\nS={s:+.3f} ({viol}), C={c:.3f}", fontsize=11)
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 0.6)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    fig.savefig(f"{OUT}/semantic_states.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  semantic_states.png")


def qc_parameterized_entanglement():
    thetas = np.linspace(0, 90, 200)
    concs, s_vals, ents = [], [], []
    for td in thetas:
        psi = parameterized_entangled_state(np.radians(td))
        concs.append(concurrence(psi))
        s_vals.append(chsh_s_value(psi))
        ents.append(entanglement_entropy(psi))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    l1, = ax1.plot(thetas, concs, "b-", linewidth=2.5, label="Concurrence")
    l3, = ax1.plot(thetas, ents, "g--", linewidth=2, label="Ent. Entropy (bits)")
    ax1.set_xlabel("θ (degrees)", fontsize=12)
    ax1.set_ylabel("Concurrence / Entropy", fontsize=12, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    l2, = ax2.plot(thetas, s_vals, "r-", linewidth=2.5, label="S-value")
    ax2.axhline(y=2, color="red", linestyle="--", alpha=0.5)
    ax2.set_ylabel("S-value", fontsize=12, color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.legend([l1, l3, l2], ["Concurrence", "Ent. Entropy", "S-value"],
               loc="lower center", fontsize=10)
    ax1.set_title("Parameterized Entanglement: cos(θ)|00⟩ + sin(θ)|11⟩", fontsize=13)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{OUT}/parameterized_entanglement.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  parameterized_entanglement.png")


# ═════════════════════════════════════════════════════════════════════
# CHSH Experiments
# ═════════════════════════════════════════════════════════════════════

from qstk.chsh import (
    compute_chsh_products_binary,
    calculate_expectation_values_direct,
    calculate_expectation_values_density_matrix,
    calculate_s_value,
    check_violation,
)


def _simulate_classical(n, seed):
    rng = np.random.default_rng(seed)
    products = []
    for _ in range(n):
        lam = rng.choice([-1, 1])
        products.append(compute_chsh_products_binary({
            "A": lam, "A_prime": lam, "B": lam, "B_prime": rng.choice([-1, 1])
        }))
    return products


def _simulate_quantum(n, noise, seed):
    rng = np.random.default_rng(seed)
    products = []
    p_corr = (1 + 1 / np.sqrt(2)) / 2
    for _ in range(n):
        if rng.random() < noise:
            o = {k: rng.choice([-1, 1]) for k in ["A", "A_prime", "B", "B_prime"]}
        else:
            a = rng.choice([-1, 1])
            ap = rng.choice([-1, 1])
            o = {
                "A": a, "A_prime": ap,
                "B": a if rng.random() < p_corr else -a,
                "B_prime": -a if rng.random() < p_corr else a,
            }
        products.append(compute_chsh_products_binary(o))
    return products


def chsh_classical_vs_quantum():
    rng = np.random.default_rng(42)
    classical_s, quantum_s, noisy_s = [], [], []
    for _ in range(300):
        seed = rng.integers(100000)
        ev = calculate_expectation_values_direct(_simulate_classical(100, seed))
        classical_s.append(calculate_s_value(ev))
        ev = calculate_expectation_values_direct(_simulate_quantum(100, 0.05, seed))
        quantum_s.append(calculate_s_value(ev))
        ev = calculate_expectation_values_direct(_simulate_quantum(100, 0.5, seed))
        noisy_s.append(calculate_s_value(ev))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Classical vs Quantum Bell Tests (Synthetic)", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.hist(classical_s, bins=30, alpha=0.6, color="#2196F3", label="Classical", density=True)
    ax.hist(quantum_s, bins=30, alpha=0.6, color="#f44336", label="Quantum (5% noise)", density=True)
    ax.hist(noisy_s, bins=30, alpha=0.6, color="#FF9800", label="Quantum (50% noise)", density=True)
    ax.axvline(x=2, color="k", linestyle="--", linewidth=2, label="|S|=2")
    ax.axvline(x=-2, color="k", linestyle="--", linewidth=2)
    ax.set_xlabel("S-value")
    ax.set_ylabel("Density")
    ax.set_title("S-value Distributions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cats = ["Classical", "Quantum\n(5% noise)", "Quantum\n(50% noise)"]
    rates = [sum(abs(s) > 2 for s in d) / len(d) for d in [classical_s, quantum_s, noisy_s]]
    cols = ["#2196F3", "#f44336", "#FF9800"]
    bars = ax.bar(cats, rates, color=cols, edgecolor="k")
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{r:.0%}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Violation Rate")
    ax.set_title("Fraction with |S| > 2")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    noise_levels = np.linspace(0, 1, 20)
    mean_s, viol_rate = [], []
    for noise in noise_levels:
        ss = [abs(calculate_s_value(calculate_expectation_values_direct(
            _simulate_quantum(100, noise, rng.integers(100000))))) for _ in range(80)]
        mean_s.append(np.mean(ss))
        viol_rate.append(sum(s > 2 for s in ss) / len(ss))
    ax.plot(noise_levels, mean_s, "o-", color="#f44336", linewidth=2, markersize=5, label="Mean |S|")
    ax2 = ax.twinx()
    ax2.plot(noise_levels, viol_rate, "s-", color="#2196F3", linewidth=2, markersize=5, label="Violation rate")
    ax.axhline(y=2, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Noise fraction")
    ax.set_ylabel("Mean |S|", color="#f44336")
    ax2.set_ylabel("Violation rate", color="#2196F3")
    ax.set_title("Noise Degrades Violation")
    ax.legend(loc="center left", fontsize=8)
    ax2.legend(loc="center right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{OUT}/chsh_classical_vs_quantum.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  chsh_classical_vs_quantum.png")


# ═════════════════════════════════════════════════════════════════════
# Run all
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating README images...\n")

    print("QC plots:")
    qc_bell_states()
    qc_chsh_angle_sweep()
    qc_werner_sweep()
    qc_s_heatmap()
    qc_entanglement_landscape()
    qc_semantic_states()
    qc_parameterized_entanglement()

    print("\nCHSH plots:")
    chsh_classical_vs_quantum()

    print(f"\nDone! {len([f for f in __import__('os').listdir(OUT) if f.endswith('.png')])} images in {OUT}/")
