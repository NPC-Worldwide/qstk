"""Generate all README images headlessly. Run from repo root:
    python docs/generate_images.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider

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
# Decoherence
# ═════════════════════════════════════════════════════════════════════

from qstk.decoherence import compute_decoherence_metrics

SAMPLES = {
    0.3: "The theory of quantum entanglement describes correlations between "
         "particles that cannot be explained by classical physics. When two "
         "particles are entangled, measuring one instantly affects the other.",
    0.7: "Quantum entanglement weaves particles into shared destinies — "
         "measurement of one collapses the wavefunction of its partner. "
         "Bell's theorem shatters classical intuitions about separability.",
    1.0: "The entangled particles dance in superposition, their states a "
         "kaleidoscope of possibility until observation crystallizes reality. "
         "Quantum teleportation leverages this spooky correlation.",
    1.3: "Particles whispering across void-space, entanglement webs spreading "
         "through measurement collapse — the universe breathes in probabilistic "
         "waves. Schrödinger's cat grins from every superposed branch.",
    1.6: "ENTANGLEMENT spiraling through phase-space || the Bell operator "
         "B_CHSH = A⊗B - A⊗B' + A'⊗B + A'⊗B' — violation means NON-LOCAL "
         "// particles dream in Hilbert space ~10^(-35)m uncertain.",
    1.9: "квантовая ENTANGLEMENT 量子もつれ || Bell >> 2√2 sp!n st@tes "
         "{|0⟩, |1⟩} → def quantum_teleport(state): return alice.measure() "
         "XOR bob.rotate(π/4) |ψ⟩ = α|00⟩ + β|11⟩ СПИН スピン",
    2.2: "qU@nTuM ψ⟩⟨ψ| ≈ ∫d³x ρ(x) → 量子 kvantová 양자 QUANTUM "
         "def f(x): return x**2 + 1j*x.conj() 0xDEADBEEF << 42 "
         "кот Шрёдингера 猫 ネコ {{{SPIN}}} [|↑⟩+|↓⟩]/√2",
}


def decoherence_dashboard():
    temps = sorted(SAMPLES.keys())
    metrics = {t: compute_decoherence_metrics(SAMPLES[t], "quantum entanglement") for t in temps}

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("LLM Decoherence: How Temperature Breaks Coherence", fontsize=15, fontweight="bold")
    plt.subplots_adjust(hspace=0.4, wspace=0.35)

    t_arr = np.array(temps)
    script_div = [metrics[t].script_diversity for t in temps]
    char_ent = [metrics[t].char_entropy for t in temps]
    word_ent = [metrics[t].word_entropy for t in temps]
    code_dens = [metrics[t].code_fragment_density for t in temps]
    coherent = [metrics[t].longest_coherent_run for t in temps]

    # Script diversity
    ax = axes[0, 0]
    ax.bar(range(len(temps)), script_div,
           color=plt.cm.hot(np.linspace(0.2, 0.8, len(temps))),
           edgecolor="k", linewidth=0.5)
    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([f"{t:.1f}" for t in temps])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Script Diversity")
    ax.set_title("Unicode Script Diversity")
    ax.axhline(y=2, color="orange", linestyle="--", alpha=0.7, label="Decoherence threshold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Entropy
    ax = axes[0, 1]
    ax.plot(temps, char_ent, "o-", color="#2196F3", linewidth=2, markersize=8, label="Char entropy")
    ax.plot(temps, word_ent, "s-", color="#f44336", linewidth=2, markersize=8, label="Word entropy")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_title("Character & Word Entropy")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Code density
    ax = axes[0, 2]
    cols = ["#4CAF50" if d < 0.01 else "#FF9800" if d < 0.03 else "#f44336" for d in code_dens]
    ax.bar(range(len(temps)), code_dens, color=cols, edgecolor="k", linewidth=0.5)
    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([f"{t:.1f}" for t in temps])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Code Fragment Density")
    ax.set_title("Code Emergence")
    ax.grid(True, alpha=0.3, axis="y")

    # Coherence
    ax = axes[1, 0]
    ax.plot(temps, coherent, "D-", color="#9C27B0", linewidth=2, markersize=10)
    ax.fill_between(temps, coherent, alpha=0.2, color="#9C27B0")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Words")
    ax.set_title("Longest Coherent Run")
    ax.grid(True, alpha=0.3)

    # Script composition
    ax = axes[1, 1]
    latin_f, other_f = [], []
    for t in temps:
        m = metrics[t]
        total = sum(m.script_distribution.values())
        lat = m.script_distribution.get("Latin", 0)
        latin_f.append(lat / total if total else 0)
        other_f.append(1 - lat / total if total else 0)
    x = np.arange(len(temps))
    ax.bar(x, latin_f, color="#2196F3", label="Latin", edgecolor="k", linewidth=0.5)
    ax.bar(x, other_f, bottom=latin_f, color="#FF9800", label="Non-Latin", edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.1f}" for t in temps])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Fraction")
    ax.set_title("Script Composition")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Summary
    ax = axes[1, 2]
    def norm01(arr):
        mn, mx = min(arr), max(arr)
        return [(v - mn) / (mx - mn) if mx > mn else 0.5 for v in arr]
    metric_names = ["Script Div.", "Char Entropy", "Code Density",
                    "1/Coherence", "Word Entropy"]
    all_normed = {
        "Script Div.": norm01(script_div),
        "Char Entropy": norm01(char_ent),
        "Code Density": norm01(code_dens),
        "1/Coherence": norm01([1 / max(c, 1) for c in coherent]),
        "Word Entropy": norm01(word_ent),
    }
    for i, t in enumerate(temps):
        vals = [all_normed[k][i] for k in metric_names]
        color = plt.cm.hot(t / max(temps) * 0.8)
        ax.plot(range(len(metric_names)), vals, "o-", color=color, linewidth=1.5,
                markersize=6, alpha=0.7, label=f"T={t:.1f}")
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Normalized")
    ax.set_title("Decoherence Profile")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    fig.savefig(f"{OUT}/decoherence_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  decoherence_dashboard.png")


# ═════════════════════════════════════════════════════════════════════
# Orbital Dynamics
# ═════════════════════════════════════════════════════════════════════

from qstk.orbits import compute_trajectory_dynamics, compute_orbital_elements, fit_ellipse


def orbital_gallery():
    orbit_configs = {
        "Circular": lambda t, rng: np.column_stack([2 * np.cos(t) + rng.normal(0, 0.05, len(t)),
                                                     2 * np.sin(t) + rng.normal(0, 0.05, len(t))]),
        "Elliptical": lambda t, rng: np.column_stack([3 * np.cos(t) + rng.normal(0, 0.05, len(t)),
                                                       1.5 * np.sin(t) + rng.normal(0, 0.05, len(t))]),
        "Spiral In": lambda t, rng: np.column_stack([3 * np.exp(-0.1 * t) * np.cos(t) + rng.normal(0, 0.05, len(t)),
                                                      3 * np.exp(-0.1 * t) * np.sin(t) + rng.normal(0, 0.05, len(t))]),
        "Spiral Out": lambda t, rng: np.column_stack([0.5 * np.exp(0.08 * t) * np.cos(t) + rng.normal(0, 0.05, len(t)),
                                                       0.5 * np.exp(0.08 * t) * np.sin(t) + rng.normal(0, 0.05, len(t))]),
        "Figure-8": lambda t, rng: np.column_stack([np.sin(t) + rng.normal(0, 0.05, len(t)),
                                                     np.sin(2 * t) / 2 + rng.normal(0, 0.05, len(t))]),
    }

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    fig.suptitle("Orbital Dynamics Gallery", fontsize=15, fontweight="bold")
    rng = np.random.default_rng(42)

    for ax, (name, gen_fn) in zip(axes, orbit_configs.items()):
        t = np.linspace(0, 4 * np.pi, 300)
        pos = gen_fn(t, rng)
        dyn = compute_trajectory_dynamics(pos)
        n = len(pos)
        colors = plt.cm.viridis(np.linspace(0, 1, n))
        for i in range(n - 1):
            ax.plot(pos[i:i + 2, 0], pos[i:i + 2, 1], color=colors[i], linewidth=1.2, alpha=0.7)
        ax.plot(pos[0, 0], pos[0, 1], "go", markersize=8, zorder=5)
        ax.plot(pos[-1, 0], pos[-1, 1], "r*", markersize=12, zorder=5)

        if name not in ["Spiral Out"]:
            try:
                ep = fit_ellipse(pos)
                e = Ellipse(xy=ep["center"], width=2 * ep["semi_major"], height=2 * ep["semi_minor"],
                            angle=np.degrees(ep["angle"]), fill=False, edgecolor="red",
                            linestyle="--", linewidth=1.5)
                ax.add_patch(e)
            except Exception:
                pass

        ax.set_title(f"{name}\nλ={dyn.lyapunov_exponent:.2f}", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{OUT}/orbital_gallery.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  orbital_gallery.png")


def orbital_detail():
    t = np.linspace(0, 4 * np.pi, 500)
    rng = np.random.default_rng(7)
    pos = np.column_stack([3 * np.cos(t) + rng.normal(0, 0.03, 500),
                            1.5 * np.sin(t) + rng.normal(0, 0.03, 500)])
    dyn = compute_trajectory_dynamics(pos)
    orb = compute_orbital_elements(pos, dyn.velocities)
    steps = np.arange(len(pos))

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle(f"Elliptical Orbit Detail — e={orb.eccentricity:.3f}, {orb.orbit_type.value}",
                 fontsize=14, fontweight="bold")
    plt.subplots_adjust(hspace=0.35, wspace=0.35)

    ax = axes[0, 0]
    ax.plot(pos[:, 0], pos[:, 1], "b-", alpha=0.4, linewidth=1)
    step = 20
    ax.quiver(pos[::step, 0], pos[::step, 1],
              dyn.velocities[::step, 0], dyn.velocities[::step, 1],
              color="red", scale=2, width=0.004, alpha=0.6)
    ep = fit_ellipse(pos)
    e = Ellipse(xy=ep["center"], width=2 * ep["semi_major"], height=2 * ep["semi_minor"],
                angle=np.degrees(ep["angle"]), fill=False, edgecolor="green", linestyle="--", linewidth=2)
    ax.add_patch(e)
    ax.set_title("Trajectory + Velocity")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, np.linalg.norm(dyn.velocities, axis=1), color="#2196F3", linewidth=1.5)
    ax.set_title("Speed")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(steps, dyn.kinetic_energy, label="Kinetic", color="#f44336", linewidth=1.5)
    ax.plot(steps, dyn.potential_energy, label="Potential", color="#2196F3", linewidth=1.5)
    ax.plot(steps, dyn.total_energy, label="Total", color="k", linewidth=2)
    ax.set_title("Energy")
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, dyn.angular_momentum, color="#9C27B0", linewidth=1.5)
    ax.set_title("Angular Momentum")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    dists = np.linalg.norm(pos, axis=1)
    ax.plot(steps, dists, color="#FF9800", linewidth=1.5)
    ax.axhline(y=orb.periapsis, color="blue", linestyle="--", alpha=0.7, label=f"Periapsis={orb.periapsis:.2f}")
    ax.axhline(y=orb.apoapsis, color="red", linestyle="--", alpha=0.7, label=f"Apoapsis={orb.apoapsis:.2f}")
    ax.set_title("Distance from Origin")
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    dr = np.diff(dists)
    sc = ax.scatter(dists[1:], dr, c=steps[1:], cmap="viridis", s=5, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Time")
    ax.set_title("Phase Space (r, ṙ)")
    ax.set_xlabel("r")
    ax.set_ylabel("dr/dt")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{OUT}/orbital_detail.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  orbital_detail.png")


# ═════════════════════════════════════════════════════════════════════
# Feynman-Kac
# ═════════════════════════════════════════════════════════════════════

from qstk.feynman_kac import Agent, Environment


def feynman_kac_sim():
    external = {"difficulty": {0: 0.1, 10: 0.3, 20: 0.9, 30: 0.5, 40: 0.1}}
    env = Environment(external_factors=external)
    rng = np.random.default_rng(42)
    for i in range(50):
        personality = {
            "bias": rng.uniform(-0.8, 0.8),
            "opinion_adaptability": rng.uniform(0.02, 0.1),
            "social_susceptibility": rng.uniform(0.05, 0.3),
            "opinion_volatility": rng.uniform(0.05, 0.25),
            "energy_recovery_rate": rng.uniform(0.03, 0.08),
            "energy_volatility": rng.uniform(0.01, 0.04),
        }
        agent = Agent(i, {"opinion": rng.uniform(-0.5, 0.5), "energy": rng.uniform(0.5, 1.0)},
                      personality, env, state_bounds={"opinion": (-1, 1), "energy": (0, 1)})
        env.add_agent(agent)

    history = env.simulate(total_time=40.0, dt=0.05)

    times = [h["time"] for h in history]
    n_agents = [h["num_agents"] for h in history]
    mean_op = [h.get("mean_opinion", 0) for h in history]
    var_op = [h.get("var_opinion", 0) for h in history]
    mean_en = [h.get("mean_energy", 0) for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Feynman-Kac Agent Population Simulation", fontsize=15, fontweight="bold")
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    # Population
    ax = axes[0, 0]
    ax.plot(times, n_agents, "b-", linewidth=2)
    ax.set_title("Population Size")
    ax.set_xlabel("Time")
    ax.set_ylabel("Agents")
    ax2 = ax.twinx()
    dt = np.linspace(0, 40, 200)
    dv = [env.get_external_factor("difficulty", t) for t in dt]
    ax2.fill_between(dt, dv, alpha=0.15, color="red")
    ax2.set_ylabel("Difficulty", color="red")
    ax2.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3)

    # Opinion
    ax = axes[0, 1]
    std_op = [np.sqrt(v) for v in var_op]
    ax.plot(times, mean_op, "g-", linewidth=2)
    ax.fill_between(times, [m - s for m, s in zip(mean_op, std_op)],
                     [m + s for m, s in zip(mean_op, std_op)], alpha=0.2, color="green")
    ax.set_title("Opinion (mean ± std)")
    ax.set_xlabel("Time")
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)

    # Energy
    ax = axes[0, 2]
    ax.plot(times, mean_en, "r-", linewidth=2)
    ax.set_title("Mean Energy")
    ax.set_xlabel("Time")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Trajectories
    ax = axes[1, 0]
    for agent in env.agents[:20]:
        ops = [h.get("opinion", 0) for h in agent.history]
        ax.plot(np.linspace(0, 40, len(ops)), ops, alpha=0.4, linewidth=0.8)
    ax.set_title("Individual Trajectories")
    ax.set_xlabel("Time")
    ax.set_ylabel("Opinion")
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)

    # Final distribution
    ax = axes[1, 1]
    if env.agents:
        ax.hist([a.state["opinion"] for a in env.agents], bins=20, color="#4CAF50", edgecolor="k")
    ax.set_title(f"Final Opinion (n={len(env.agents)})")
    ax.set_xlabel("Opinion")
    ax.grid(True, alpha=0.3)

    # Phase space
    ax = axes[1, 2]
    if env.agents:
        ops = [a.state["opinion"] for a in env.agents]
        ens = [a.state["energy"] for a in env.agents]
        biases = [a.personality["bias"] for a in env.agents]
        sc = ax.scatter(ops, ens, c=biases, cmap="RdBu_r", s=40, edgecolors="k",
                        linewidths=0.5, vmin=-0.8, vmax=0.8)
        plt.colorbar(sc, ax=ax, label="Bias")
    ax.set_title("Opinion-Energy Phase Space")
    ax.set_xlabel("Opinion")
    ax.set_ylabel("Energy")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{OUT}/feynman_kac.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  feynman_kac.png")


def feynman_kac_polarization():
    def polarizing_drift(state, personality, env, t):
        op = state.get("opinion", 0)
        return {"opinion": op - op ** 3 + personality.get("bias", 0) * 0.1,
                "energy": personality.get("energy_recovery_rate", 0.05) - 0.03}

    def noisy_diffusion(state, personality, env, t):
        return {"opinion": 0.15 + 0.1 * env.get_external_factor("noise", t), "energy": 0.02}

    env = Environment(external_factors={"noise": {0: 0.1, 10: 0.5, 20: 1.0, 30: 0.3, 40: 0.1}})
    rng = np.random.default_rng(123)
    for i in range(80):
        p = {"bias": rng.uniform(-0.3, 0.3), "energy_recovery_rate": rng.uniform(0.03, 0.07)}
        a = Agent(i, {"opinion": rng.uniform(-0.2, 0.2), "energy": 0.9}, p, env,
                  drift_fn=polarizing_drift, diffusion_fn=noisy_diffusion,
                  state_bounds={"opinion": (-1.5, 1.5), "energy": (0, 1)})
        env.add_agent(a)
    history = env.simulate(total_time=40.0, dt=0.05)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Double-Well Polarization Dynamics", fontsize=14, fontweight="bold")

    ax = axes[0]
    for agent in env.agents[:30]:
        ops = [h.get("opinion", 0) for h in agent.history]
        ax.plot(np.linspace(0, 40, len(ops)), ops, alpha=0.3, linewidth=0.7)
    ax.axhline(y=1, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=-1, color="blue", linestyle="--", alpha=0.5)
    ax.set_title("Opinion Trajectories")
    ax.set_xlabel("Time")
    ax.set_ylabel("Opinion")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    times_arr = [h["time"] for h in history]
    for snap_t in [0, 5, 10, 20, 40]:
        idx = min(range(len(times_arr)), key=lambda i: abs(times_arr[i] - snap_t))
        ops = [a.history[idx].get("opinion", 0) for a in env.agents if idx < len(a.history)]
        if ops:
            ax.hist(ops, bins=20, alpha=0.4, label=f"t={snap_t}", density=True, range=(-1.5, 1.5))
    ax.set_title("Opinion Distribution Over Time")
    ax.set_xlabel("Opinion")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    x = np.linspace(-1.5, 1.5, 100)
    ax.plot(x, x - x ** 3, "b-", linewidth=2.5, label="Drift: x - x³")
    ax.fill_between(x, x - x ** 3, alpha=0.1, color="blue")
    ax.axhline(y=0, color="gray", linestyle="--")
    ax.axvline(x=-1, color="red", linestyle=":", alpha=0.7, label="Stable fixed points")
    ax.axvline(x=1, color="red", linestyle=":", alpha=0.7)
    ax.axvline(x=0, color="orange", linestyle=":", alpha=0.7, label="Unstable")
    ax.set_title("Phase Portrait")
    ax.set_xlabel("Opinion (x)")
    ax.set_ylabel("dx/dt")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{OUT}/feynman_kac_polarization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  feynman_kac_polarization.png")


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

    print("\nDecoherence plots:")
    decoherence_dashboard()

    print("\nOrbital plots:")
    orbital_gallery()
    orbital_detail()

    print("\nFeynman-Kac plots:")
    feynman_kac_sim()
    feynman_kac_polarization()

    print(f"\nDone! {len([f for f in __import__('os').listdir(OUT) if f.endswith('.png')])} images in {OUT}/")
