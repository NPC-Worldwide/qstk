"""Decoherence metrics analysis — no LLM calls needed.

Demonstrates the decoherence metric computation on synthetic text samples
that simulate what happens when LLMs are pushed to high temperatures.

Run: python examples/decoherence_analysis.py
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from chroptiks.plotting_utils import scatter, plothist, makefig

from qstk.decoherence import compute_decoherence_metrics


# ── Synthetic text samples at increasing "temperature" ──────────────

SAMPLES = {
    0.3: (
        "The theory of quantum entanglement describes correlations between "
        "particles that cannot be explained by classical physics. When two "
        "particles are entangled, measuring one instantly affects the other, "
        "regardless of distance. This phenomenon was famously called "
        "'spooky action at a distance' by Einstein."
    ),
    0.7: (
        "Quantum entanglement weaves particles into shared destinies — "
        "measurement of one collapses the wavefunction of its partner "
        "across any distance. The Bell inequalities proved that no local "
        "hidden variable theory can reproduce quantum predictions. This "
        "non-locality is now a resource for quantum computing."
    ),
    1.0: (
        "The entangled particles dance in superposition, their states "
        "a kaleidoscope of possibility until observation crystallizes "
        "reality. Bell's theorem shatters classical intuitions about "
        "separability. Quantum teleportation leverages this spooky "
        "correlation — information travels without traversing space."
    ),
    1.3: (
        "Particles whispering across void-space, entanglement webs "
        "spreading through measurement collapse — the universe breathes "
        "in probabilistic waves. Hidden variables? No: Bell proved the "
        "cosmos genuinely indeterminate. Schrödinger's cat grins from "
        "every superposed branch of the wavefunction-tree."
    ),
    1.6: (
        "ENTANGLEMENT spiraling through phase-space || the Bell operator "
        "B_CHSH = A⊗B - A⊗B' + A'⊗B + A'⊗B' — violation means NON-LOCAL "
        "// particles dream in Hilbert space dimensions >> 3.14159 the "
        "cat is alive AND dead — quantum foam bubbles at the Planck scale "
        "~10^(-35)m where geometry itself becomes uncertain."
    ),
    1.9: (
        "квантовая запутанность ENTANGLEMENT 量子もつれ || Bell >> 2√2 "
        "the particles sp!n in superposed st@tes {|0⟩, |1⟩} → measurement "
        "collaps3s → def quantum_teleport(state): return alice.measure() "
        "XOR bob.rotate(π/4) |ψ⟩ = α|00⟩ + β|11⟩ СПИН вверх スピン "
        "class WaveFunction: def __init__(self): self.collapsed = False"
    ),
    2.2: (
        "qU@nTuM ψ⟩⟨ψ| ≈ ∫d³x ρ(x) → 量子 kvantová 양자 QUANTUM "
        "def f(x): return x**2 + 1j*x.conj() // 0xDEADBEEF << 42 "
        "кот Шрёдингера 猫 ネコ gato chat {{{SPIN}}} [|↑⟩+|↓⟩]/√2 "
        "import quantum; from reality import * # EVERYTHING IS WAVES "
        "∀ε>0 ∃δ>0: |ψ-φ|<δ ⟹ |⟨ψ|O|ψ⟩-⟨φ|O|φ⟩|<ε 🌀🔮✨"
    ),
}


def analyze_samples():
    """Compute decoherence metrics for all synthetic samples."""

    print("Computing decoherence metrics for synthetic samples...\n")

    temperatures = sorted(SAMPLES.keys())
    all_metrics = {}

    for temp in temperatures:
        text = SAMPLES[temp]
        metrics = compute_decoherence_metrics(text, prompt="quantum entanglement")
        all_metrics[temp] = metrics
        print(f"  T={temp:.1f}: scripts={metrics.script_diversity:2d}  "
              f"char_entropy={metrics.char_entropy:.2f}  "
              f"word_entropy={metrics.word_entropy:.2f}  "
              f"code_density={metrics.code_fragment_density:.3f}  "
              f"coherent_run={metrics.longest_coherent_run}")

    return temperatures, all_metrics


def plot_decoherence_dashboard(temperatures, all_metrics):
    """Plot comprehensive decoherence analysis."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("LLM Decoherence: How Temperature Breaks Coherence",
                 fontsize=16, fontweight="bold")
    plt.subplots_adjust(hspace=0.35, wspace=0.35)

    temps = np.array(temperatures)

    # Extract metric arrays
    script_div = [all_metrics[t].script_diversity for t in temperatures]
    char_ent = [all_metrics[t].char_entropy for t in temperatures]
    word_ent = [all_metrics[t].word_entropy for t in temperatures]
    code_dens = [all_metrics[t].code_fragment_density for t in temperatures]
    coherent = [all_metrics[t].longest_coherent_run for t in temperatures]
    ws_ratio = [all_metrics[t].whitespace_ratio for t in temperatures]
    punct_ratio = [all_metrics[t].punctuation_ratio for t in temperatures]
    num_ratio = [all_metrics[t].numeric_ratio for t in temperatures]
    avg_word = [all_metrics[t].avg_word_length for t in temperatures]

    # 1. Script diversity
    ax = axes[0, 0]
    ax.bar(range(len(temps)), script_div,
           color=plt.cm.hot(np.linspace(0.2, 0.8, len(temps))),
           edgecolor="k", linewidth=0.5)
    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([f"{t:.1f}" for t in temps])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Script Diversity")
    ax.set_title("Unicode Script Diversity\n(1=monolingual, >3=multilingual)")
    ax.grid(True, alpha=0.3, axis="y")
    # Mark decoherence threshold
    ax.axhline(y=2, color="orange", linestyle="--", alpha=0.7, label="Threshold")
    ax.legend(fontsize=8)

    # 2. Entropy measures
    ax = axes[0, 1]
    ax.plot(temps, char_ent, "o-", color="#2196F3", linewidth=2, markersize=8, label="Char entropy")
    ax.plot(temps, word_ent, "s-", color="#f44336", linewidth=2, markersize=8, label="Word entropy")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_title("Character & Word Entropy")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Code fragment density
    ax = axes[0, 2]
    colors = ["#4CAF50" if d < 0.01 else "#FF9800" if d < 0.03 else "#f44336" for d in code_dens]
    ax.bar(range(len(temps)), code_dens, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([f"{t:.1f}" for t in temps])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Code Fragment Density")
    ax.set_title("Code Emergence\n(code-like patterns per word)")
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Coherent run length
    ax = axes[1, 0]
    ax.plot(temps, coherent, "D-", color="#9C27B0", linewidth=2, markersize=10)
    ax.fill_between(temps, coherent, alpha=0.2, color="#9C27B0")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Longest Coherent Run (words)")
    ax.set_title("Coherence Length\n(longest same-script word streak)")
    ax.grid(True, alpha=0.3)

    # 5. Text composition breakdown (stacked bar)
    ax = axes[1, 1]
    latin_frac = []
    other_frac = []
    for t in temperatures:
        m = all_metrics[t]
        total = sum(m.script_distribution.values())
        latin = m.script_distribution.get("Latin", 0)
        latin_frac.append(latin / total if total > 0 else 0)
        other_frac.append(1 - latin / total if total > 0 else 0)
    x = np.arange(len(temps))
    ax.bar(x, latin_frac, color="#2196F3", label="Latin", edgecolor="k", linewidth=0.5)
    ax.bar(x, other_frac, bottom=latin_frac, color="#FF9800", label="Non-Latin",
           edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.1f}" for t in temps])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Fraction")
    ax.set_title("Script Composition")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # 6. Decoherence summary radar
    ax = axes[1, 2]
    # Normalize all metrics to [0, 1] for overlay comparison
    def norm01(arr):
        mn, mx = min(arr), max(arr)
        return [(v - mn) / (mx - mn) if mx > mn else 0.5 for v in arr]

    metrics_to_show = {
        "Script Div.": norm01(script_div),
        "Char Entropy": norm01(char_ent),
        "Code Density": norm01(code_dens),
        "1/Coherence": norm01([1 / max(c, 1) for c in coherent]),
        "Numeric Ratio": norm01(num_ratio),
    }
    # Plot as parallel coordinates
    labels = list(metrics_to_show.keys())
    for i, t in enumerate(temperatures):
        vals = [metrics_to_show[k][i] for k in labels]
        color = plt.cm.hot(t / max(temps) * 0.8)
        ax.plot(range(len(labels)), vals, "o-", color=color, linewidth=1.5,
                markersize=6, alpha=0.7, label=f"T={t:.1f}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Normalized value")
    ax.set_title("Decoherence Profile (parallel coords)")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.show(block=False)
    return fig


def plot_script_distributions(temperatures, all_metrics):
    """Show how script diversity changes with temperature."""

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Script Distribution per Temperature", fontsize=14, fontweight="bold")

    for idx, temp in enumerate(temperatures):
        row, col = divmod(idx, 4)
        if row >= 2:
            break
        ax = axes[row, col]
        m = all_metrics[temp]
        scripts = m.script_distribution
        if scripts:
            sorted_s = sorted(scripts.items(), key=lambda x: -x[1])
            names = [s[0] for s in sorted_s]
            counts = [s[1] for s in sorted_s]
            colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
            ax.barh(range(len(names)), counts, color=colors, edgecolor="k", linewidth=0.5)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
        ax.set_title(f"T = {temp:.1f}", fontsize=11)
        ax.set_xlabel("Char count")
        ax.grid(True, alpha=0.3, axis="x")

    # Hide unused subplots
    for idx in range(len(temperatures), 8):
        row, col = divmod(idx, 4)
        if row < 2:
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.show(block=False)
    return fig


if __name__ == "__main__":
    plt.ion()

    print("=" * 60)
    print("Decoherence Analysis — Synthetic Temperature Sweep")
    print("=" * 60)

    temps, metrics = analyze_samples()

    print("\nGenerating plots...")
    fig1 = plot_decoherence_dashboard(temps, metrics)
    fig2 = plot_script_distributions(temps, metrics)

    print("\nAll plots open. Close windows or Ctrl+C to exit.")
    plt.show(block=True)
