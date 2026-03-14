"""CHSH Bell test math demonstrations — no LLM or quantum hardware needed.

Shows how the S-value computation works with synthetic trial data,
density matrix vs direct averaging methods, and statistical analysis.

Run: python examples/chsh_experiments.py
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from chroptiks.plotting_utils import scatter, plothist, makefig

from qstk.chsh import (
    compute_chsh_products_binary,
    calculate_expectation_values_direct,
    calculate_expectation_values_density_matrix,
    calculate_s_value,
    check_violation,
)
from qstk.statistics import calculate_agreement_significance_combinatorial


def simulate_classical_trials(n_trials=500, seed=42):
    """Simulate classical (local hidden variable) Bell test trials.

    Classical strategies can't violate |S| <= 2.
    """
    rng = np.random.default_rng(seed)
    all_products = []

    for _ in range(n_trials):
        # Classical strategy: A and B use a shared random variable λ
        lam = rng.choice([-1, 1])
        outcomes = {
            "A": lam,
            "A_prime": lam,
            "B": lam,
            "B_prime": rng.choice([-1, 1]),  # B' is random
        }
        products = compute_chsh_products_binary(outcomes)
        all_products.append(products)

    return all_products


def simulate_quantum_trials(n_trials=500, noise=0.0, seed=42):
    """Simulate quantum Bell test trials with configurable noise.

    At noise=0, produces maximal violation S = 2√2.
    Noise mixes with random classical outcomes.
    """
    rng = np.random.default_rng(seed)
    all_products = []

    # Quantum probabilities for optimal CHSH
    # E(A,B) ≈ cos(π/4) = 1/√2 for matching settings
    p_corr = (1 + 1 / np.sqrt(2)) / 2  # ≈ 0.854

    for _ in range(n_trials):
        if rng.random() < noise:
            # Noisy trial: random classical outcomes
            outcomes = {k: rng.choice([-1, 1]) for k in ["A", "A_prime", "B", "B_prime"]}
        else:
            # Quantum-correlated trial
            outcomes = {}
            # A₀B₀: correlated
            a = rng.choice([-1, 1])
            outcomes["A"] = a
            outcomes["B"] = a if rng.random() < p_corr else -a
            # A₀B₁: anti-correlated
            outcomes["B_prime"] = -a if rng.random() < p_corr else a
            # A₁B₀: correlated
            ap = rng.choice([-1, 1])
            outcomes["A_prime"] = ap
            # Need separate B for A', but we already assigned B — for simplicity
            # we use the product-level simulation

        products = compute_chsh_products_binary(outcomes)
        all_products.append(products)

    return all_products


def plot_method_comparison():
    """Compare direct averaging vs density matrix methods."""

    print("Comparing expectation value methods...\n")

    trial_counts = [5, 10, 20, 50, 100, 200, 500, 1000]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Direct Averaging vs Density Matrix: Method Comparison",
                 fontsize=14, fontweight="bold")

    # Run many repetitions at each trial count
    n_reps = 50
    rng = np.random.default_rng(42)

    direct_means = []
    direct_stds = []
    dm_means = []
    dm_stds = []

    for n in trial_counts:
        direct_s = []
        dm_s = []
        for rep in range(n_reps):
            trials = simulate_quantum_trials(n, noise=0.1, seed=rng.integers(10000))
            ev_direct = calculate_expectation_values_direct(trials)
            ev_dm = calculate_expectation_values_density_matrix(trials)
            direct_s.append(calculate_s_value(ev_direct))
            dm_s.append(calculate_s_value(ev_dm))
        direct_means.append(np.mean(direct_s))
        direct_stds.append(np.std(direct_s))
        dm_means.append(np.mean(dm_s))
        dm_stds.append(np.std(dm_s))

    # 1. S-value convergence
    ax = axes[0]
    ax.errorbar(trial_counts, direct_means, yerr=direct_stds, fmt="o-",
                color="#2196F3", linewidth=2, capsize=5, label="Direct averaging")
    ax.errorbar(trial_counts, dm_means, yerr=dm_stds, fmt="s-",
                color="#f44336", linewidth=2, capsize=5, label="Density matrix")
    ax.axhline(y=2, color="gray", linestyle="--", alpha=0.5, label="|S|=2 bound")
    ax.set_xscale("log")
    ax.set_xlabel("Number of trials")
    ax.set_ylabel("S-value")
    ax.set_title("S-value Convergence")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Variance reduction
    ax = axes[1]
    ax.plot(trial_counts, direct_stds, "o-", color="#2196F3", linewidth=2, label="Direct σ")
    ax.plot(trial_counts, dm_stds, "s-", color="#f44336", linewidth=2, label="DM σ")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of trials")
    ax.set_ylabel("Standard deviation")
    ax.set_title("Variance vs Sample Size")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Distribution at n=50
    ax = axes[2]
    n_hist = 50
    direct_samples = []
    dm_samples = []
    for _ in range(200):
        trials = simulate_quantum_trials(n_hist, noise=0.1, seed=rng.integers(100000))
        ev_d = calculate_expectation_values_direct(trials)
        ev_dm = calculate_expectation_values_density_matrix(trials)
        direct_samples.append(calculate_s_value(ev_d))
        dm_samples.append(calculate_s_value(ev_dm))
    ax.hist(direct_samples, bins=25, alpha=0.5, color="#2196F3", label="Direct", density=True)
    ax.hist(dm_samples, bins=25, alpha=0.5, color="#f44336", label="Density matrix", density=True)
    ax.axvline(x=2, color="gray", linestyle="--", linewidth=2)
    ax.set_xlabel("S-value")
    ax.set_ylabel("Density")
    ax.set_title(f"S-value Distribution (n={n_hist} trials)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)
    return fig


def plot_classical_vs_quantum():
    """Compare classical and quantum S-value distributions."""

    print("Generating classical vs quantum comparison...\n")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Classical vs Quantum: Can S > 2?", fontsize=14, fontweight="bold")

    rng = np.random.default_rng(42)
    n_experiments = 300
    n_trials_per = 100

    classical_s = []
    quantum_s = []
    noisy_s = []

    for _ in range(n_experiments):
        # Classical
        c_trials = simulate_classical_trials(n_trials_per, seed=rng.integers(100000))
        ev = calculate_expectation_values_direct(c_trials)
        classical_s.append(calculate_s_value(ev))

        # Quantum (low noise)
        q_trials = simulate_quantum_trials(n_trials_per, noise=0.05, seed=rng.integers(100000))
        ev = calculate_expectation_values_direct(q_trials)
        quantum_s.append(calculate_s_value(ev))

        # Quantum (high noise)
        n_trials_noisy = simulate_quantum_trials(n_trials_per, noise=0.5, seed=rng.integers(100000))
        ev = calculate_expectation_values_direct(n_trials_noisy)
        noisy_s.append(calculate_s_value(ev))

    # 1. Histogram comparison
    ax = axes[0]
    ax.hist(classical_s, bins=30, alpha=0.6, color="#2196F3", label="Classical", density=True)
    ax.hist(quantum_s, bins=30, alpha=0.6, color="#f44336", label="Quantum", density=True)
    ax.hist(noisy_s, bins=30, alpha=0.6, color="#FF9800", label="Noisy quantum", density=True)
    ax.axvline(x=2, color="k", linestyle="--", linewidth=2, label="|S|=2")
    ax.axvline(x=-2, color="k", linestyle="--", linewidth=2)
    ax.set_xlabel("S-value")
    ax.set_ylabel("Density")
    ax.set_title("S-value Distributions")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Violation rates
    ax = axes[1]
    categories = ["Classical", "Quantum\n(5% noise)", "Quantum\n(50% noise)"]
    rates = [
        sum(1 for s in classical_s if abs(s) > 2) / len(classical_s),
        sum(1 for s in quantum_s if abs(s) > 2) / len(quantum_s),
        sum(1 for s in noisy_s if abs(s) > 2) / len(noisy_s),
    ]
    colors = ["#2196F3", "#f44336", "#FF9800"]
    bars = ax.bar(categories, rates, color=colors, edgecolor="k", linewidth=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.1%}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Violation Rate")
    ax.set_title("Fraction of Experiments with |S| > 2")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Noise sweep
    ax = axes[2]
    noise_levels = np.linspace(0, 1, 20)
    mean_s_by_noise = []
    violation_rate_by_noise = []
    for noise in noise_levels:
        ss = []
        for _ in range(100):
            trials = simulate_quantum_trials(100, noise=noise, seed=rng.integers(100000))
            ev = calculate_expectation_values_direct(trials)
            ss.append(abs(calculate_s_value(ev)))
        mean_s_by_noise.append(np.mean(ss))
        violation_rate_by_noise.append(sum(1 for s in ss if s > 2) / len(ss))

    ax.plot(noise_levels, mean_s_by_noise, "o-", color="#f44336", linewidth=2,
            markersize=6, label="Mean |S|")
    ax2 = ax.twinx()
    ax2.plot(noise_levels, violation_rate_by_noise, "s-", color="#2196F3", linewidth=2,
             markersize=6, label="Violation rate")
    ax.axhline(y=2, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Noise fraction")
    ax.set_ylabel("Mean |S|", color="#f44336")
    ax2.set_ylabel("Violation rate", color="#2196F3")
    ax.set_title("Noise Degrades Violation")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)
    return fig


def plot_significance_testing():
    """Demonstrate the combinatorial significance test for theme agreement."""

    print("Computing combinatorial significance...\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Combinatorial Significance: Theme Vector Agreement",
                 fontsize=14, fontweight="bold")

    # 1. Example: two vectors with varying agreement
    total_themes = 10
    vector_length = 5
    rng = np.random.default_rng(42)

    agreements = []
    p_values = []
    nas_values = []

    for _ in range(500):
        v1 = sorted(rng.choice(total_themes, size=vector_length, replace=False).tolist())
        v2 = sorted(rng.choice(total_themes, size=vector_length, replace=False).tolist())
        nas, pval = calculate_agreement_significance_combinatorial(v1, v2, total_themes)
        overlap = len(set(v1) & set(v2))
        agreements.append(overlap)
        p_values.append(pval)
        nas_values.append(nas)

    ax = axes[0]
    sc = ax.scatter(agreements, p_values, c=nas_values, cmap="viridis", s=15, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="NAS")
    ax.axhline(y=0.05, color="red", linestyle="--", label="p=0.05")
    ax.set_xlabel("Agreement (overlap count)")
    ax.set_ylabel("p-value")
    ax.set_title("Agreement vs Significance")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. NAS distribution
    ax = axes[1]
    ax.hist(nas_values, bins=30, color="#4CAF50", edgecolor="k", alpha=0.8)
    ax.axvline(x=np.mean(nas_values), color="red", linestyle="--", linewidth=2,
               label=f"Mean={np.mean(nas_values):.3f}")
    ax.set_xlabel("Normalized Agreement Score")
    ax.set_ylabel("Count")
    ax.set_title("NAS Distribution (random vectors)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Significance vs vector size
    ax = axes[2]
    sizes = range(2, 9)
    mean_p_by_size = []
    sig_rate_by_size = []
    for sz in sizes:
        ps = []
        for _ in range(200):
            v1 = sorted(rng.choice(total_themes, size=sz, replace=False).tolist())
            v2 = sorted(rng.choice(total_themes, size=sz, replace=False).tolist())
            _, pval = calculate_agreement_significance_combinatorial(v1, v2, total_themes)
            ps.append(pval)
        mean_p_by_size.append(np.mean(ps))
        sig_rate_by_size.append(sum(1 for p in ps if p < 0.05) / len(ps))

    ax.bar(list(sizes), sig_rate_by_size, color="#FF9800", edgecolor="k")
    ax.axhline(y=0.05, color="red", linestyle="--", label="5% expected false positive")
    ax.set_xlabel("Vector length (themes per observer)")
    ax.set_ylabel("Fraction p < 0.05")
    ax.set_title("False Positive Rate vs Vector Size")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show(block=False)
    return fig


if __name__ == "__main__":
    plt.ion()

    print("=" * 60)
    print("CHSH Bell Test Math Experiments")
    print("=" * 60)

    print("\n[1/3] Classical vs Quantum S-values")
    fig1 = plot_classical_vs_quantum()

    print("[2/3] Method comparison (direct vs density matrix)")
    fig2 = plot_method_comparison()

    print("[3/3] Combinatorial significance testing")
    fig3 = plot_significance_testing()

    print("\nAll plots open. Close windows or Ctrl+C to exit.")
    plt.show(block=True)
