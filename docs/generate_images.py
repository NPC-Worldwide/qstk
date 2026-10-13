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
    ax.hist(classical_s, bins=20 if len(set(round(s, 6) for s in classical_s)) > 1 else [min(classical_s) - 0.1, max(classical_s) + 0.1], alpha=0.6, color="#2196F3", label="Classical", density=True)
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


def _tokenize(text):
    return text.lower().split()


def _term_counts(text, vocab):
    tokens = _tokenize(text)
    return {term: tokens.count(term) for term in vocab}


def _compute_idf(documents, vocab):
    n = len(documents)
    idf = {}
    for term in vocab:
        df = sum(1 for d in documents if term in _tokenize(d))
        idf[term] = np.log((n + 1) / (df + 1)) + 1.0
    return idf


def _tfidf_vector(counts, idf, terms):
    total = sum(counts.values())
    vec = np.array([counts.get(term, 0) / total * idf.get(term, 0.0) for term in terms], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _pad_to_power_of_two(vec):
    dim = 2 ** int(np.ceil(np.log2(max(len(vec), 1))))
    out = np.zeros(dim, dtype=complex)
    out[:len(vec)] = vec
    norm = np.linalg.norm(out)
    if norm == 0:
        out[0] = 1.0
        return out
    return out / norm


def _select_top_terms_by_query(vocab, idf, query, k):
    counts = _term_counts(query, vocab)
    total = sum(counts.values())
    weights = {term: counts.get(term, 0) / total * idf.get(term, 0.0) if total > 0 else 0.0 for term in vocab}
    return [term for term, _ in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:k]]


def _run_swap_test(state_a, state_b, n_shots, seed):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qstk.qc.circuits import swap_test_circuit
    simulator = AerSimulator(seed_simulator=seed)
    qc = swap_test_circuit(state_a, state_b)
    transpiled = transpile(qc, simulator)
    counts = simulator.run(transpiled, shots=n_shots).result().get_counts(transpiled)
    total = sum(counts.values())
    p0 = counts.get("0", 0) / total if total > 0 else 0.5
    return 2.0 * p0 - 1.0


def _classical_cosine(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def nlp_semantic_search_bar():
    """Swap-test semantic search: quantum vs classical similarity."""
    corpus = [
        "quantum computing uses qubits to process information",
        "neural networks learn patterns from large datasets",
        "superconducting qubits operate at cryogenic temperatures",
        "natural language processing models understand text",
        "ion traps confine atoms for quantum experiments",
        "transformer architectures power modern nlp systems",
    ]
    query = "quantum information processing with qubits"

    vocab = sorted(set([t for d in corpus for t in _tokenize(d)]))
    idf = _compute_idf(corpus, vocab)
    terms = _select_top_terms_by_query(vocab, idf, query, 8)
    doc_vectors = [_tfidf_vector(_term_counts(d, vocab), idf, terms) for d in corpus]
    query_vec = _tfidf_vector(_term_counts(query, vocab), idf, terms)
    query_state = _pad_to_power_of_two(query_vec)
    doc_states = [_pad_to_power_of_two(v) for v in doc_vectors]

    docs = [d[:45] + "..." if len(d) > 45 else d for d in corpus]
    q_sims = [_run_swap_test(query_state, s, 8192, 42 + i) for i, s in enumerate(doc_states)]
    c_sims = [_classical_cosine(query_vec, v) ** 2 for v in doc_vectors]

    x = np.arange(len(docs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w / 2, q_sims, w, label="Quantum swap-test |<q|d>|²", color="#4CAF50", edgecolor="k", linewidth=0.5)
    ax.bar(x + w / 2, c_sims, w, label="Classical cos²", color="#2196F3", edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(docs, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Similarity")
    ax.set_title("Quantum Semantic Search: Swap-Test vs Classical Cosine", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(f"{OUT}/nlp_semantic_search.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  nlp_semantic_search.png")


def nlp_text_classification_kernel():
    """Heatmap of the quantum kernel matrix for sentiment classification."""
    documents = [
        ("the movie was absolutely excellent and thrilling", 1),
        ("a wonderful performance by the amazing cast", 1),
        ("one of the greatest films ever brilliant and excellent", 1),
        ("amazing direction excellent story truly wonderful", 1),
        ("terrible acting and a boring predictable plot", 0),
        ("worst movie ever completely awful dull waste of time", 0),
        ("awful acting boring story disappointing in every way", 0),
        ("terrible dull and awful no redeeming qualities", 0),
    ]
    vocabulary = [
        "excellent", "amazing", "brilliant", "wonderful",
        "terrible", "awful", "boring", "dull",
    ]

    idf = _compute_idf([text for text, _ in documents], vocabulary)
    states = [_pad_to_power_of_two(_tfidf_vector(_term_counts(text, vocabulary), idf, vocabulary))
              for text, _ in documents]
    labels = ["pos" if label else "neg" for _, label in documents]

    n = len(documents)
    K = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            k = _run_swap_test(states[i], states[j], 4096, 42 + i * n + j)
            K[i, j] = k
            K[j, i] = k

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(K, cmap="RdBu_r", vmin=-0.3, vmax=1.0)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([f"{i}\n{l}" for i, l in enumerate(labels)], fontsize=9)
    ax.set_yticklabels([f"{i}\n{l}" for i, l in enumerate(labels)], fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{K[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
    plt.colorbar(im, ax=ax, label="Swap-test |<dᵢ|dⱼ>|²")
    ax.set_title("Quantum Kernel Matrix for Sentiment Classification", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUT}/nlp_text_classification_kernel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  nlp_text_classification_kernel.png")


def nlp_topic_clustering():
    """Quantum k-means vs classical k-means topic assignments."""
    documents = [
        ("the stock market rose after strong earnings as investors bought shares", "finance"),
        ("investors bought bonds as interest rates fell sharply in the market", "finance"),
        ("the quarterback threw a touchdown in the final minute of the game", "sports"),
        ("the team clinched the championship with a clean sweep of the series", "sports"),
        ("gene editing could cure inherited diseases according to the study", "biology"),
        ("the new study maps proteins inside a single cell with high precision", "biology"),
    ]
    vocabulary = [
        "stock", "market", "investors",
        "quarterback", "touchdown", "team", "championship",
        "gene", "cell", "proteins",
    ]
    k = 3

    def average_state(states):
        if not states:
            return None
        avg = np.zeros_like(states[0], dtype=complex)
        for s in states:
            avg += s
        return avg / np.linalg.norm(avg)

    def estimate_distance(state_a, state_b, n_shots, seed):
        k_est = _run_swap_test(state_a, state_b, n_shots, seed)
        return max(0.0, 2.0 - 2.0 * k_est)

    def quantum_kmeans(states, k, n_iterations, n_shots, seed):
        rng = np.random.default_rng(seed)
        centroids = [states[i].copy() for i in rng.choice(len(states), size=k, replace=False)]
        assignments = [0] * len(states)
        for iteration in range(n_iterations):
            new_assignments = []
            for i, state in enumerate(states):
                distances = [estimate_distance(state, centroid, n_shots, seed + iteration * 100 + i * 10 + c)
                             for c, centroid in enumerate(centroids)]
                new_assignments.append(int(np.argmin(distances)))
            if new_assignments == assignments and iteration > 0:
                assignments = new_assignments
                break
            assignments = new_assignments
            new_centroids = []
            for c in range(k):
                cluster = [states[i] for i, a in enumerate(assignments) if a == c]
                new_centroids.append(average_state(cluster) if cluster else centroids[c])
            centroids = new_centroids
        return assignments

    def classical_kmeans(vectors, k, n_iterations, seed):
        rng = np.random.default_rng(seed)
        centroids = [vectors[i].copy() for i in rng.choice(len(vectors), size=k, replace=False)]
        assignments = [0] * len(vectors)
        for iteration in range(n_iterations):
            new_assignments = []
            for vec in vectors:
                distances = [np.linalg.norm(vec - c) for c in centroids]
                new_assignments.append(int(np.argmin(distances)))
            if new_assignments == assignments and iteration > 0:
                assignments = new_assignments
                break
            assignments = new_assignments
            new_centroids = []
            for c in range(k):
                cluster = [vectors[i] for i, a in enumerate(assignments) if a == c]
                new_centroids.append(np.mean(cluster, axis=0) if cluster else centroids[c])
                norm = np.linalg.norm(new_centroids[-1])
                if norm > 0:
                    new_centroids[-1] = new_centroids[-1] / norm
            centroids = new_centroids
        return assignments

    idf = _compute_idf([text for text, _ in documents], vocabulary)
    vectors = [_tfidf_vector(_term_counts(text, vocabulary), idf, vocabulary) for text, _ in documents]
    states = [_pad_to_power_of_two(v) for v in vectors]
    true_labels = [label for _, label in documents]
    q_assignments = quantum_kmeans(states, k, 8, 4096, 42)
    c_assignments = classical_kmeans(vectors, k, 8, 42)

    n = len(documents)
    pair_true = np.array([[true_labels[i] == true_labels[j] for j in range(n)] for i in range(n)])
    pair_q = np.array([[q_assignments[i] == q_assignments[j] for j in range(n)] for i in range(n)])
    pair_c = np.array([[c_assignments[i] == c_assignments[j] for j in range(n)] for i in range(n)])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Quantum-Enhanced Topic Clustering", fontsize=14, fontweight="bold")

    titles = ["True Topics", "Quantum K-Means", "Classical K-Means"]
    matrices = [pair_true, pair_q, pair_c]
    for ax, mat, title in zip(axes, matrices, titles):
        im = ax.imshow(mat, cmap="binary", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels([f"{i}" for i in range(n)])
        ax.set_yticklabels([f"{i}" for i in range(n)])
        for i in range(n):
            for j in range(n):
                ax.text(j, i, "1" if mat[i, j] else "0", ha="center", va="center", color="#d32f2f" if mat[i, j] else "gray", fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout()
    fig.savefig(f"{OUT}/nlp_topic_clustering.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  nlp_topic_clustering.png")


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

    print("\nQuantum NLP plots:")
    nlp_semantic_search_bar()
    nlp_text_classification_kernel()
    nlp_topic_clustering()

    print(f"\nDone! {len([f for f in __import__('os').listdir(OUT) if f.endswith('.png')])} images in {OUT}/")
