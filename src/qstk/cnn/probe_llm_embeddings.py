"""
Probing GPT-2 embeddings for latent complex/phase structure.

This experiment takes the real-valued token embedding matrix from GPT-2,
projects it into complex space via the Hilbert transform, and then asks:

  1. Do the projected embeddings have non-random phase coherence?
  2. Do k-means clusters in phase space correspond to semantic categories?
  3. Is the interference score higher for semantically related word pairs
     than for unrelated pairs?
  4. Does the CHSH-like S-value for semantically entangled pairs violate
     the classical bound of 2?
  5. Does phase-space distance capture structure beyond cosine similarity?
  6. Do GPT-2's attention matrices (Q, K) show phase structure?

Connects the complex-valued NN probe tools (qstk.cnn.probe) back to the
quantum semantics Bell test framework (ket-nlp).

The analogy is biological: a sunflower's seeds arrange in Fibonacci spirals
not because the plant "knows" math, but because phyllotaxis emerges from
a simple growth rule. Similarly, if transformer embeddings show phase
structure in complex space, it would mean the optimization landscape
naturally selects for interference-like patterns -- the network discovers
wave mechanics without being told to.

Usage:
    /home/caug/npcww/qstk/qllm2/.venv/bin/python probe_llm_embeddings.py
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import spearmanr, pearsonr

# Add parent so we can import probe
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe import (
    embed_to_complex,
    phase_coherence,
    phase_clusters,
    interference_score,
    semantic_entanglement,
)


# ---------------------------------------------------------------------------
# 0. Load GPT-2 embedding matrix + tokenizer
# ---------------------------------------------------------------------------
def load_gpt2():
    """Load GPT-2 token embeddings and tokenizer."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("Loading GPT-2 model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    embeds = model.transformer.wte.weight.detach().numpy()  # [50257, 768]
    print(f"  Embedding matrix shape: {embeds.shape}")
    print(f"  Vocab size: {len(tokenizer)}")

    return embeds, tokenizer, model


def token_id(tokenizer, word):
    """Get the single-token ID for a word. Try with/without leading space."""
    # GPT-2 tokenizer prepends a space for most words
    candidates = [word, f" {word}", word.capitalize(), f" {word.capitalize()}"]
    for w in candidates:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    # Fall back to first token of the encoding
    ids = tokenizer.encode(f" {word}", add_special_tokens=False)
    return ids[0] if ids else None


# ---------------------------------------------------------------------------
# 1. Phase coherence analysis
# ---------------------------------------------------------------------------
def analyze_phase_coherence(embeds, n_random_baselines=20):
    """Compare phase coherence of real GPT-2 embeddings vs random baselines."""
    print("\n" + "=" * 60)
    print("1. PHASE COHERENCE ANALYSIS")
    print("=" * 60)

    results = {}

    for method in ["hilbert", "paired", "random_proj"]:
        z = embed_to_complex(embeds, method=method)
        coh = phase_coherence(z)

        # Random baselines: shuffle the embedding matrix, then project
        baseline_cohs = []
        for i in range(n_random_baselines):
            shuffled = embeds.copy()
            rng = np.random.RandomState(i)
            for row in range(shuffled.shape[0]):
                rng.shuffle(shuffled[row])
            z_shuf = embed_to_complex(shuffled, method=method)
            baseline_cohs.append(phase_coherence(z_shuf))

        baseline_mean = np.mean(baseline_cohs)
        baseline_std = np.std(baseline_cohs)
        z_score = (coh - baseline_mean) / (baseline_std + 1e-10)

        results[method] = {
            "coherence": coh,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "z_score": z_score,
        }

        print(f"\n  Method: {method}")
        print(f"    Phase coherence:    {coh:.6f}")
        print(f"    Random baseline:    {baseline_mean:.6f} +/- {baseline_std:.6f}")
        print(f"    Z-score:            {z_score:.2f}")
        print(f"    Verdict:            {'*** NON-RANDOM ***' if abs(z_score) > 3 else 'consistent with random'}")

    return results


# ---------------------------------------------------------------------------
# 2. Phase cluster analysis
# ---------------------------------------------------------------------------
def analyze_phase_clusters(embeds, tokenizer, n_clusters=8):
    """Cluster embeddings in phase space and inspect what lands together."""
    print("\n" + "=" * 60)
    print("2. PHASE CLUSTER ANALYSIS")
    print("=" * 60)

    z = embed_to_complex(embeds, method="hilbert")
    labels, centroids = phase_clusters(z, n_clusters=n_clusters)

    # Inspect each cluster: what tokens are in it?
    cluster_info = {}
    for c in range(n_clusters):
        mask = labels == c
        indices = np.where(mask)[0]
        count = len(indices)

        # Decode the tokens in this cluster (sample if large)
        sample_size = min(30, count)
        sampled_idx = np.random.choice(indices, size=sample_size, replace=False)
        tokens = [tokenizer.decode([i]).strip() for i in sampled_idx]

        cluster_info[c] = {
            "count": count,
            "sample_tokens": tokens,
        }

        print(f"\n  Cluster {c} ({count} tokens):")
        print(f"    Sample: {tokens[:15]}")

    # Measure cluster purity via simple heuristics:
    # - Are numbers clustered together?
    # - Are punctuation tokens clustered?
    # - Are function words (the, a, is) together?
    print("\n  --- Semantic category concentrations ---")
    categories = {
        "digits": [str(i) for i in range(10)],
        "punctuation": [".", ",", "!", "?", ";", ":", "-", "'", '"', "(", ")"],
        "articles": ["the", "a", "an", "The", "A", "An"],
        "pronouns": ["he", "she", "it", "they", "we", "I", "you"],
        "prepositions": ["in", "on", "at", "to", "of", "for", "with", "from"],
    }

    for cat_name, cat_words in categories.items():
        cat_ids = [token_id(tokenizer, w) for w in cat_words]
        cat_ids = [i for i in cat_ids if i is not None]
        if not cat_ids:
            continue
        cat_labels = labels[cat_ids]
        # Concentration = fraction in the most common cluster
        if len(cat_labels) > 0:
            from collections import Counter
            most_common_frac = Counter(cat_labels).most_common(1)[0][1] / len(cat_labels)
            expected_frac = 1.0 / n_clusters
            print(f"    {cat_name:15s}: concentration = {most_common_frac:.2f}  (random baseline = {expected_frac:.2f})")

    return labels, centroids, cluster_info


# ---------------------------------------------------------------------------
# 3. Interference score: related vs unrelated pairs
# ---------------------------------------------------------------------------
def analyze_interference(embeds, tokenizer):
    """Compare interference scores for semantically related vs unrelated word pairs."""
    print("\n" + "=" * 60)
    print("3. INTERFERENCE SCORE ANALYSIS")
    print("=" * 60)

    related_pairs = [
        ("king", "queen"),
        ("cat", "dog"),
        ("run", "walk"),
        ("hot", "cold"),
        ("big", "small"),
        ("man", "woman"),
        ("day", "night"),
        ("love", "hate"),
        ("up", "down"),
        ("fast", "slow"),
        ("good", "bad"),
        ("happy", "sad"),
        ("sun", "moon"),
        ("black", "white"),
        ("old", "young"),
    ]

    unrelated_pairs = [
        ("king", "banana"),
        ("cat", "democracy"),
        ("run", "purple"),
        ("hot", "paragraph"),
        ("big", "Tuesday"),
        ("man", "orbit"),
        ("day", "friction"),
        ("love", "concrete"),
        ("up", "molecule"),
        ("fast", "library"),
        ("good", "electron"),
        ("happy", "granite"),
        ("sun", "syntax"),
        ("black", "theorem"),
        ("old", "voltage"),
    ]

    z = embed_to_complex(embeds, method="hilbert")

    def score_pairs(pairs, label):
        scores = []
        cosine_sims = []
        for w1, w2 in pairs:
            id1 = token_id(tokenizer, w1)
            id2 = token_id(tokenizer, w2)
            if id1 is None or id2 is None:
                continue
            iscore = interference_score(z[id1], z[id2])
            csim = 1.0 - cosine_dist(embeds[id1], embeds[id2])
            scores.append(iscore)
            cosine_sims.append(csim)
        return scores, cosine_sims

    rel_iscores, rel_cosines = score_pairs(related_pairs, "related")
    unrel_iscores, unrel_cosines = score_pairs(unrelated_pairs, "unrelated")

    print(f"\n  Related pairs:")
    print(f"    Interference score:  mean = {np.mean(rel_iscores):.4f}, std = {np.std(rel_iscores):.4f}")
    print(f"    Cosine similarity:   mean = {np.mean(rel_cosines):.4f}, std = {np.std(rel_cosines):.4f}")

    print(f"\n  Unrelated pairs:")
    print(f"    Interference score:  mean = {np.mean(unrel_iscores):.4f}, std = {np.std(unrel_iscores):.4f}")
    print(f"    Cosine similarity:   mean = {np.mean(unrel_cosines):.4f}, std = {np.std(unrel_cosines):.4f}")

    # Statistical test
    from scipy.stats import mannwhitneyu
    if len(rel_iscores) > 1 and len(unrel_iscores) > 1:
        U, p = mannwhitneyu(rel_iscores, unrel_iscores, alternative="greater")
        print(f"\n  Mann-Whitney U test (related > unrelated):")
        print(f"    U = {U:.1f}, p = {p:.6f}")
        print(f"    Verdict: {'*** SIGNIFICANT ***' if p < 0.05 else 'not significant'}")

    # Also print individual pairs
    print(f"\n  {'Pair':<25s} {'Interference':>12s} {'Cosine':>10s}")
    print(f"  {'-'*25} {'-'*12} {'-'*10}")
    for (w1, w2), iscore, csim in zip(related_pairs, rel_iscores, rel_cosines):
        print(f"  {w1+'/'+w2:<25s} {iscore:>12.4f} {csim:>10.4f}")
    print()
    for (w1, w2), iscore, csim in zip(unrelated_pairs, unrel_iscores, unrel_cosines):
        print(f"  {w1+'/'+w2:<25s} {iscore:>12.4f} {csim:>10.4f}")

    return {
        "related_interference": rel_iscores,
        "unrelated_interference": unrel_iscores,
        "related_cosine": rel_cosines,
        "unrelated_cosine": unrel_cosines,
        "related_pairs": related_pairs,
        "unrelated_pairs": unrelated_pairs,
    }


# ---------------------------------------------------------------------------
# 4. Semantic entanglement (CHSH S-value)
# ---------------------------------------------------------------------------
def analyze_entanglement(embeds, tokenizer):
    """Compute CHSH-like S-values for semantically entangled vs random pairs."""
    print("\n" + "=" * 60)
    print("4. SEMANTIC ENTANGLEMENT (CHSH S-VALUE)")
    print("=" * 60)

    z = embed_to_complex(embeds, method="hilbert")

    # Semantically entangled pairs (antonyms/synonyms/strong associations)
    entangled_pairs = [
        ("king", "queen"), ("man", "woman"), ("boy", "girl"),
        ("hot", "cold"), ("up", "down"), ("big", "small"),
        ("good", "bad"), ("love", "hate"), ("day", "night"),
        ("black", "white"), ("fast", "slow"), ("old", "young"),
        ("cat", "dog"), ("sun", "moon"), ("happy", "sad"),
        ("run", "walk"), ("eat", "drink"), ("give", "take"),
        ("open", "close"), ("start", "stop"),
    ]

    # Random pairs
    rng = np.random.RandomState(42)
    vocab_size = embeds.shape[0]
    random_pairs_idx = [(rng.randint(0, vocab_size), rng.randint(0, vocab_size))
                        for _ in range(len(entangled_pairs))]

    def make_z_pairs(pair_list, use_ids=False):
        z_list = []
        for w1, w2 in pair_list:
            if use_ids:
                id1, id2 = w1, w2
            else:
                id1 = token_id(tokenizer, w1)
                id2 = token_id(tokenizer, w2)
            if id1 is None or id2 is None:
                continue
            z_list.append(np.stack([z[id1], z[id2]], axis=0))
        if not z_list:
            return None
        return np.stack(z_list, axis=0)  # [n_pairs, 2, dim]

    z_entangled = make_z_pairs(entangled_pairs)
    z_random = make_z_pairs(random_pairs_idx, use_ids=True)

    S_entangled = semantic_entanglement(z_entangled) if z_entangled is not None else float("nan")
    S_random = semantic_entanglement(z_random) if z_random is not None else float("nan")

    print(f"\n  Semantically entangled pairs ({len(entangled_pairs)} pairs):")
    print(f"    S-value = {S_entangled:.4f}")
    print(f"    Classical bound: |S| <= 2.0")
    print(f"    Tsirelson bound: |S| <= 2*sqrt(2) = {2*np.sqrt(2):.4f}")
    print(f"    Violation: {'*** YES ***' if S_entangled > 2.0 else 'No'}")

    print(f"\n  Random pairs ({len(random_pairs_idx)} pairs):")
    print(f"    S-value = {S_random:.4f}")
    print(f"    Violation: {'*** YES ***' if S_random > 2.0 else 'No'}")

    print(f"\n  Entangled/Random ratio: {S_entangled/S_random:.3f}" if S_random > 0 else "")

    # Also compute S-value for different semantic categories
    categories = {
        "antonyms": [("hot", "cold"), ("up", "down"), ("big", "small"),
                     ("good", "bad"), ("fast", "slow"), ("old", "young"),
                     ("happy", "sad"), ("black", "white"), ("love", "hate"),
                     ("day", "night")],
        "gender_pairs": [("king", "queen"), ("man", "woman"), ("boy", "girl"),
                         ("father", "mother"), ("son", "daughter"), ("he", "she"),
                         ("him", "her"), ("his", "hers"), ("brother", "sister"),
                         ("husband", "wife")],
        "actions": [("run", "walk"), ("eat", "drink"), ("give", "take"),
                    ("open", "close"), ("start", "stop"), ("push", "pull"),
                    ("buy", "sell"), ("win", "lose"), ("come", "go"),
                    ("sit", "stand")],
    }

    print(f"\n  --- S-values by semantic category ---")
    cat_results = {}
    for cat_name, pairs in categories.items():
        z_cat = make_z_pairs(pairs)
        if z_cat is not None and z_cat.shape[0] >= 3:
            S_cat = semantic_entanglement(z_cat)
            cat_results[cat_name] = S_cat
            violation = "VIOLATES" if S_cat > 2.0 else "classical"
            print(f"    {cat_name:20s}: S = {S_cat:.4f}  [{violation}]")

    return {
        "S_entangled": S_entangled,
        "S_random": S_random,
        "cat_results": cat_results,
    }


# ---------------------------------------------------------------------------
# 5. Phase distance vs cosine similarity
# ---------------------------------------------------------------------------
def analyze_phase_vs_cosine(embeds, tokenizer, n_sample_pairs=2000):
    """Plot phase-space distance against cosine similarity."""
    print("\n" + "=" * 60)
    print("5. PHASE DISTANCE vs COSINE SIMILARITY")
    print("=" * 60)

    z = embed_to_complex(embeds, method="hilbert")
    vocab_size = embeds.shape[0]

    rng = np.random.RandomState(123)
    idx1 = rng.randint(0, vocab_size, size=n_sample_pairs)
    idx2 = rng.randint(0, vocab_size, size=n_sample_pairs)

    cosines = []
    phase_dists = []
    interferences = []

    for i1, i2 in zip(idx1, idx2):
        if i1 == i2:
            continue
        # Cosine similarity in real space
        csim = 1.0 - cosine_dist(embeds[i1], embeds[i2])
        cosines.append(csim)

        # Phase distance: angular distance between phase vectors
        phi1 = np.angle(z[i1])
        phi2 = np.angle(z[i2])
        # Circular mean absolute difference
        phase_diff = np.abs(np.angle(np.exp(1j * (phi1 - phi2))))
        phase_dist = np.mean(phase_diff)
        phase_dists.append(phase_dist)

        # Interference score
        iscore = interference_score(z[i1], z[i2])
        interferences.append(iscore)

    cosines = np.array(cosines)
    phase_dists = np.array(phase_dists)
    interferences = np.array(interferences)

    # Correlations
    rho_phase_cos, p_phase_cos = spearmanr(phase_dists, cosines)
    rho_inter_cos, p_inter_cos = spearmanr(interferences, cosines)
    r_inter_cos, _ = pearsonr(interferences, cosines)

    print(f"\n  Sampled {len(cosines)} random token pairs")
    print(f"\n  Phase distance vs Cosine similarity:")
    print(f"    Spearman rho = {rho_phase_cos:.4f}  (p = {p_phase_cos:.2e})")
    print(f"\n  Interference score vs Cosine similarity:")
    print(f"    Spearman rho = {rho_inter_cos:.4f}  (p = {p_inter_cos:.2e})")
    print(f"    Pearson  r   = {r_inter_cos:.4f}")

    # Residual analysis: tokens where phase and cosine disagree most
    # Normalize both to [0,1] range for comparison
    cos_norm = (cosines - cosines.min()) / (cosines.max() - cosines.min() + 1e-10)
    inter_norm = (interferences - interferences.min()) / (interferences.max() - interferences.min() + 1e-10)
    residuals = inter_norm - cos_norm

    print(f"\n  Residual (interference - cosine, normalized):")
    print(f"    mean = {np.mean(residuals):.4f}, std = {np.std(residuals):.4f}")
    print(f"    Fraction where phase adds info (|residual| > 0.3): "
          f"{np.mean(np.abs(residuals) > 0.3):.2%}")

    return {
        "cosines": cosines,
        "phase_dists": phase_dists,
        "interferences": interferences,
        "idx1": idx1,
        "idx2": idx2,
    }


# ---------------------------------------------------------------------------
# 6. Attention Q/K phase structure
# ---------------------------------------------------------------------------
def analyze_attention_phase(model_obj):
    """Project attention Q, K matrices into complex space and measure phase structure."""
    print("\n" + "=" * 60)
    print("6. ATTENTION HEAD PHASE STRUCTURE")
    print("=" * 60)

    results = {}

    # GPT-2 has 12 layers, each with Q, K, V packed into c_attn
    # c_attn.weight is [768, 2304] where 2304 = 3 * 768 (Q, K, V)
    for layer_idx in [0, 3, 6, 9, 11]:
        attn = model_obj.transformer.h[layer_idx].attn
        W = attn.c_attn.weight.detach().numpy()  # [768, 2304]
        d_model = W.shape[0]  # 768

        # Split into Q, K, V
        W_q = W[:, :d_model]           # [768, 768]
        W_k = W[:, d_model:2*d_model]  # [768, 768]

        # Project Q and K into complex space
        z_q = embed_to_complex(W_q, method="hilbert")
        z_k = embed_to_complex(W_k, method="hilbert")

        coh_q = phase_coherence(z_q)
        coh_k = phase_coherence(z_k)

        # Cross-coherence: do Q and K phases align?
        # Take the phase difference and check its coherence
        phase_diff = np.angle(z_q) - np.angle(z_k)
        cross_R = np.abs(np.mean(np.exp(1j * phase_diff), axis=0))
        cross_coh = float(np.mean(cross_R))

        # Interference between Q and K columns (attention head alignment)
        n_heads = 12
        head_dim = d_model // n_heads  # 64
        head_interferences = []
        for h in range(n_heads):
            q_head = z_q[:, h*head_dim//2:(h+1)*head_dim//2]
            k_head = z_k[:, h*head_dim//2:(h+1)*head_dim//2]
            # Mean interference across the rows (input dims)
            head_scores = []
            for row in range(min(50, q_head.shape[0])):
                head_scores.append(interference_score(q_head[row], k_head[row]))
            head_interferences.append(np.mean(head_scores))

        results[layer_idx] = {
            "coh_q": coh_q,
            "coh_k": coh_k,
            "cross_coh": cross_coh,
            "head_interferences": head_interferences,
        }

        print(f"\n  Layer {layer_idx}:")
        print(f"    Q phase coherence:     {coh_q:.6f}")
        print(f"    K phase coherence:     {coh_k:.6f}")
        print(f"    Q-K cross coherence:   {cross_coh:.6f}")
        print(f"    Head Q-K interference: {[f'{x:.3f}' for x in head_interferences]}")
        print(f"    Mean head interference: {np.mean(head_interferences):.4f}")

    # Compare with random matrices
    print(f"\n  --- Random matrix baselines ---")
    for _ in range(3):
        W_rand = np.random.randn(768, 768)
        z_rand = embed_to_complex(W_rand, method="hilbert")
        coh_rand = phase_coherence(z_rand)
        print(f"    Random [768,768] coherence: {coh_rand:.6f}")

    return results


# ---------------------------------------------------------------------------
# 7. Plotting everything
# ---------------------------------------------------------------------------
def make_plots(
    coherence_results,
    interference_results,
    entanglement_results,
    phase_cosine_results,
    attention_results,
    save_path,
):
    """Create a comprehensive figure with all experiment results."""
    print(f"\nGenerating plots -> {save_path}")

    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    # --- Panel A: Phase coherence comparison ---
    ax_a = fig.add_subplot(gs[0, 0])
    methods = list(coherence_results.keys())
    real_cohs = [coherence_results[m]["coherence"] for m in methods]
    base_cohs = [coherence_results[m]["baseline_mean"] for m in methods]
    base_stds = [coherence_results[m]["baseline_std"] for m in methods]

    x = np.arange(len(methods))
    ax_a.bar(x - 0.15, real_cohs, 0.3, label="GPT-2", color="#4C72B0", alpha=0.9)
    ax_a.bar(x + 0.15, base_cohs, 0.3, yerr=base_stds, label="Random", color="#DD8452", alpha=0.7)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(methods, fontsize=9)
    ax_a.set_ylabel("Phase Coherence")
    ax_a.set_title("A. Phase Coherence:\nGPT-2 vs Shuffled Baseline")
    ax_a.legend(fontsize=8)
    for i, (r, b) in enumerate(zip(real_cohs, base_cohs)):
        z = coherence_results[methods[i]]["z_score"]
        ax_a.text(i, max(r, b) + 0.005, f"z={z:.1f}", ha="center", fontsize=8,
                  color="red" if abs(z) > 3 else "gray")

    # --- Panel B: Interference scores comparison ---
    ax_b = fig.add_subplot(gs[0, 1])
    rel_scores = interference_results["related_interference"]
    unrel_scores = interference_results["unrelated_interference"]
    bp = ax_b.boxplot([rel_scores, unrel_scores],
                      labels=["Semantically\nRelated", "Unrelated"],
                      patch_artist=True,
                      medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor("#55A868")
    bp["boxes"][1].set_facecolor("#C44E52")
    ax_b.set_ylabel("Interference Score")
    ax_b.set_title("B. Interference Score:\nRelated vs Unrelated Pairs")
    ax_b.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # --- Panel C: Individual pair interference scores ---
    ax_c = fig.add_subplot(gs[0, 2:4])
    n_rel = len(rel_scores)
    n_unrel = len(unrel_scores)
    pair_labels_rel = [f"{w1}/{w2}" for w1, w2 in interference_results["related_pairs"][:n_rel]]
    pair_labels_unrel = [f"{w1}/{w2}" for w1, w2 in interference_results["unrelated_pairs"][:n_unrel]]
    all_labels = pair_labels_rel + pair_labels_unrel
    all_scores = list(rel_scores) + list(unrel_scores)
    colors = ["#55A868"] * n_rel + ["#C44E52"] * n_unrel

    bars = ax_c.barh(range(len(all_scores)), all_scores, color=colors, alpha=0.8)
    ax_c.set_yticks(range(len(all_labels)))
    ax_c.set_yticklabels(all_labels, fontsize=7)
    ax_c.set_xlabel("Interference Score")
    ax_c.set_title("C. Per-Pair Interference Scores\n(green=related, red=unrelated)")
    ax_c.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax_c.invert_yaxis()

    # --- Panel D: CHSH S-values ---
    ax_d = fig.add_subplot(gs[1, 0])
    s_vals = [entanglement_results["S_entangled"], entanglement_results["S_random"]]
    s_labels = ["Semantic\nPairs", "Random\nPairs"]
    bar_colors = ["#4C72B0" if s > 2.0 else "#DD8452" for s in s_vals]
    ax_d.bar(s_labels, s_vals, color=bar_colors, alpha=0.9, edgecolor="black")
    ax_d.axhline(y=2.0, color="red", linestyle="--", linewidth=2, label="Classical bound (S=2)")
    ax_d.axhline(y=2*np.sqrt(2), color="green", linestyle=":", linewidth=1.5,
                 label=f"Tsirelson ({2*np.sqrt(2):.2f})")
    ax_d.set_ylabel("CHSH S-value")
    ax_d.set_title("D. CHSH S-Value:\nSemantic Entanglement")
    ax_d.legend(fontsize=7, loc="upper right")

    # --- Panel E: S-values by category ---
    ax_e = fig.add_subplot(gs[1, 1])
    cat_results = entanglement_results.get("cat_results", {})
    if cat_results:
        cat_names = list(cat_results.keys())
        cat_svals = [cat_results[c] for c in cat_names]
        bar_colors_cat = ["#4C72B0" if s > 2.0 else "#DD8452" for s in cat_svals]
        ax_e.barh(cat_names, cat_svals, color=bar_colors_cat, alpha=0.9, edgecolor="black")
        ax_e.axvline(x=2.0, color="red", linestyle="--", linewidth=2, label="Classical bound")
        ax_e.set_xlabel("S-value")
        ax_e.set_title("E. S-Value by Semantic\nCategory")
        ax_e.legend(fontsize=7)
    else:
        ax_e.text(0.5, 0.5, "No category data", ha="center", va="center", transform=ax_e.transAxes)

    # --- Panel F: Phase distance vs Cosine similarity scatter ---
    ax_f = fig.add_subplot(gs[1, 2])
    cosines = phase_cosine_results["cosines"]
    phase_dists = phase_cosine_results["phase_dists"]
    ax_f.scatter(cosines, phase_dists, alpha=0.1, s=3, c="#4C72B0")
    ax_f.set_xlabel("Cosine Similarity (real space)")
    ax_f.set_ylabel("Phase Distance (complex space)")
    rho, _ = spearmanr(cosines, phase_dists)
    ax_f.set_title(f"F. Phase Distance vs Cosine\nSpearman rho = {rho:.3f}")

    # --- Panel G: Interference vs Cosine scatter ---
    ax_g = fig.add_subplot(gs[1, 3])
    interferences = phase_cosine_results["interferences"]
    ax_g.scatter(cosines, interferences, alpha=0.1, s=3, c="#55A868")
    ax_g.set_xlabel("Cosine Similarity (real space)")
    ax_g.set_ylabel("Interference Score (complex space)")
    rho2, _ = spearmanr(cosines, interferences)
    r2, _ = pearsonr(cosines, interferences)
    ax_g.set_title(f"G. Interference vs Cosine\nPearson r = {r2:.3f}")
    # Add diagonal reference
    lims = [min(ax_g.get_xlim()[0], ax_g.get_ylim()[0]),
            max(ax_g.get_xlim()[1], ax_g.get_ylim()[1])]
    ax_g.plot(lims, lims, "r--", alpha=0.3, label="y=x")

    # --- Panel H: Attention head phase coherence across layers ---
    ax_h = fig.add_subplot(gs[2, 0:2])
    layers = sorted(attention_results.keys())
    coh_q_vals = [attention_results[l]["coh_q"] for l in layers]
    coh_k_vals = [attention_results[l]["coh_k"] for l in layers]
    cross_vals = [attention_results[l]["cross_coh"] for l in layers]
    x_layers = np.arange(len(layers))
    w = 0.25
    ax_h.bar(x_layers - w, coh_q_vals, w, label="Q coherence", color="#4C72B0", alpha=0.9)
    ax_h.bar(x_layers, coh_k_vals, w, label="K coherence", color="#55A868", alpha=0.9)
    ax_h.bar(x_layers + w, cross_vals, w, label="Q-K cross", color="#C44E52", alpha=0.9)
    ax_h.set_xticks(x_layers)
    ax_h.set_xticklabels([f"Layer {l}" for l in layers], fontsize=9)
    ax_h.set_ylabel("Phase Coherence")
    ax_h.set_title("H. Attention Head Phase Structure Across Layers")
    ax_h.legend(fontsize=8)

    # --- Panel I: Per-head Q-K interference by layer ---
    ax_i = fig.add_subplot(gs[2, 2:4])
    for l in layers:
        head_intf = attention_results[l]["head_interferences"]
        ax_i.plot(range(len(head_intf)), head_intf, "o-", label=f"Layer {l}", alpha=0.8, markersize=4)
    ax_i.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax_i.set_xlabel("Attention Head Index")
    ax_i.set_ylabel("Q-K Interference Score")
    ax_i.set_title("I. Per-Head Q-K Interference by Layer")
    ax_i.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "Probing GPT-2 Embeddings for Latent Complex/Phase Structure\n"
        "Hilbert Transform Projection: R^768 -> C^384",
        fontsize=16, fontweight="bold", y=1.01,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved to {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("PROBING GPT-2 EMBEDDINGS FOR LATENT PHASE STRUCTURE")
    print("Projecting R^768 -> C^384 via Hilbert Transform")
    print("=" * 60)

    embeds, tokenizer, model_obj = load_gpt2()

    # Run all analyses
    coherence_results = analyze_phase_coherence(embeds)
    labels, centroids, cluster_info = analyze_phase_clusters(embeds, tokenizer)
    interference_results = analyze_interference(embeds, tokenizer)
    entanglement_results = analyze_entanglement(embeds, tokenizer)
    phase_cosine_results = analyze_phase_vs_cosine(embeds, tokenizer)
    attention_results = analyze_attention_phase(model_obj)

    # Generate the combined figure
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "probe_results.png",
    )
    make_plots(
        coherence_results,
        interference_results,
        entanglement_results,
        phase_cosine_results,
        attention_results,
        save_path,
    )

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Phase coherence (Hilbert):  {coherence_results['hilbert']['coherence']:.6f}")
    print(f"    z-score vs random:         {coherence_results['hilbert']['z_score']:.2f}")
    print(f"\n  Interference (related):      {np.mean(interference_results['related_interference']):.4f}")
    print(f"  Interference (unrelated):    {np.mean(interference_results['unrelated_interference']):.4f}")
    print(f"\n  CHSH S-value (semantic):     {entanglement_results['S_entangled']:.4f}")
    print(f"  CHSH S-value (random):       {entanglement_results['S_random']:.4f}")
    print(f"  Classical bound:             2.0000")
    print(f"  Tsirelson bound:             {2*np.sqrt(2):.4f}")

    phase_cos_rho, _ = spearmanr(
        phase_cosine_results["cosines"],
        phase_cosine_results["interferences"],
    )
    print(f"\n  Interference-Cosine corr:    Spearman rho = {phase_cos_rho:.4f}")

    # The big question
    S = entanglement_results["S_entangled"]
    coh_z = coherence_results["hilbert"]["z_score"]
    print("\n  *** VERDICT ***")
    if coh_z > 3:
        print("  Phase structure is NON-RANDOM (z > 3).")
        print("  GPT-2 embeddings carry latent phase information when projected")
        print("  to complex space. The optimization landscape appears to select")
        print("  for wave-like interference patterns, even in a purely real-valued")
        print("  architecture.")
    else:
        print("  Phase structure is CONSISTENT WITH RANDOM.")
        print("  The Hilbert projection does not reveal hidden complex structure")
        print("  in GPT-2's embedding space.")

    if S > 2.0:
        print(f"\n  CHSH violation detected (S = {S:.4f} > 2).")
        print("  Semantically related embeddings show 'entanglement-like' phase")
        print("  correlations that exceed the classical bound. This is analogous")
        print("  to the Bell test results from ket-nlp, but operating on the")
        print("  embedding geometry rather than LLM response patterns.")
    else:
        print(f"\n  No CHSH violation (S = {S:.4f} <= 2).")
        print("  Phase correlations between semantic pairs stay within classical bounds.")

    print(f"\n  Plots saved to: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
