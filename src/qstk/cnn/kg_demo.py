"""
ComplexKG Demo: complex-valued knowledge graph vs real-valued baseline.

Builds a small geographic knowledge graph (countries, cities, languages,
continents) and trains both a ComplexKG (RotatE-style, phase interference)
and a RealKGBaseline (TransE). Evaluates link prediction, multi-hop
reasoning, and phase structure analysis.

The key hypothesis: complex embeddings should learn cleaner structure
because relations are rotations (a group operation), while real TransE
uses translations (which don't compose as cleanly). Multi-hop queries
should especially benefit: composing rotations is exact multiplication,
while composing translations accumulates noise.

Run:
    python -m qstk.cnn.kg_demo

Saves plots to /home/caug/npcww/qstk/results/kg_demo.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
import time

# Allow running from the source tree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from qstk.cnn.kg import (
    ComplexKG,
    RealKGBaseline,
    hits_at_k,
    mean_rank,
    mrr,
    compare_to_baseline,
)


# ---------------------------------------------------------------------------
# Knowledge graph data: geography
# ---------------------------------------------------------------------------

def build_geography_kg():
    """Build a small geographic knowledge graph.

    Entities: cities, countries, continents, languages
    Relations: capital_of, located_in, speaks, borders, part_of

    Returns:
        entity_names: list of strings
        relation_names: list of strings
        triples: np.ndarray [N, 3] of (h_idx, r_idx, t_idx)
        entity_types: dict mapping entity name -> type
    """
    # --- Entities ---
    entities = [
        # Cities (0-11)
        "Paris", "Berlin", "Madrid", "Rome", "London",
        "Tokyo", "Beijing", "Moscow", "Cairo", "Brasilia",
        "Ottawa", "Canberra",
        # Countries (12-23)
        "France", "Germany", "Spain", "Italy", "UK",
        "Japan", "China", "Russia", "Egypt", "Brazil",
        "Canada", "Australia",
        # Continents (24-29)
        "Europe", "Asia", "Africa", "South_America",
        "North_America", "Oceania",
        # Languages (30-39)
        "French", "German", "Spanish", "Italian", "English",
        "Japanese", "Mandarin", "Russian", "Arabic", "Portuguese",
    ]

    entity_types = {}
    for i, e in enumerate(entities):
        if i < 12:
            entity_types[e] = "city"
        elif i < 24:
            entity_types[e] = "country"
        elif i < 30:
            entity_types[e] = "continent"
        else:
            entity_types[e] = "language"

    eidx = {name: i for i, name in enumerate(entities)}

    # --- Relations ---
    relations = [
        "capital_of",    # 0: city -> country
        "located_in",    # 1: country -> continent
        "speaks",        # 2: country -> language
        "borders",       # 3: country -> country
        "part_of",       # 4: continent -> continent (or region grouping)
        "city_in",       # 5: city -> continent (transitive)
    ]
    ridx = {name: i for i, name in enumerate(relations)}

    # --- Triples ---
    raw_triples = [
        # capital_of: city -> country
        ("Paris", "capital_of", "France"),
        ("Berlin", "capital_of", "Germany"),
        ("Madrid", "capital_of", "Spain"),
        ("Rome", "capital_of", "Italy"),
        ("London", "capital_of", "UK"),
        ("Tokyo", "capital_of", "Japan"),
        ("Beijing", "capital_of", "China"),
        ("Moscow", "capital_of", "Russia"),
        ("Cairo", "capital_of", "Egypt"),
        ("Brasilia", "capital_of", "Brazil"),
        ("Ottawa", "capital_of", "Canada"),
        ("Canberra", "capital_of", "Australia"),

        # located_in: country -> continent
        ("France", "located_in", "Europe"),
        ("Germany", "located_in", "Europe"),
        ("Spain", "located_in", "Europe"),
        ("Italy", "located_in", "Europe"),
        ("UK", "located_in", "Europe"),
        ("Japan", "located_in", "Asia"),
        ("China", "located_in", "Asia"),
        ("Russia", "located_in", "Asia"),
        ("Egypt", "located_in", "Africa"),
        ("Brazil", "located_in", "South_America"),
        ("Canada", "located_in", "North_America"),
        ("Australia", "located_in", "Oceania"),

        # speaks: country -> language
        ("France", "speaks", "French"),
        ("Germany", "speaks", "German"),
        ("Spain", "speaks", "Spanish"),
        ("Italy", "speaks", "Italian"),
        ("UK", "speaks", "English"),
        ("Japan", "speaks", "Japanese"),
        ("China", "speaks", "Mandarin"),
        ("Russia", "speaks", "Russian"),
        ("Egypt", "speaks", "Arabic"),
        ("Brazil", "speaks", "Portuguese"),
        ("Canada", "speaks", "English"),
        ("Canada", "speaks", "French"),
        ("Australia", "speaks", "English"),

        # borders: country -> country
        ("France", "borders", "Germany"),
        ("France", "borders", "Spain"),
        ("France", "borders", "Italy"),
        ("Germany", "borders", "France"),
        ("Germany", "borders", "Italy"),   # not exact, but close enough
        ("Spain", "borders", "France"),
        ("Italy", "borders", "France"),
        ("Russia", "borders", "China"),
        ("China", "borders", "Russia"),
        ("Egypt", "borders", "Africa"),    # continent-level neighbor
        ("Canada", "borders", "UK"),       # not really, but for graph density
        ("Japan", "borders", "China"),     # sea neighbor

        # city_in (transitive shortcut): city -> continent
        ("Paris", "city_in", "Europe"),
        ("Berlin", "city_in", "Europe"),
        ("Madrid", "city_in", "Europe"),
        ("Rome", "city_in", "Europe"),
        ("London", "city_in", "Europe"),
        ("Tokyo", "city_in", "Asia"),
        ("Beijing", "city_in", "Asia"),
        ("Moscow", "city_in", "Asia"),
        ("Cairo", "city_in", "Africa"),
        ("Brasilia", "city_in", "South_America"),
        ("Ottawa", "city_in", "North_America"),
        ("Canberra", "city_in", "Oceania"),
    ]

    triples = np.array([
        [eidx[h], ridx[r], eidx[t]] for h, r, t in raw_triples
    ], dtype=np.int64)

    return entities, relations, triples, entity_types, eidx, ridx


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, triples, n_epochs=300, lr=0.05, neg_samples=15,
                verbose=True, label="model"):
    """Train a model (ComplexKG or RealKGBaseline) on triples.

    Returns list of loss values per epoch.
    """
    losses = []
    np.random.seed(42)

    t0 = time.time()
    for epoch in range(n_epochs):
        # Shuffle triples each epoch
        perm = np.random.permutation(len(triples))
        batch = triples[perm]
        loss = model.train_step(batch, neg_samples=neg_samples, lr=lr)
        losses.append(loss)

        if verbose and (epoch + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] epoch {epoch+1:4d} | loss {loss:.4f} | {elapsed:.1f}s")

    return losses


# ---------------------------------------------------------------------------
# Multi-hop tests
# ---------------------------------------------------------------------------

def test_multi_hop(model, entities, eidx, ridx):
    """Test multi-hop reasoning via PAM interference.

    Query: Paris -> capital_of -> ? -> located_in -> ?
    Expected chain: Paris -> France -> Europe
    """
    print("\n--- Multi-hop reasoning (PAM interference) ---")

    tests = [
        {
            'name': 'Paris -> capital_of -> ? -> located_in -> ?',
            'entity': 'Paris',
            'path': ['capital_of', 'located_in'],
            'expected': 'Europe',
        },
        {
            'name': 'Tokyo -> capital_of -> ? -> speaks -> ?',
            'entity': 'Tokyo',
            'path': ['capital_of', 'speaks'],
            'expected': 'Japanese',
        },
        {
            'name': 'Berlin -> capital_of -> ? -> borders -> ?',
            'entity': 'Berlin',
            'path': ['capital_of', 'borders'],
            'expected': 'France',
        },
    ]

    results = []
    for test in tests:
        e_idx = eidx[test['entity']]
        r_path = [ridx[r] for r in test['path']]
        expected_idx = eidx[test['expected']]

        top_idx, top_scores = model.multi_hop_search(e_idx, r_path, top_k=5)

        found = expected_idx in top_idx
        if found:
            rank = int(np.where(top_idx == expected_idx)[0][0]) + 1
        else:
            rank = -1

        print(f"\n  {test['name']}")
        print(f"  Expected: {test['expected']} (idx {expected_idx})")
        print(f"  Top-5 results:")
        for j, (idx, sc) in enumerate(zip(top_idx, top_scores)):
            marker = " <--" if idx == expected_idx else ""
            print(f"    {j+1}. {entities[idx]:20s}  score={sc:.4f}{marker}")
        print(f"  Found in top-5: {found} (rank={rank})")

        results.append({
            'name': test['name'],
            'found': found,
            'rank': rank,
        })

    return results


# ---------------------------------------------------------------------------
# Single-hop search tests
# ---------------------------------------------------------------------------

def test_single_hop(model, entities, eidx, ridx):
    """Test single-hop PAM retrieval."""
    print("\n--- Single-hop PAM search ---")

    tests = [
        ('France', 'capital_of', 'Paris'),     # inverse: what city is capital of France?
        ('Germany', 'speaks', 'German'),
        ('Japan', 'located_in', 'Asia'),
    ]

    # Note: The PAM encodes (h, r) -> t, so to find "capital of France"
    # we search with France as entity and capital_of as relation.
    # But our triples have (Paris, capital_of, France).
    # So let's search forward: (Paris, capital_of, ?) -> France
    fwd_tests = [
        ('Paris', 'capital_of', 'France'),
        ('Germany', 'speaks', 'German'),
        ('Japan', 'located_in', 'Asia'),
        ('France', 'borders', 'Germany'),
        ('Cairo', 'capital_of', 'Egypt'),
    ]

    for h_name, r_name, expected_t in fwd_tests:
        top_idx, top_scores = model.search(eidx[h_name], ridx[r_name], top_k=5)
        expected_idx = eidx[expected_t]
        found = expected_idx in top_idx
        rank = int(np.where(top_idx == expected_idx)[0][0]) + 1 if found else -1
        print(f"  ({h_name}, {r_name}, ?) -> top-1: {entities[top_idx[0]]}, "
              f"expected: {expected_t}, found@5: {found}, rank: {rank}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ComplexKG Demo: Phase-Interference Knowledge Graph")
    print("=" * 70)

    # Build data
    entities, relations, triples, entity_types, eidx, ridx = build_geography_kg()
    n_entities = len(entities)
    n_relations = len(relations)

    print(f"\nKnowledge graph: {n_entities} entities, {n_relations} relations, "
          f"{len(triples)} triples")

    # Train/test split (80/20)
    np.random.seed(123)
    perm = np.random.permutation(len(triples))
    split = int(0.8 * len(triples))
    train_triples = triples[perm[:split]]
    test_triples = triples[perm[split:]]
    print(f"Train: {len(train_triples)}, Test: {len(test_triples)}")

    # --- Complex model ---
    DIM = 32
    print(f"\n{'='*70}")
    print(f"Training ComplexKG (dim={DIM}, RotatE scoring, phase interference)")
    print(f"{'='*70}")

    np.random.seed(42)
    complex_model = ComplexKG(
        n_entities=n_entities,
        n_relations=n_relations,
        dim=DIM,
        n_heads=4,
        d_head=DIM,
        margin=2.0,
        score_mode='rotate',
        gamma=1.0,
    )
    complex_losses = train_model(
        complex_model, train_triples,
        n_epochs=400, lr=0.03, neg_samples=20,
        label="ComplexKG"
    )

    # --- Real baseline ---
    print(f"\n{'='*70}")
    print(f"Training RealKGBaseline (dim={DIM}, TransE scoring)")
    print(f"{'='*70}")

    np.random.seed(42)
    real_model = RealKGBaseline(
        n_entities=n_entities,
        n_relations=n_relations,
        dim=DIM,
        margin=2.0,
    )
    real_losses = train_model(
        real_model, train_triples,
        n_epochs=400, lr=0.03, neg_samples=20,
        label="RealKG  "
    )

    # --- Evaluate ---
    print(f"\n{'='*70}")
    print("Link Prediction Evaluation")
    print(f"{'='*70}")

    results = compare_to_baseline(complex_model, real_model, test_triples)
    for name, metrics in results.items():
        print(f"\n  {name.upper()}:")
        for k, v in metrics.items():
            print(f"    {k:12s}: {v:.4f}")

    # Also evaluate on ALL triples (train+test) for completeness
    print(f"\n  --- On all triples ---")
    results_all = compare_to_baseline(complex_model, real_model, triples)
    for name, metrics in results_all.items():
        print(f"\n  {name.upper()} (all):")
        for k, v in metrics.items():
            print(f"    {k:12s}: {v:.4f}")

    # --- Encode graph into PAM and test search ---
    print(f"\n{'='*70}")
    print("Encoding graph into PAM memory and testing search")
    print(f"{'='*70}")

    complex_model.encode_graph(triples)
    test_single_hop(complex_model, entities, eidx, ridx)
    multi_hop_results = test_multi_hop(complex_model, entities, eidx, ridx)

    # --- Phase analysis ---
    print(f"\n{'='*70}")
    print("Phase Structure Analysis")
    print(f"{'='*70}")

    analysis = complex_model.phase_analysis()

    # Group entities by type and show mean phase
    for etype in ['city', 'country', 'continent', 'language']:
        idxs = [eidx[e] for e in entities if entity_types[e] == etype]
        phases = analysis['entity_phase_mean'][idxs]
        mags = np.mean(analysis['entity_magnitudes'][idxs], axis=-1)
        print(f"\n  {etype:12s}: mean_phase={np.mean(phases):+.3f} "
              f"std_phase={np.std(phases):.3f} "
              f"mean_mag={np.mean(mags):.3f}")

    print("\n  Relation phases (should be distinct rotations):")
    for i, rname in enumerate(relations):
        phase = analysis['relation_phases'][i]
        mag = analysis['relation_magnitudes'][i]
        print(f"    {rname:15s}: mean_phase={np.mean(phase):+.3f} "
              f"std={np.std(phase):.3f} "
              f"mean_mag={np.mean(mag):.4f}")

    print(f"\n  PAM state magnitudes: {analysis['pam_state_magnitude']}")

    # --- Entity comparisons ---
    print(f"\n{'='*70}")
    print("Entity Comparisons (interference analysis)")
    print(f"{'='*70}")

    comparisons = [
        ("Paris", "Berlin"),        # both European capitals
        ("Paris", "Tokyo"),         # different continents
        ("France", "Germany"),      # neighboring countries
        ("France", "Japan"),        # distant countries
        ("French", "Spanish"),      # both Romance languages
        ("French", "Japanese"),     # very different languages
        ("Europe", "Asia"),         # neighboring continents
    ]

    for e1, e2 in comparisons:
        comp = complex_model.compare_embeddings(eidx[e1], eidx[e2])
        print(f"  {e1:12s} vs {e2:12s}: "
              f"interference={comp['interference_score']:+.4f}  "
              f"phase_dist={comp['phase_distance']:.3f}  "
              f"mag_ratio={comp['magnitude_ratio']:.3f}")

    # --- Plot ---
    print(f"\n{'='*70}")
    print("Generating plots...")
    print(f"{'='*70}")

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    # 1. Training loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(complex_losses, label="ComplexKG (RotatE)", color="#2196F3", linewidth=1.5)
    ax1.semilogy(real_losses, label="RealKG (TransE)", color="#FF5722", linewidth=1.5, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (log)")
    ax1.set_title("Training Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Evaluation comparison bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    metrics_names = ['hits_at_1', 'hits_at_3', 'hits_at_10', 'mrr']
    x_pos = np.arange(len(metrics_names))
    width = 0.35
    complex_vals = [results['complex'][m] for m in metrics_names]
    real_vals = [results['real'][m] for m in metrics_names]
    ax2.bar(x_pos - width/2, complex_vals, width, label="Complex", color="#2196F3", alpha=0.8)
    ax2.bar(x_pos + width/2, real_vals, width, label="Real", color="#FF5722", alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["H@1", "H@3", "H@10", "MRR"], fontsize=9)
    ax2.set_ylabel("Score")
    ax2.set_title("Link Prediction (test)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Entity phase distribution (polar plot)
    ax3 = fig.add_subplot(gs[0, 2], projection='polar')
    type_colors = {'city': '#4CAF50', 'country': '#2196F3',
                   'continent': '#FF9800', 'language': '#9C27B0'}
    for etype, color in type_colors.items():
        idxs = [eidx[e] for e in entities if entity_types[e] == etype]
        phases = analysis['entity_phase_mean'][idxs]
        mags = np.mean(analysis['entity_magnitudes'][idxs], axis=-1)
        ax3.scatter(phases, mags, c=color, label=etype, s=40, alpha=0.8, zorder=3)
    ax3.set_title("Entity Phase Distribution", pad=15, fontsize=10)
    ax3.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # 4. Relation phase vectors (polar plot)
    ax4 = fig.add_subplot(gs[0, 3], projection='polar')
    rel_colors = plt.cm.Set1(np.linspace(0, 1, n_relations))
    for i, rname in enumerate(relations):
        phase = np.mean(analysis['relation_phases'][i])
        mag = np.mean(analysis['relation_magnitudes'][i])
        ax4.annotate("", xy=(phase, mag), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color=rel_colors[i], lw=2))
        ax4.scatter([phase], [mag], c=[rel_colors[i]], s=60, zorder=5)
        ax4.annotate(rname, (phase, mag * 1.1), fontsize=6, ha='center')
    ax4.set_title("Relation Phase Rotations", pad=15, fontsize=10)

    # 5. Entity embeddings in complex plane (first 2 dims)
    ax5 = fig.add_subplot(gs[1, 0])
    ent_embs = complex_model.embeddings.params['entities']
    for etype, color in type_colors.items():
        idxs = [eidx[e] for e in entities if entity_types[e] == etype]
        z = ent_embs[idxs, 0]  # first complex dimension
        ax5.scatter(z.real, z.imag, c=color, label=etype, s=50, alpha=0.8, zorder=3)
        for idx in idxs:
            ax5.annotate(entities[idx], (ent_embs[idx, 0].real, ent_embs[idx, 0].imag),
                        fontsize=5, alpha=0.7)
    ax5.set_xlabel("Re(dim 0)")
    ax5.set_ylabel("Im(dim 0)")
    ax5.set_title("Entity Embeddings (dim 0)")
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='k', linewidth=0.5)
    ax5.axvline(x=0, color='k', linewidth=0.5)

    # 6. PCA of entity phases (2D projection)
    ax6 = fig.add_subplot(gs[1, 1])
    ent_phases_full = analysis['entity_phases']  # [n_entities, dim]
    # Simple 2D PCA on phases
    phase_centered = ent_phases_full - ent_phases_full.mean(axis=0)
    U, S_svd, Vt = np.linalg.svd(phase_centered, full_matrices=False)
    phase_2d = U[:, :2] * S_svd[:2]

    for etype, color in type_colors.items():
        idxs = [eidx[e] for e in entities if entity_types[e] == etype]
        ax6.scatter(phase_2d[idxs, 0], phase_2d[idxs, 1], c=color,
                   label=etype, s=50, alpha=0.8, zorder=3)
        for idx in idxs:
            ax6.annotate(entities[idx], (phase_2d[idx, 0], phase_2d[idx, 1]),
                        fontsize=5, alpha=0.7)
    ax6.set_xlabel("Phase PC1")
    ax6.set_ylabel("Phase PC2")
    ax6.set_title("Phase PCA (entity clustering)")
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)

    # 7. Interference matrix: entity-entity similarity
    ax7 = fig.add_subplot(gs[1, 2:4])
    # Pick a subset of interesting entities
    subset_names = [
        "Paris", "Berlin", "Tokyo", "Cairo",
        "France", "Germany", "Japan", "Egypt",
        "Europe", "Asia", "Africa",
        "French", "German", "Japanese", "Arabic",
    ]
    subset_idx = [eidx[n] for n in subset_names]
    n_sub = len(subset_idx)
    interference_matrix = np.zeros((n_sub, n_sub))
    for i in range(n_sub):
        for j in range(n_sub):
            comp = complex_model.compare_embeddings(subset_idx[i], subset_idx[j])
            interference_matrix[i, j] = comp['interference_score']

    im = ax7.imshow(interference_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax7.set_xticks(range(n_sub))
    ax7.set_xticklabels(subset_names, rotation=45, ha='right', fontsize=7)
    ax7.set_yticks(range(n_sub))
    ax7.set_yticklabels(subset_names, fontsize=7)
    ax7.set_title("Interference Matrix (entity similarity)")
    plt.colorbar(im, ax=ax7, fraction=0.046, pad=0.04)

    # 8. PAM state magnitude spectrum
    ax8 = fig.add_subplot(gs[2, 0])
    for head_i in range(complex_model.memory.n_heads):
        sv = np.linalg.svd(complex_model.memory.S[head_i], compute_uv=False)
        ax8.semilogy(sv, label=f"Head {head_i}", alpha=0.8)
    ax8.set_xlabel("Singular value index")
    ax8.set_ylabel("Singular value (log)")
    ax8.set_title("PAM State Spectrum")
    ax8.legend(fontsize=7)
    ax8.grid(True, alpha=0.3)

    # 9. Relation rotation visualization
    ax9 = fig.add_subplot(gs[2, 1])
    rel_embs = complex_model.embeddings.params['relations']
    for i, rname in enumerate(relations):
        phases = np.angle(rel_embs[i])
        ax9.hist(phases, bins=20, alpha=0.5, label=rname, density=True)
    ax9.set_xlabel("Phase angle (rad)")
    ax9.set_ylabel("Density")
    ax9.set_title("Relation Phase Distributions")
    ax9.legend(fontsize=6)
    ax9.grid(True, alpha=0.3)

    # 10. Mean rank comparison on all triples
    ax10 = fig.add_subplot(gs[2, 2])
    all_metrics = ['hits_at_1', 'hits_at_3', 'hits_at_10', 'mrr']
    complex_all_vals = [results_all['complex'][m] for m in all_metrics]
    real_all_vals = [results_all['real'][m] for m in all_metrics]
    ax10.bar(x_pos - width/2, complex_all_vals, width, label="Complex", color="#2196F3", alpha=0.8)
    ax10.bar(x_pos + width/2, real_all_vals, width, label="Real", color="#FF5722", alpha=0.8)
    ax10.set_xticks(x_pos)
    ax10.set_xticklabels(["H@1", "H@3", "H@10", "MRR"], fontsize=9)
    ax10.set_ylabel("Score")
    ax10.set_title("Link Prediction (all triples)")
    ax10.legend(fontsize=8)
    ax10.set_ylim(0, 1.05)
    ax10.grid(True, alpha=0.3, axis='y')

    # 11. Multi-hop success summary
    ax11 = fig.add_subplot(gs[2, 3])
    hop_names = [r['name'].split(' -> ')[0] + "\n" + "->".join(r['name'].split(' -> ')[1:3])
                 for r in multi_hop_results]
    hop_found = [1 if r['found'] else 0 for r in multi_hop_results]
    hop_ranks = [r['rank'] if r['rank'] > 0 else n_entities for r in multi_hop_results]
    colors = ['#4CAF50' if f else '#F44336' for f in hop_found]
    bars = ax11.bar(range(len(hop_names)), hop_ranks, color=colors, alpha=0.8)
    ax11.set_xticks(range(len(hop_names)))
    ax11.set_xticklabels(hop_names, fontsize=6)
    ax11.set_ylabel("Rank (lower=better)")
    ax11.set_title("Multi-hop Rank")
    ax11.grid(True, alpha=0.3, axis='y')
    for bar, rank in zip(bars, hop_ranks):
        ax11.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(rank) if rank < n_entities else "miss",
                 ha='center', fontsize=8, fontweight='bold')

    fig.suptitle(
        "Complex-Valued Knowledge Graph: Phase Interference vs Real Baseline",
        fontsize=14, fontweight='bold', y=0.98
    )

    # Save
    out_dir = "/home/caug/npcww/qstk/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "kg_demo.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nPlots saved to {out_path}")

    plt.close(fig)

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Complex KG (RotatE, phase interference):")
    print(f"    Hits@1  = {results['complex']['hits_at_1']:.3f}")
    print(f"    Hits@10 = {results['complex']['hits_at_10']:.3f}")
    print(f"    MRR     = {results['complex']['mrr']:.3f}")
    print(f"    MeanRnk = {results['complex']['mean_rank']:.1f}")
    print(f"\n  Real KG (TransE baseline):")
    print(f"    Hits@1  = {results['real']['hits_at_1']:.3f}")
    print(f"    Hits@10 = {results['real']['hits_at_10']:.3f}")
    print(f"    MRR     = {results['real']['mrr']:.3f}")
    print(f"    MeanRnk = {results['real']['mean_rank']:.1f}")

    delta_mrr = results['complex']['mrr'] - results['real']['mrr']
    print(f"\n  MRR advantage (complex - real): {delta_mrr:+.3f}")

    mh_found = sum(1 for r in multi_hop_results if r['found'])
    print(f"  Multi-hop success: {mh_found}/{len(multi_hop_results)} found in top-5")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
