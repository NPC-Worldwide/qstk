"""Demonstration: Phase operator analysis of GPT-2 token embeddings.

We treat each token in a sentence as a point in complex semantic space
and study the *operators* (phase rotations + magnitude scalings) that
move us from one token to the next.

The pipeline:
    1. Tokenize Shakespeare with GPT-2
    2. Extract token embeddings from GPT-2's embedding layer
    3. Project R^768 -> C^384 via Hilbert transform
    4. Compute transition operators between consecutive tokens
    5. Analyze: diversity, spectrum, trajectory coherence
    6. Compare "low temp" (common tokens) vs "high temp" (rare tokens)
    7. Visualize everything

Run:
    python3 -m qstk.cnn.operator_demo
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ---- imports from this package ----
from qstk.cnn.probe import embed_to_complex
from qstk.cnn.operators import (
    transition_operator,
    extract_operators,
    operator_diversity,
    operator_spectrum,
    trajectory_coherence,
    compare_temperature_regimes,
    creativity_correlation,
)


# =====================================================================
# Data: get GPT-2 embeddings for tokens
# =====================================================================

def get_gpt2_embeddings(texts: list[str]):
    """Tokenize texts with GPT-2, return token strings and embedding matrix.

    Returns:
        tokens: list of token strings
        embeddings: np.ndarray, shape [n_tokens, 768]
    """
    from transformers import GPT2Tokenizer, GPT2Model
    import torch

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    model.eval()

    all_tokens = []
    all_embeddings = []

    for text in texts:
        input_ids = tokenizer.encode(text, return_tensors='pt')
        token_strs = [tokenizer.decode([tid]) for tid in input_ids[0]]

        with torch.no_grad():
            # Get the embedding layer output (before any transformer blocks)
            emb = model.wte(input_ids)  # [1, seq_len, 768]
            emb_np = emb.squeeze(0).numpy()

        all_tokens.extend(token_strs)
        all_embeddings.append(emb_np)

    embeddings = np.concatenate(all_embeddings, axis=0)
    return all_tokens, embeddings


def get_targeted_embeddings(token_list: list[str]):
    """Get GPT-2 embeddings for specific tokens (for controlled comparison)."""
    from transformers import GPT2Tokenizer, GPT2Model
    import torch

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    model.eval()

    embeddings = []
    valid_tokens = []

    for token in token_list:
        ids = tokenizer.encode(token)
        # Some tokens may split into multiple sub-tokens; take the first
        if len(ids) > 0:
            tid = ids[0]
            with torch.no_grad():
                emb = model.wte(torch.tensor([[tid]]))
                embeddings.append(emb.squeeze().numpy())
                valid_tokens.append(token)

    return valid_tokens, np.array(embeddings)


# =====================================================================
# Main demo
# =====================================================================

def main():
    output_dir = Path('/home/caug/npcww/qstk/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'operator_demo.png'

    print("=" * 70)
    print("  Phase Operator Analysis of LLM Generation Traces")
    print("=" * 70)

    # ---- 1. Shakespeare trajectory ----
    shakespeare_texts = [
        "To be or not to be that is the question",
        "Whether tis nobler in the mind to suffer",
        "The slings and arrows of outrageous fortune",
    ]
    print("\n[1] Loading GPT-2 embeddings for Shakespeare...")
    tokens, embeddings = get_gpt2_embeddings(shakespeare_texts)
    print(f"    {len(tokens)} tokens, embedding dim = {embeddings.shape[1]}")

    # ---- 2. Project to complex space ----
    print("[2] Projecting to complex space via Hilbert transform...")
    z = embed_to_complex(embeddings, method='hilbert')
    print(f"    Complex shape: {z.shape}  (C^{z.shape[1]})")

    # ---- 3. Extract operators ----
    print("[3] Extracting transition operators...")
    ops = extract_operators(z)
    print(f"    {ops.shape[0]} operators extracted")

    # Verify reconstruction
    z_reconstructed = np.empty_like(z)
    z_reconstructed[0] = z[0]
    for t in range(ops.shape[0]):
        z_reconstructed[t + 1] = ops[t] * z_reconstructed[t]
    recon_error = np.mean(np.abs(z_reconstructed - z))
    print(f"    Reconstruction error: {recon_error:.2e}")

    # ---- 4. Operator diversity ----
    print("[4] Computing operator diversity...")
    div = operator_diversity(ops)
    print(f"    Phase diversity:     {div['phase_diversity']:.4f}")
    print(f"    Magnitude diversity: {div['magnitude_diversity']:.4f}")
    print(f"    Cluster count:       {div['cluster_count']}")
    print(f"    Cluster entropy:     {div['cluster_entropy']:.4f}")

    # ---- 5. Operator spectrum ----
    print("[5] Computing operator spectrum (PCA on phase vectors)...")
    spec = operator_spectrum(ops, n_components=min(10, ops.shape[0] - 1))
    cumvar = np.cumsum(spec['explained_variance_ratio'])
    n_for_90 = int(np.searchsorted(cumvar, 0.90) + 1)
    print(f"    Top eigenvalue explains {spec['explained_variance_ratio'][0]*100:.1f}% of variance")
    print(f"    {n_for_90} components needed for 90% variance")

    # ---- 6. Trajectory coherence ----
    print("[6] Computing trajectory coherence...")
    coh = trajectory_coherence(z)
    print(f"    Phase alignment:   {coh['phase_alignment']:.4f}")
    print(f"    Drift ratio:       {coh['drift_ratio']:.2f}")
    print(f"    Return tendency:   {coh['return_tendency']:.4f}")
    print(f"    Winding number:    {coh['winding_number']:.2f}")

    # ---- 7. Low-temp vs high-temp comparison ----
    print("[7] Comparing low-temp vs high-temp operator regimes...")

    # Low temp: common, predictable function words
    low_temp_tokens = [
        " the", " and", " of", " is", " to", " in", " it", " that",
        " was", " for", " on", " are", " with", " as", " at", " be",
        " this", " have", " from", " or", " an", " by", " not", " but",
        " what", " all", " were", " when", " we", " there", " can",
        " had", " each", " which", " their", " if", " has", " do",
        " will", " about", " up", " out", " them", " then", " she",
        " many", " some", " so", " these", " would", " other",
    ]

    # High temp: rare, vivid, surprising words
    high_temp_tokens = [
        " phantasmagoria", " serendipity", " labyrinthine", " ephemeral",
        " kaleidoscope", " melancholy", " thunderstruck", " luminescent",
        " cataclysm", " iridescent", " juxtaposition", " paradoxical",
        " effervescent", " metamorphosis", " transcendental", " ethereal",
        " incandescent", " quintessential", " halcyon", " resplendent",
        " cacophony", " surreptitious", " magniloquent", " phosphorescent",
        " prismatic", " gossamer", " crystalline", " vermillion",
        " amethyst", " obsidian", " labyrinth", " maelstrom",
        " crescendo", " rhapsody", " nocturne", " tempest",
        " aurora", " solstice", " zenith", " nadir",
        " quixotic", " nebula", " specter", " chimera",
        " vortex", " alchemy", " enigma", " mirage",
        " opalescent", " archipelago",
    ]

    print("    Getting embeddings for common tokens (low temp proxy)...")
    _, low_emb = get_targeted_embeddings(low_temp_tokens)
    print("    Getting embeddings for rare tokens (high temp proxy)...")
    _, high_emb = get_targeted_embeddings(high_temp_tokens)

    z_low = embed_to_complex(low_emb, method='hilbert')
    z_high = embed_to_complex(high_emb, method='hilbert')

    ops_low = extract_operators(z_low)
    ops_high = extract_operators(z_high)

    print(f"    Low-temp operators:  {ops_low.shape[0]}")
    print(f"    High-temp operators: {ops_high.shape[0]}")

    comparison = compare_temperature_regimes(ops_low, ops_high)

    print("\n    --- Low Temperature ---")
    ld = comparison['low_temp']['diversity']
    print(f"    Phase diversity:   {ld['phase_diversity']:.4f}")
    print(f"    Mag diversity:     {ld['magnitude_diversity']:.4f}")
    print(f"    Cluster count:     {ld['cluster_count']}")
    print(f"    Cluster entropy:   {ld['cluster_entropy']:.4f}")
    print(f"    Mean rotation:     {comparison['low_temp']['mean_rotation']:.4f}")

    print("\n    --- High Temperature ---")
    hd = comparison['high_temp']['diversity']
    print(f"    Phase diversity:   {hd['phase_diversity']:.4f}")
    print(f"    Mag diversity:     {hd['magnitude_diversity']:.4f}")
    print(f"    Cluster count:     {hd['cluster_count']}")
    print(f"    Cluster entropy:   {hd['cluster_entropy']:.4f}")
    print(f"    Mean rotation:     {comparison['high_temp']['mean_rotation']:.4f}")

    print("\n    --- Comparison ---")
    c = comparison['comparison']
    print(f"    Basis overlap:     {c['basis_overlap']:.4f}")
    print(f"    Diversity ratio:   {c['diversity_ratio']:.4f}")
    print(f"    Rotation ratio:    {c['rotation_ratio']:.4f}")
    print(f"    Entropy ratio:     {c['entropy_ratio']:.4f}")

    # ---- 8. Creativity correlation (synthetic) ----
    print("\n[8] Creativity correlation (synthetic demonstration)...")

    # Create synthetic generations that blend low/high temp at different ratios
    # Ratio = fraction of high-temp operators mixed in
    np.random.seed(42)
    n_gens = 20
    blend_ratios = np.linspace(0.0, 1.0, n_gens)

    # Synthetic creativity ratings: peak at ~0.7 blend (creative but not random)
    # This models the inverted-U hypothesis: some surprise is creative,
    # pure randomness is not
    creativity_ratings = np.exp(-8 * (blend_ratios - 0.7) ** 2)
    # Add a bit of noise
    creativity_ratings += np.random.randn(n_gens) * 0.05
    creativity_ratings = np.clip(creativity_ratings, 0, 1)

    ops_list = []
    n_low = ops_low.shape[0]
    n_high = ops_high.shape[0]
    seq_len = min(n_low, n_high, 15)

    for ratio in blend_ratios:
        n_from_high = int(ratio * seq_len)
        n_from_low = seq_len - n_from_high
        idx_low = np.random.choice(n_low, size=n_from_low, replace=True)
        idx_high = np.random.choice(n_high, size=n_from_high, replace=True)
        blended = np.concatenate([ops_low[idx_low], ops_high[idx_high]], axis=0)
        np.random.shuffle(blended)
        ops_list.append(blended)

    corr_results = creativity_correlation(ops_list, creativity_ratings)
    print(f"    Best predictor:  {corr_results['best_predictor']}")
    print(f"    Best |r|:        {corr_results['best_predictor_r']:.4f}")
    for key in ['phase_diversity', 'magnitude_diversity', 'mean_rotation', 'cluster_entropy']:
        r = corr_results[f'corr_{key}']
        p = corr_results[f'pval_{key}']
        print(f"    {key:25s}  r = {r:+.4f}  (p = {p:.4f})")

    # =====================================================================
    # Visualization
    # =====================================================================
    print(f"\n[9] Generating plots -> {output_path}")

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Phase Operator Analysis of LLM Generation Traces',
                 fontsize=16, fontweight='bold', y=0.98)

    # --- Panel 1: Operator phases for Shakespeare trajectory ---
    ax1 = fig.add_subplot(3, 3, 1)
    phase_mat = np.angle(ops)
    im = ax1.imshow(phase_mat.T[:50, :], aspect='auto', cmap='twilight',
                    vmin=-np.pi, vmax=np.pi)
    ax1.set_xlabel('Token step')
    ax1.set_ylabel('Complex dimension')
    ax1.set_title('Operator Phases\n(Shakespeare)')
    plt.colorbar(im, ax=ax1, label='Phase (rad)')

    # --- Panel 2: Operator magnitudes ---
    ax2 = fig.add_subplot(3, 3, 2)
    mag_mat = np.abs(ops)
    im2 = ax2.imshow(np.log1p(mag_mat.T[:50, :]), aspect='auto', cmap='magma')
    ax2.set_xlabel('Token step')
    ax2.set_ylabel('Complex dimension')
    ax2.set_title('Operator Magnitudes\n(log scale)')
    plt.colorbar(im2, ax=ax2, label='log(1 + |O|)')

    # --- Panel 3: Spectrum (eigenvalue decay) ---
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.bar(range(len(spec['explained_variance_ratio'])),
            spec['explained_variance_ratio'], color='steelblue', alpha=0.8)
    ax3.plot(range(len(spec['explained_variance_ratio'])),
             np.cumsum(spec['explained_variance_ratio']),
             'o-', color='crimson', label='Cumulative')
    ax3.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    ax3.set_xlabel('Principal component')
    ax3.set_ylabel('Variance explained')
    ax3.set_title('Operator Spectrum\n(phase PCA)')
    ax3.legend(fontsize=8)

    # --- Panel 4: Phase trajectory (first 2 PCA dims) ---
    ax4 = fig.add_subplot(3, 3, 4)
    phases_all = np.angle(ops)
    from sklearn.decomposition import PCA
    pca2d = PCA(n_components=2)
    proj = pca2d.fit_transform(phases_all)
    colors = np.arange(len(proj))
    sc = ax4.scatter(proj[:, 0], proj[:, 1], c=colors, cmap='viridis',
                     s=40, zorder=2)
    ax4.plot(proj[:, 0], proj[:, 1], '-', color='gray', alpha=0.3, zorder=1)
    # Mark start and end
    ax4.scatter(proj[0, 0], proj[0, 1], marker='s', s=120, c='green',
                edgecolors='black', zorder=3, label='Start')
    ax4.scatter(proj[-1, 0], proj[-1, 1], marker='*', s=180, c='red',
                edgecolors='black', zorder=3, label='End')
    ax4.set_xlabel('Phase PC1')
    ax4.set_ylabel('Phase PC2')
    ax4.set_title('Operator Trajectory\n(phase space)')
    ax4.legend(fontsize=8)
    plt.colorbar(sc, ax=ax4, label='Token step')

    # --- Panel 5: Low-temp vs high-temp phase distributions ---
    ax5 = fig.add_subplot(3, 3, 5)
    low_mean_phases = np.mean(np.angle(ops_low), axis=1)
    high_mean_phases = np.mean(np.angle(ops_high), axis=1)
    bins = np.linspace(-np.pi, np.pi, 30)
    ax5.hist(low_mean_phases, bins=bins, alpha=0.6, color='steelblue',
             label=f'Low temp (div={ld["phase_diversity"]:.3f})', density=True)
    ax5.hist(high_mean_phases, bins=bins, alpha=0.6, color='coral',
             label=f'High temp (div={hd["phase_diversity"]:.3f})', density=True)
    ax5.set_xlabel('Mean operator phase')
    ax5.set_ylabel('Density')
    ax5.set_title('Phase Distribution\nby Temperature')
    ax5.legend(fontsize=7)

    # --- Panel 6: Low vs high temp magnitude distributions ---
    ax6 = fig.add_subplot(3, 3, 6)
    low_mean_mags = np.mean(np.abs(ops_low), axis=1)
    high_mean_mags = np.mean(np.abs(ops_high), axis=1)
    ax6.hist(low_mean_mags, bins=25, alpha=0.6, color='steelblue',
             label='Low temp', density=True)
    ax6.hist(high_mean_mags, bins=25, alpha=0.6, color='coral',
             label='High temp', density=True)
    ax6.set_xlabel('Mean operator magnitude')
    ax6.set_ylabel('Density')
    ax6.set_title('Magnitude Distribution\nby Temperature')
    ax6.legend(fontsize=8)

    # --- Panel 7: Spectrum comparison ---
    ax7 = fig.add_subplot(3, 3, 7)
    low_evr = comparison['low_temp']['spectrum']['explained_variance_ratio']
    high_evr = comparison['high_temp']['spectrum']['explained_variance_ratio']
    n_show = min(len(low_evr), len(high_evr))
    x = np.arange(n_show)
    w = 0.35
    ax7.bar(x - w/2, low_evr[:n_show], w, color='steelblue', alpha=0.8,
            label='Low temp')
    ax7.bar(x + w/2, high_evr[:n_show], w, color='coral', alpha=0.8,
            label='High temp')
    ax7.set_xlabel('Principal component')
    ax7.set_ylabel('Variance explained')
    ax7.set_title(f'Spectrum Comparison\n(basis overlap = {c["basis_overlap"]:.3f})')
    ax7.legend(fontsize=8)

    # --- Panel 8: Creativity correlation scatter ---
    ax8 = fig.add_subplot(3, 3, 8)
    # Plot best predictor vs creativity
    best_key = corr_results['best_predictor']
    predictor_values = []
    for op_set in ops_list:
        d = operator_diversity(op_set)
        if best_key == 'phase_diversity':
            predictor_values.append(d['phase_diversity'])
        elif best_key == 'magnitude_diversity':
            predictor_values.append(d['magnitude_diversity'])
        elif best_key == 'cluster_entropy':
            predictor_values.append(d['cluster_entropy'])
        elif best_key == 'mean_rotation':
            predictor_values.append(float(np.mean(np.abs(np.angle(op_set)))))
    predictor_values = np.array(predictor_values)

    ax8.scatter(predictor_values, creativity_ratings, c=blend_ratios,
                cmap='coolwarm', s=50, edgecolors='black', linewidth=0.5)
    # Fit line
    if np.std(predictor_values) > 1e-10:
        coeffs = np.polyfit(predictor_values, creativity_ratings, 1)
        x_fit = np.linspace(predictor_values.min(), predictor_values.max(), 50)
        ax8.plot(x_fit, np.polyval(coeffs, x_fit), '--', color='gray', alpha=0.7)
    ax8.set_xlabel(best_key.replace('_', ' ').title())
    ax8.set_ylabel('Creativity Rating')
    r_val = corr_results[f'corr_{best_key}']
    ax8.set_title(f'Creativity Correlation\nr = {r_val:+.3f}')

    # --- Panel 9: Summary metrics table ---
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = (
        "TRAJECTORY COHERENCE (Shakespeare)\n"
        f"  Phase alignment:   {coh['phase_alignment']:.4f}\n"
        f"  Drift ratio:       {coh['drift_ratio']:.2f}\n"
        f"  Return tendency:   {coh['return_tendency']:.4f}\n"
        f"  Winding number:    {coh['winding_number']:.2f}\n"
        "\n"
        "TEMPERATURE COMPARISON\n"
        f"  Diversity ratio (H/L):  {c['diversity_ratio']:.3f}\n"
        f"  Rotation ratio (H/L):   {c['rotation_ratio']:.3f}\n"
        f"  Entropy ratio (H/L):    {c['entropy_ratio']:.3f}\n"
        f"  Basis overlap:          {c['basis_overlap']:.3f}\n"
        "\n"
        "CREATIVITY CORRELATION\n"
        f"  Best predictor: {corr_results['best_predictor']}\n"
        f"  |r| = {corr_results['best_predictor_r']:.3f}\n"
    )
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax9.set_title('Summary Metrics')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n    Plot saved to {output_path}")
    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == '__main__':
    main()
