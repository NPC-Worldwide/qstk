"""Probe real-valued embeddings in complex space.

Demonstrates:
  - Projecting real embeddings to complex space (no torch needed)
  - Measuring phase coherence
  - Computing interference scores between embedding pairs
  - Analyzing semantic entanglement via CHSH-like S-values

Usage:
    python -m qstk.cnn.examples.probe_embeddings
"""

import numpy as np
from qstk.cnn import embed_to_complex, phase_coherence, interference_score
from qstk.cnn.probe import phase_clusters, semantic_entanglement


def main():
    rng = np.random.RandomState(42)

    # -- Simulate real-valued embeddings (e.g., from an LLM) --
    n_tokens = 1000
    real_dim = 768
    real_embeds = rng.randn(n_tokens, real_dim)
    print(f"Real embeddings: {real_embeds.shape} ({real_embeds.dtype})")

    # -- Project to complex space --
    complex_embeds = embed_to_complex(real_embeds)
    print(f"Complex embeddings: {complex_embeds.shape} ({complex_embeds.dtype})")

    # -- Phase coherence --
    pc = phase_coherence(complex_embeds)
    print(f"\nPhase coherence: {pc:.4f}")
    print(f"  (1.0 = perfect alignment, ~0.03 = random for n={n_tokens})")

    # -- Compare projection methods --
    for method in ["hilbert", "paired", "random_proj"]:
        z = embed_to_complex(real_embeds, method=method)
        pc_m = phase_coherence(z)
        print(f"  {method:12s} -> coherence = {pc_m:.4f}")

    # -- Interference between pairs --
    print("\nInterference scores (phase-aware cosine similarity):")
    z = complex_embeds

    # Self-interference (always 1.0)
    score_self = interference_score(z[0], z[0])
    print(f"  z[0] vs z[0] (self):     {score_self:+.4f}")

    # Arbitrary pair
    score_pair = interference_score(z[0], z[1])
    print(f"  z[0] vs z[1] (random):   {score_pair:+.4f}")

    # Negation (destructive)
    score_neg = interference_score(z[0], -z[0])
    print(f"  z[0] vs -z[0] (neg):     {score_neg:+.4f}")

    # Orthogonal rotation
    score_orth = interference_score(z[0], z[0] * 1j)
    print(f"  z[0] vs z[0]*i (orth):   {score_orth:+.4f}")

    # -- Phase clustering --
    print("\nPhase clustering (k=4):")
    labels, centroids = phase_clusters(z[:200], n_clusters=4)
    for c in range(4):
        count = int(np.sum(labels == c))
        print(f"  Cluster {c}: {count} vectors")

    # -- Semantic entanglement (CHSH) --
    # Create synthetic "related" pairs: shift each embedding slightly
    z_a = z[:100]
    z_b = z_a * np.exp(1j * 0.3) + 0.1 * (rng.randn(100, z.shape[1]) + 1j * rng.randn(100, z.shape[1]))
    pairs = np.stack([z_a, z_b], axis=1)  # [100, 2, dim]

    S = semantic_entanglement(pairs)
    print(f"\nSemantic entanglement (CHSH S-value): {S:.3f}")
    print(f"  Classical bound = 2.0, quantum bound = 2*sqrt(2) ~ 2.828")
    print(f"  {'Entangled (S > 2)' if S > 2.0 else 'Classical (S <= 2)'}")


if __name__ == "__main__":
    main()
