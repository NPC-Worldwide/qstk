"""
Tools for probing real-valued LLM embeddings in complex space.

Project R^d embeddings to C^{d/2}, then analyze the emergent phase
structure: coherence, clustering, interference, and entanglement.

Functions:
    embed_to_complex      -- project real embeddings to complex space
    phase_coherence       -- mean phase alignment across a set of vectors
    phase_clusters        -- k-means on phases, return cluster assignments
    interference_score    -- cosine-like score via complex inner product
    semantic_entanglement -- CHSH-like S-value from phase correlations
"""

import numpy as np
from typing import Literal, Tuple


def embed_to_complex(
    real_embeddings: np.ndarray,
    method: Literal['hilbert', 'paired', 'random_proj'] = 'hilbert',
    seed: int = 42,
) -> np.ndarray:
    """Project real-valued embeddings R^d -> C^{d/2}.

    Three methods:
      - 'hilbert': Apply discrete Hilbert transform to get analytic signal.
        Pairs each frequency component with its Hilbert-transformed partner.
      - 'paired': Simply pair adjacent dimensions: z_k = x_{2k} + i*x_{2k+1}.
      - 'random_proj': Learned random projection using a fixed complex matrix.

    Args:
        real_embeddings: Real array, shape [..., d]. d must be even.
        method:          Projection method.
        seed:            Random seed for 'random_proj' method.

    Returns:
        Complex array, shape [..., d//2].

    Raises:
        ValueError: If d is odd.

    Example:
        >>> real_emb = np.random.randn(100, 768)   # e.g. BERT embeddings
        >>> z = embed_to_complex(real_emb)          # [100, 384] complex128
    """
    d = real_embeddings.shape[-1]
    if d % 2 != 0:
        raise ValueError(f"Embedding dim must be even, got {d}")

    if method == 'hilbert':
        return _hilbert_projection(real_embeddings)
    elif method == 'paired':
        return _paired_projection(real_embeddings)
    elif method == 'random_proj':
        return _random_projection(real_embeddings, seed)
    else:
        raise ValueError(f"Unknown method: {method!r}")


def _hilbert_projection(x: np.ndarray) -> np.ndarray:
    """Hilbert transform: analytic signal in frequency domain.

    The analytic signal z = x + i*H(x) has the property that its
    spectrum is one-sided (only positive frequencies). The phase of z
    captures the instantaneous frequency structure of x.
    """
    d = x.shape[-1]
    # FFT along the feature dimension
    X = np.fft.fft(x, axis=-1)

    # Build Hilbert multiplier: double positive freqs, zero negative
    h = np.zeros(d)
    h[0] = 1.0  # DC component
    if d % 2 == 0:
        h[d // 2] = 1.0  # Nyquist
    h[1:(d + 1) // 2] = 2.0  # positive frequencies

    Z = X * h
    analytic = np.fft.ifft(Z, axis=-1)

    # Take first d//2 components (positive freq half)
    return analytic[..., :d // 2]


def _paired_projection(x: np.ndarray) -> np.ndarray:
    """Simple pairing: z_k = x_{2k} + i*x_{2k+1}."""
    d = x.shape[-1]
    return x[..., 0::2] + 1j * x[..., 1::2]


def _random_projection(x: np.ndarray, seed: int) -> np.ndarray:
    """Fixed random complex projection matrix."""
    d = x.shape[-1]
    d_out = d // 2
    rng = np.random.RandomState(seed)
    W_re = rng.randn(d, d_out) / np.sqrt(d)
    W_im = rng.randn(d, d_out) / np.sqrt(d)
    W = W_re + 1j * W_im
    return x @ W


def phase_coherence(z: np.ndarray) -> float:
    """Mean phase alignment across a set of complex vectors.

    For each dimension k, compute the circular mean resultant length:
        R_k = |mean(e^{i*angle(z[:, k])})|
    Then average over all dimensions.

    R_k = 1 means all vectors have identical phase in dim k (perfect alignment).
    R_k = 0 means phases are uniformly distributed (no structure).

    Args:
        z: Complex array, shape [n_vectors, dim].

    Returns:
        Scalar in [0, 1]. Higher = more phase structure.

    Example:
        >>> z_random = np.random.randn(100, 64) + 1j * np.random.randn(100, 64)
        >>> phase_coherence(z_random)  # close to 0.1 (1/sqrt(n))
        >>> z_aligned = np.exp(1j * 0.5) * np.abs(z_random)
        >>> phase_coherence(z_aligned)  # close to 1.0
    """
    if z.ndim == 1:
        z = z[None, :]

    phases = np.angle(z)  # [n, dim]
    # Circular mean resultant length per dimension
    R = np.abs(np.mean(np.exp(1j * phases), axis=0))  # [dim]
    return float(np.mean(R))


def phase_clusters(
    z: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """K-means clustering on phases.

    Clusters vectors by their phase patterns (ignoring magnitude).
    Uses circular k-means: distance = 1 - |<unit_z, centroid>| where
    the inner product is on unit-magnitude complex vectors.

    Args:
        z:          Complex array, shape [n_vectors, dim].
        n_clusters: Number of clusters.
        max_iter:   Maximum iterations for k-means.
        seed:       Random seed for initialization.

    Returns:
        Tuple of:
          - assignments: int array [n_vectors], cluster labels.
          - centroids: complex array [n_clusters, dim], cluster centers
            (unit magnitude).

    Example:
        >>> z = np.vstack([
        ...     np.exp(1j * 0.5) * np.ones((50, 32)),
        ...     np.exp(1j * 2.0) * np.ones((50, 32)),
        ... ])
        >>> labels, centroids = phase_clusters(z, n_clusters=2)
    """
    n, dim = z.shape
    rng = np.random.RandomState(seed)

    # Normalize to unit magnitude (phase-only)
    z_unit = z / (np.abs(z) + 1e-10)

    # Initialize centroids randomly from data
    idx = rng.choice(n, size=n_clusters, replace=False)
    centroids = z_unit[idx].copy()

    assignments = np.zeros(n, dtype=np.int64)

    for _ in range(max_iter):
        # Assign each point to nearest centroid
        # Similarity = |z_unit @ centroid^H| (phase alignment)
        sim = np.abs(z_unit @ centroids.conj().T)  # [n, n_clusters]
        new_assignments = np.argmax(sim, axis=1)

        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments

        # Update centroids
        for c in range(n_clusters):
            mask = assignments == c
            if np.any(mask):
                mean_z = z_unit[mask].mean(axis=0)
                centroids[c] = mean_z / (np.abs(mean_z) + 1e-10)

    return assignments, centroids


def interference_score(z1: np.ndarray, z2: np.ndarray) -> float:
    """Interference score: phase-aware cosine similarity.

    Computes Re(z1 . conj(z2)) / (|z1| * |z2|).

    This is the real part of the normalized complex inner product:
      +1 = constructive interference (phases aligned)
       0 = orthogonal (phases at 90 degrees)
      -1 = destructive interference (phases opposed)

    Args:
        z1: Complex array, shape [dim] or [batch, dim].
        z2: Complex array, same shape as z1.

    Returns:
        Scalar interference score in [-1, 1].

    Example:
        >>> z = np.array([1+1j, 2+0j, 0+3j])
        >>> interference_score(z, z)       # 1.0 (self-aligned)
        >>> interference_score(z, -z)      # -1.0 (destructive)
        >>> interference_score(z, z * 1j)  # 0.0 (orthogonal)
    """
    dot = np.sum(z1 * z2.conj())
    norm1 = np.sqrt(np.sum(np.abs(z1) ** 2))
    norm2 = np.sqrt(np.sum(np.abs(z2) ** 2))
    return float(dot.real / (norm1 * norm2 + 1e-10))


def semantic_entanglement(z_pairs: np.ndarray) -> float:
    """Compute CHSH-like S-value from phase correlations.

    Inspired by the CHSH inequality from quantum mechanics.
    For pairs of semantically related tokens, measure whether their
    phase correlations exceed the classical bound of 2.

    The S-value is computed as:
        S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|
    where E(a,b) = Re(<z_a, z_b>) / (|z_a| |z_b|) is the interference
    score, and a/a', b/b' are obtained by rotating the measurement basis
    by 0 and pi/4.

    Args:
        z_pairs: Complex array, shape [n_pairs, 2, dim].
                 Each row is a pair of semantically related embeddings.

    Returns:
        S-value. Classical bound is 2.0; values > 2.0 suggest
        "quantum-like" phase entanglement (non-local correlations
        in the phase structure).

    Example:
        >>> pairs = np.stack([z_synonyms, z_antonyms], axis=1)  # [n, 2, dim]
        >>> S = semantic_entanglement(pairs)
        >>> print(f"S = {S:.3f} ({'entangled' if S > 2 else 'classical'})")
    """
    n_pairs = z_pairs.shape[0]
    dim = z_pairs.shape[2]

    z_a = z_pairs[:, 0, :]  # [n, dim]
    z_b = z_pairs[:, 1, :]  # [n, dim]

    def E(za: np.ndarray, zb: np.ndarray) -> float:
        """Mean interference score across pairs."""
        dots = np.sum(za * zb.conj(), axis=-1)  # [n]
        norms = np.sqrt(np.sum(np.abs(za) ** 2, axis=-1)) * \
                np.sqrt(np.sum(np.abs(zb) ** 2, axis=-1))
        return float(np.mean(dots.real / (norms + 1e-10)))

    # Four measurement settings: rotate by 0, pi/8, pi/4, 3pi/8
    theta = [0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8]
    rotations = [np.exp(1j * t) for t in theta]

    # CHSH correlators:
    # E(a, b)   with angles (0, pi/8)
    # E(a, b')  with angles (0, 3pi/8)
    # E(a', b)  with angles (pi/4, pi/8)
    # E(a', b') with angles (pi/4, 3pi/8)
    E_ab  = E(z_a * rotations[0], z_b * rotations[1])
    E_ab2 = E(z_a * rotations[0], z_b * rotations[3])
    E_a2b = E(z_a * rotations[2], z_b * rotations[1])
    E_a2b2 = E(z_a * rotations[2], z_b * rotations[3])

    S = abs(E_ab - E_ab2 + E_a2b + E_a2b2)
    return float(S)
