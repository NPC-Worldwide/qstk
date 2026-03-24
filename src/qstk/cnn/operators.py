"""Phase operator analysis for LLM generation traces in complex space.

When a language model generates text, each token-to-token transition is
an *operator* acting on the semantic state.  If we embed each token into
C^d (via Hilbert transform of real embeddings), the generation trajectory
becomes a sequence of complex vectors:

    z_0  -->  z_1  -->  z_2  -->  ...  -->  z_T

Each step is governed by an element-wise operator O_t such that:

    z_t  =  O_t  (hadamard)  z_{t-1}

where O_t encodes two things per dimension:
    - a phase rotation (the *direction* the model turns in semantic space)
    - a magnitude scaling (how much the model amplifies or suppresses that
      dimension)

This is the complex-valued analogue of studying the Jacobian of a
dynamical system, but dimension-by-dimension and in polar coordinates.

Why this matters for creativity research:
    - Predictable text reuses the same few operators (low diversity)
    - Creative text explores a richer operator vocabulary
    - Temperature acts as a dial on the operator entropy
    - The principal rotation axes (from PCA on phase vectors) reveal the
      finite basis of semantic maneuvers the model actually uses

Functions:
    transition_operator        -- O such that z2 ~ O (hadamard) z1
    extract_operators          -- full operator sequence from a trajectory
    operator_diversity         -- diversity metrics on a set of operators
    operator_spectrum          -- PCA decomposition into basis rotations
    trajectory_coherence       -- coherence/winding analysis of a path
    compare_temperature_regimes -- contrast operator statistics at two temps
    creativity_correlation     -- link operator statistics to creativity ratings
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Transition operator
# ---------------------------------------------------------------------------

def transition_operator(z1: np.ndarray, z2: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Compute the element-wise operator O such that z2 ~ O (hadamard) z1.

    Decompose in polar form:

        O_k = |z2_k| / |z1_k|  *  exp(i * (angle(z2_k) - angle(z1_k)))

    This is just z2 / z1 done carefully -- we separate phase rotation from
    magnitude scaling, handle zeros gracefully, and return the result in
    complex form so it composes naturally: applying O twice is just O*O.

    Args:
        z1: Complex array, shape [d] or [batch, d].
        z2: Complex array, same shape.
        eps: Stability floor for near-zero magnitudes.

    Returns:
        Complex array same shape as z1, z2.
        The operator in polar form: |scale| * exp(i * delta_phase).

    Example:
        >>> z1 = np.array([1+1j, 2+0j, 0+3j])
        >>> z2 = np.array([0+2j, 4+0j, 0-3j])
        >>> O = transition_operator(z1, z2)
        >>> np.allclose(O * z1, z2, atol=1e-8)
        True
    """
    mag1 = np.abs(z1)
    mag2 = np.abs(z2)

    # Magnitude scaling per dimension
    scale = mag2 / np.maximum(mag1, eps)

    # Phase rotation per dimension
    phase1 = np.angle(z1)
    phase2 = np.angle(z2)
    delta_phase = phase2 - phase1

    # Combine into a single complex operator
    return scale * np.exp(1j * delta_phase)


# ---------------------------------------------------------------------------
# 2. Extract operator sequence
# ---------------------------------------------------------------------------

def extract_operators(embeddings_sequence: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Compute transition operators between consecutive embeddings.

    Given a trajectory [z_0, z_1, ..., z_T], return [O_1, O_2, ..., O_T]
    where O_t is the operator that maps z_{t-1} to z_t.

    Think of it like computing the velocity field of a particle path,
    except the "velocity" is a complex rotation+scaling at each step.

    Args:
        embeddings_sequence: Complex array, shape [T+1, d].
        eps: Stability floor.

    Returns:
        Complex array, shape [T, d].
    """
    T_plus_1, d = embeddings_sequence.shape
    operators = np.empty((T_plus_1 - 1, d), dtype=np.complex128)
    for t in range(T_plus_1 - 1):
        operators[t] = transition_operator(
            embeddings_sequence[t], embeddings_sequence[t + 1], eps=eps
        )
    return operators


# ---------------------------------------------------------------------------
# 3. Operator diversity
# ---------------------------------------------------------------------------

def operator_diversity(operators: np.ndarray, max_k: int = 10) -> Dict[str, float]:
    """Measure the diversity of a set of operators.

    Four complementary metrics:

    Phase diversity:  Circular variance of the *mean phase* of each operator.
        If every operator rotates by the same angle, circular variance = 0.
        If rotations are spread uniformly around the circle, it approaches 1.

    Magnitude diversity:  Standard deviation of the mean magnitude scaling.
        Uniform scaling = 0; wildly varying amplification = high.

    Cluster count:  How many distinct "types" of operator are there?
        We cluster on the phase vectors using k-means and pick the k
        with the best silhouette score (up to max_k).

    Entropy:  Shannon entropy of the cluster assignment distribution.
        log(k) for a uniform partition; 0 if everything is in one cluster.

    Args:
        operators: Complex array, shape [n_ops, d].
        max_k: Maximum number of clusters to try.

    Returns:
        Dict with keys: phase_diversity, magnitude_diversity,
        cluster_count, cluster_entropy.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n_ops = operators.shape[0]

    # --- Phase diversity ---
    phases = np.angle(operators)          # [n_ops, d]
    mean_phase_per_op = np.mean(phases, axis=1)  # [n_ops]
    # Circular variance: 1 - |mean(e^{i*theta})|
    resultant = np.abs(np.mean(np.exp(1j * mean_phase_per_op)))
    phase_div = 1.0 - resultant

    # --- Magnitude diversity ---
    magnitudes = np.abs(operators)        # [n_ops, d]
    mean_mag_per_op = np.mean(magnitudes, axis=1)  # [n_ops]
    mag_div = float(np.std(mean_mag_per_op))

    # --- Clustering on phase vectors ---
    # Represent each operator by its phase vector (circular data on the
    # unit circle per dimension).  For k-means in Euclidean space, we
    # embed angles as (cos, sin) pairs.
    cos_sin = np.concatenate([np.cos(phases), np.sin(phases)], axis=1)  # [n_ops, 2d]

    best_k = 1
    best_score = -1.0
    best_labels = np.zeros(n_ops, dtype=int)

    upper_k = min(max_k, n_ops - 1)
    if upper_k < 2:
        # Too few operators to cluster meaningfully
        best_k = 1
        best_labels = np.zeros(n_ops, dtype=int)
    else:
        for k in range(2, upper_k + 1):
            km = KMeans(n_clusters=k, n_init=5, random_state=42, max_iter=100)
            labels = km.fit_predict(cos_sin)
            # Silhouette score requires at least 2 distinct labels
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(cos_sin, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

    # --- Entropy of cluster distribution ---
    counts = np.bincount(best_labels, minlength=best_k).astype(float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log(probs)))

    return {
        'phase_diversity': float(phase_div),
        'magnitude_diversity': float(mag_div),
        'cluster_count': int(best_k),
        'cluster_entropy': entropy,
    }


# ---------------------------------------------------------------------------
# 4. Operator spectrum
# ---------------------------------------------------------------------------

def operator_spectrum(operators: np.ndarray, n_components: int = 10) -> Dict[str, np.ndarray]:
    """Decompose operators into a basis via PCA on phase vectors.

    The key insight: if an LLM uses a finite repertoire of semantic
    maneuvers, then the space of phase rotation vectors it actually
    visits is low-dimensional.  PCA on the phase vectors finds the
    *principal rotation axes* -- the basis operators from which all
    observed operators can be (approximately) reconstructed.

    This is analogous to finding the normal modes of a vibrating string,
    except the "string" is the model's trajectory through complex
    semantic space.

    Args:
        operators: Complex array, shape [n_ops, d].
        n_components: Number of principal components to extract.

    Returns:
        Dict with:
            eigenvalues:  Variance explained per axis, shape [n_comp].
            eigenvectors: The basis operators (as phase vectors),
                          shape [n_comp, d].
            explained_variance_ratio: Fraction of total variance per axis.
    """
    from sklearn.decomposition import PCA

    phases = np.angle(operators)  # [n_ops, d]

    n_comp = min(n_components, min(phases.shape))
    pca = PCA(n_components=n_comp)
    pca.fit(phases)

    return {
        'eigenvalues': pca.explained_variance_,
        'eigenvectors': pca.components_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
    }


# ---------------------------------------------------------------------------
# 5. Trajectory coherence
# ---------------------------------------------------------------------------

def trajectory_coherence(embeddings_sequence: np.ndarray) -> Dict[str, float]:
    """Measure how coherent a generation trajectory is in complex space.

    Four complementary views of the same path:

    Phase alignment:  Mean cosine of the phase difference between
        consecutive tokens.  A value near 1 means the model keeps
        rotating in the same direction (coherent flow).  Near 0 means
        random phase jumps (decoherent).

    Embedding drift:  Cumulative distance traveled, normalized by the
        straight-line distance from start to end.  Drift ratio = 1 for
        a straight path; >>1 for a winding, exploratory path.

    Return tendency:  Cosine similarity between z_T and z_0 in complex
        space.  Positive = the trajectory orbits back; negative = it
        drifts away permanently.

    Winding number:  Total accumulated phase rotation (summed over all
        dimensions, all steps) divided by 2*pi.  Counts how many full
        turns the trajectory makes through phase space.

    Args:
        embeddings_sequence: Complex array, shape [T+1, d].

    Returns:
        Dict with keys: phase_alignment, drift_ratio, return_tendency,
        winding_number.
    """
    T_plus_1, d = embeddings_sequence.shape

    # --- Phase alignment ---
    phases = np.angle(embeddings_sequence)  # [T+1, d]
    phase_diffs = np.diff(phases, axis=0)   # [T, d]
    # Cosine of phase difference, averaged over dims and steps
    alignment = float(np.mean(np.cos(phase_diffs)))

    # --- Embedding drift ---
    step_distances = np.array([
        np.linalg.norm(embeddings_sequence[t + 1] - embeddings_sequence[t])
        for t in range(T_plus_1 - 1)
    ])
    cumulative_distance = float(np.sum(step_distances))
    start_to_end = float(np.linalg.norm(
        embeddings_sequence[-1] - embeddings_sequence[0]
    ))
    drift_ratio = cumulative_distance / max(start_to_end, 1e-10)

    # --- Return tendency ---
    z0 = embeddings_sequence[0]
    zT = embeddings_sequence[-1]
    dot = np.sum(z0 * zT.conj())
    norm0 = np.sqrt(np.sum(np.abs(z0) ** 2))
    normT = np.sqrt(np.sum(np.abs(zT) ** 2))
    return_tendency = float(dot.real / (norm0 * normT + 1e-10))

    # --- Winding number ---
    # Wrap phase differences to [-pi, pi], then sum
    wrapped_diffs = (phase_diffs + np.pi) % (2 * np.pi) - np.pi
    total_winding = float(np.sum(np.abs(wrapped_diffs))) / (2 * np.pi)

    return {
        'phase_alignment': alignment,
        'drift_ratio': drift_ratio,
        'return_tendency': return_tendency,
        'winding_number': total_winding,
    }


# ---------------------------------------------------------------------------
# 6. Compare temperature regimes
# ---------------------------------------------------------------------------

def compare_temperature_regimes(
    low_temp_ops: np.ndarray,
    high_temp_ops: np.ndarray,
) -> Dict[str, object]:
    """Compare operator statistics between two temperature regimes.

    This is the core comparison for the creativity paper: low temperature
    generates predictable text (reusing the same few operators), while
    high temperature explores a wider operator vocabulary.

    We compare:
        - Operator diversity at each temperature
        - Mean rotation magnitude (how far each step rotates)
        - Operator basis overlap (do the two regimes use the same
          principal rotation axes, or entirely different ones?)

    The basis overlap is computed as the mean absolute cosine similarity
    between the top eigenvectors from each regime.  An overlap of 1 means
    identical basis; 0 means orthogonal (completely different maneuvers).

    Args:
        low_temp_ops:  Complex array, shape [n_low, d].
        high_temp_ops: Complex array, shape [n_high, d].

    Returns:
        Dict with sub-dicts for each regime and comparison metrics.
    """
    low_div = operator_diversity(low_temp_ops)
    high_div = operator_diversity(high_temp_ops)

    # Mean rotation magnitude
    low_rot = float(np.mean(np.abs(np.angle(low_temp_ops))))
    high_rot = float(np.mean(np.abs(np.angle(high_temp_ops))))

    # Operator basis overlap: cosine similarity of top eigenvectors
    n_basis = min(5, min(low_temp_ops.shape))
    low_spec = operator_spectrum(low_temp_ops, n_components=n_basis)
    high_spec = operator_spectrum(high_temp_ops, n_components=n_basis)

    # Compute overlap matrix between basis vectors
    n_low_vecs = low_spec['eigenvectors'].shape[0]
    n_high_vecs = high_spec['eigenvectors'].shape[0]
    n_compare = min(n_low_vecs, n_high_vecs)

    if n_compare > 0:
        overlap_matrix = np.abs(
            low_spec['eigenvectors'][:n_compare]
            @ high_spec['eigenvectors'][:n_compare].T
        )
        # Normalize rows by vector norms
        low_norms = np.linalg.norm(low_spec['eigenvectors'][:n_compare], axis=1, keepdims=True)
        high_norms = np.linalg.norm(high_spec['eigenvectors'][:n_compare], axis=1, keepdims=True)
        denom = low_norms @ high_norms.T + 1e-10
        overlap_matrix = overlap_matrix / denom
        # Mean of diagonal = how well each axis matches its partner
        basis_overlap = float(np.mean(np.diag(overlap_matrix)))
    else:
        basis_overlap = 0.0

    return {
        'low_temp': {
            'diversity': low_div,
            'mean_rotation': low_rot,
            'spectrum': {
                'eigenvalues': low_spec['eigenvalues'].tolist(),
                'explained_variance_ratio': low_spec['explained_variance_ratio'].tolist(),
            },
        },
        'high_temp': {
            'diversity': high_div,
            'mean_rotation': high_rot,
            'spectrum': {
                'eigenvalues': high_spec['eigenvalues'].tolist(),
                'explained_variance_ratio': high_spec['explained_variance_ratio'].tolist(),
            },
        },
        'comparison': {
            'basis_overlap': basis_overlap,
            'diversity_ratio': (
                high_div['phase_diversity'] / max(low_div['phase_diversity'], 1e-10)
            ),
            'rotation_ratio': high_rot / max(low_rot, 1e-10),
            'entropy_ratio': (
                high_div['cluster_entropy'] / max(low_div['cluster_entropy'], 1e-10)
            ),
        },
    }


# ---------------------------------------------------------------------------
# 7. Creativity correlation
# ---------------------------------------------------------------------------

def creativity_correlation(
    operators_list: List[np.ndarray],
    ratings: np.ndarray,
) -> Dict[str, float]:
    """Correlate operator statistics with creativity ratings.

    Given operators from multiple generations and corresponding creativity
    ratings (human or LLM-judged), compute which operator statistics are
    the best predictors of creativity.

    The hypothesis: creative text is characterized by high phase diversity,
    moderate (not maximal) rotation magnitude, and high cluster entropy.
    Purely random text has maximal diversity but is not creative. The
    sweet spot is between predictable and random.

    Args:
        operators_list: List of complex arrays, one per generation.
                        Each has shape [n_ops_i, d].
        ratings:        Float array, shape [n_generations].
                        Creativity ratings in [0, 1] or any scale.

    Returns:
        Dict with Pearson correlations between each statistic and ratings,
        plus the name of the best predictor.
    """
    from scipy.stats import pearsonr

    n = len(operators_list)
    assert n == len(ratings), "Need one rating per generation"

    # Compute operator statistics for each generation
    phase_divs = np.zeros(n)
    mag_divs = np.zeros(n)
    mean_rots = np.zeros(n)
    entropies = np.zeros(n)

    for i, ops in enumerate(operators_list):
        div = operator_diversity(ops)
        phase_divs[i] = div['phase_diversity']
        mag_divs[i] = div['magnitude_diversity']
        entropies[i] = div['cluster_entropy']
        mean_rots[i] = float(np.mean(np.abs(np.angle(ops))))

    ratings = np.asarray(ratings, dtype=float)

    # Pearson correlations
    results = {}
    predictors = {
        'phase_diversity': phase_divs,
        'magnitude_diversity': mag_divs,
        'mean_rotation': mean_rots,
        'cluster_entropy': entropies,
    }

    best_name = None
    best_abs_r = -1.0

    for name, values in predictors.items():
        if np.std(values) < 1e-10 or np.std(ratings) < 1e-10:
            r, p = 0.0, 1.0
        else:
            r, p = pearsonr(values, ratings)
        results[f'corr_{name}'] = float(r)
        results[f'pval_{name}'] = float(p)
        if abs(r) > best_abs_r:
            best_abs_r = abs(r)
            best_name = name

    results['best_predictor'] = best_name
    results['best_predictor_r'] = float(best_abs_r)

    return results
