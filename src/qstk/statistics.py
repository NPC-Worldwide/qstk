"""Combinatorial significance testing for Bell test theme experiments."""

import math
from typing import List, Tuple


def get_combinations(n: int, k: int) -> int:
    """Compute binomial coefficient C(n, k)."""
    try:
        return math.comb(n, k)
    except AttributeError:
        # Fallback for Python < 3.8
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        if k > n // 2:
            k = n - k
        res = 1
        for i in range(k):
            res = res * (n - i) // (i + 1)
        return res


def calculate_agreement_significance_combinatorial(
    vector_x: List[int],
    vector_y: List[int],
    total_themes_available: int,
) -> Tuple[float, float]:
    """Calculate Normalized Agreement Score (NAS) and p-value for two theme vectors.

    Uses the hypergeometric distribution to compute the probability of
    observing at least as many shared theme selections by chance.

    Parameters
    ----------
    vector_x, vector_y : list of int
        Binary (0/1) theme assignment vectors of length ``total_themes_available``.
    total_themes_available : int
        Total number of themes in the pool.

    Returns
    -------
    (nas_score, p_value) tuple.
    """
    if not isinstance(vector_x, list) or not isinstance(vector_y, list):
        return 0.0, 1.0

    clean_vec_x = [
        int(v) if isinstance(v, (int, float, str)) and str(v).isdigit() else 0
        for v in vector_x
    ]
    clean_vec_y = [
        int(v) if isinstance(v, (int, float, str)) and str(v).isdigit() else 0
        for v in vector_y
    ]

    if total_themes_available > 0:
        if len(clean_vec_x) != total_themes_available:
            clean_vec_x = (clean_vec_x + [0] * total_themes_available)[:total_themes_available]
        if len(clean_vec_y) != total_themes_available:
            clean_vec_y = (clean_vec_y + [0] * total_themes_available)[:total_themes_available]
    elif len(clean_vec_x) != len(clean_vec_y):
        return 0.0, 1.0

    if not clean_vec_x and not clean_vec_y and total_themes_available == 0:
        return 1.0, 1.0

    K_selected_by_A = sum(clean_vec_x)
    N_selected_by_B = sum(clean_vec_y)
    s_observed_shared = sum(a * b for a, b in zip(clean_vec_x, clean_vec_y))

    avg_selected_count = (K_selected_by_A + N_selected_by_B) / 2
    nas_score = 0.0
    if avg_selected_count > 0:
        nas_score = s_observed_shared / avg_selected_count
    elif total_themes_available > 0 and K_selected_by_A == 0 and N_selected_by_B == 0:
        nas_score = 1.0

    if total_themes_available == 0:
        if K_selected_by_A == 0 and N_selected_by_B == 0:
            return 1.0, 1.0
        return nas_score, 1.0

    if (
        K_selected_by_A < 0
        or N_selected_by_B < 0
        or s_observed_shared < 0
        or K_selected_by_A > total_themes_available
        or N_selected_by_B > total_themes_available
        or s_observed_shared > K_selected_by_A
        or s_observed_shared > N_selected_by_B
    ):
        return nas_score, 1.0

    total_possible = get_combinations(total_themes_available, N_selected_by_B)
    if total_possible == 0:
        return nas_score, 1.0

    favorable = 0
    max_possible_s = min(K_selected_by_A, N_selected_by_B)
    for s_current in range(s_observed_shared, max_possible_s + 1):
        ways_shared = get_combinations(K_selected_by_A, s_current)
        remaining_needed = N_selected_by_B - s_current
        not_selected_by_A = total_themes_available - K_selected_by_A
        if remaining_needed < 0 or remaining_needed > not_selected_by_A:
            ways_remaining = 0
        else:
            ways_remaining = get_combinations(not_selected_by_A, remaining_needed)
        favorable += ways_shared * ways_remaining

    p_value = favorable / total_possible if total_possible > 0 else 1.0
    return nas_score, p_value
