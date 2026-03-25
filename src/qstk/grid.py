"""Parameter grid construction and sweep utilities for Bell test experiments."""

from itertools import product
from typing import List, Dict, Any, Optional, Tuple


# Standard parameter grid from ket-nlp experiments:
# [temperature, top_p, top_k]
DEFAULT_PARAM_GRID = [
    [0.2, 0.37, 10],    # Low temp, low p, low k
    [1.0, 0.7, 50],     # Medium
    [1.8, 1.0, 100],    # High temp, high p, high k
]

# Providers that restrict parameter sweeps
RESTRICTED_PROVIDERS = {
    "anthropic": [[1.0, None, None]],
}


def get_param_grid(
    provider: str,
    custom_grid: Optional[List[List]] = None,
) -> List[List]:
    """Return the appropriate parameter grid for a given provider.

    Parameters
    ----------
    provider : str
        The LLM provider name (e.g. "ollama", "anthropic", "gemini").
    custom_grid : list, optional
        Override the default grid with a custom one.

    Returns
    -------
    list of [temperature, top_p, top_k] triples.
    """
    if custom_grid is not None:
        return custom_grid
    return RESTRICTED_PROVIDERS.get(provider, DEFAULT_PARAM_GRID)


def build_sweep_configs(
    models: List[Dict[str, str]],
    word_pairs: List[Any],
    param_grid: Optional[List[List]] = None,
    include_flipped: bool = True,
    trials_per_point: int = 10,
    previous_counts: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """Build the full list of sweep configurations with remaining trial counts.

    Parameters
    ----------
    models : list of dicts
        Each dict has "model" and "provider" keys.
    word_pairs : list
        Word pair definitions (structure from the experiment).
    param_grid : list, optional
        Override parameter grid. If None, uses per-provider defaults.
    include_flipped : bool
        Whether to include flipped word orderings.
    trials_per_point : int
        Target number of trials per grid point.
    previous_counts : dict, optional
        Counts from a previous run (from load_previous_trials).

    Returns
    -------
    list of config dicts, each containing model, provider, word_pair, params,
    flipped, trials_remaining, and a grid_key tuple.
    """
    if previous_counts is None:
        previous_counts = {}

    configs = []
    flip_values = [False, True] if include_flipped else [False]

    for model_cfg in models:
        model = model_cfg["model"]
        provider = model_cfg["provider"]
        grid = get_param_grid(provider, param_grid)

        for wp in word_pairs:
            wp_label = f"{wp[0]['term']}/{wp[1]['term']}"
            for temp, top_p, top_k in grid:
                for flip in flip_values:
                    grid_key = (
                        model,
                        wp_label,
                        flip,
                        float(temp),
                        float(top_p) if top_p is not None else 0.0,
                        int(top_k) if top_k is not None else 0,
                    )
                    already_done = previous_counts.get(grid_key, 0)
                    remaining = max(0, trials_per_point - already_done)

                    configs.append({
                        "model": model,
                        "provider": provider,
                        "word_pair": wp,
                        "word_pair_label": wp_label,
                        "temperature": temp,
                        "top_p": top_p,
                        "top_k": top_k,
                        "flipped": flip,
                        "trials_remaining": remaining,
                        "trials_done": already_done,
                        "grid_key": grid_key,
                    })

    return configs


def sweep_summary(configs: List[Dict[str, Any]]) -> Dict[str, int]:
    """Summarize a sweep configuration list.

    Returns
    -------
    dict with total_points, total_remaining, total_skipped.
    """
    total = len(configs)
    remaining = sum(c["trials_remaining"] for c in configs)
    skipped = sum(c["trials_done"] for c in configs)
    return {
        "total_points": total,
        "total_remaining": remaining,
        "total_skipped": skipped,
    }
