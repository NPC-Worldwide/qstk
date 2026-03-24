"""NPCArray integration for vectorized Bell test experiments.

Uses npcpy's NPCArray formalism to run Bell test measurement settings
in parallel across model populations with lazy evaluation.
"""

import json
import random
from typing import List, Dict, Any, Optional, Callable

from npcpy.npc_array import NPCArray

from .chsh import (
    compute_chsh_products,
    compute_chsh_products_binary,
    calculate_s_value,
    calculate_expectation_values_direct,
    calculate_expectation_values_density_matrix,
    check_violation,
)
from .personas import generate_persona, get_persona_prompt


def create_bell_array(
    models: List[Dict[str, str]],
) -> NPCArray:
    """Create an NPCArray from a list of model configs for Bell tests.

    Parameters
    ----------
    models : list of dicts
        Each dict has "model" and "provider" keys, plus optional config.

    Returns
    -------
    NPCArray wrapping all model configs.

    Example
    -------
    >>> arr = create_bell_array([
    ...     {"model": "claude-haiku-4-5-20251001", "provider": "anthropic"},
    ...     {"model": "gemma3:12b", "provider": "ollama"},
    ... ])
    """
    return NPCArray.from_matrix(models)


def create_bell_meshgrid(
    models: List[str],
    providers: List[str],
    temperatures: List[float],
    top_ps: Optional[List[float]] = None,
    top_ks: Optional[List[int]] = None,
) -> NPCArray:
    """Create an NPCArray meshgrid spanning the Bell test parameter space.

    Parameters
    ----------
    models : list of str
        Model names.
    providers : list of str
        Provider names (matched 1:1 with models, or broadcast).
    temperatures : list of float
        Temperature values to sweep.
    top_ps : list of float, optional
        top_p values.
    top_ks : list of int, optional
        top_k values.

    Returns
    -------
    NPCArray with one spec per (model, temperature, top_p, top_k) combination.

    Example
    -------
    >>> grid = create_bell_meshgrid(
    ...     models=["gemma3:12b", "qwen3:0.6b"],
    ...     providers=["ollama"],
    ...     temperatures=[0.2, 1.0, 1.8],
    ...     top_ps=[0.37, 0.7, 1.0],
    ...     top_ks=[10, 50, 100],
    ... )
    """
    kwargs = {
        "model": models,
        "provider": providers,
        "temperature": temperatures,
    }
    if top_ps is not None:
        kwargs["top_p"] = top_ps
    if top_ks is not None:
        kwargs["top_k"] = top_ks
    return NPCArray.meshgrid(**kwargs)


def run_bell_trial_array(
    array: NPCArray,
    interpretation_prompt: str,
    settings: Dict[str, str],
    format: str = "json",
    **infer_kwargs,
) -> Dict[str, Any]:
    """Run one Bell trial across all 4 measurement settings using NPCArray.

    Infers all settings in parallel using the array's .infer().collect() pattern.

    Parameters
    ----------
    array : NPCArray
        The model array (typically 1 model for a single trial point).
    interpretation_prompt : str
        The prompt template for word interpretation (caller provides this).
    settings : dict
        Mapping of setting key to system message.
        Must have keys: "A", "A_prime", "B", "B_prime".
    format : str
        Response format ("json" by default).
    **infer_kwargs
        Additional kwargs passed to .infer() (e.g. temperature, max_tokens).

    Returns
    -------
    dict with keys:
        - "outcomes": dict of setting -> parsed response
        - "raw_responses": dict of setting -> raw string
        - "complete": bool
    """
    setting_keys = ["A", "A_prime", "B", "B_prime"]
    prompts = [interpretation_prompt] * len(setting_keys)

    # For each setting, we infer with a different system message
    # Since NPCArray.infer doesn't natively support per-prompt system messages,
    # we embed them in the prompts or make sequential calls
    raw_responses = {}
    outcomes = {}

    for key in setting_keys:
        system_msg = settings[key]
        result = array.infer(
            [interpretation_prompt],
            system_message=system_msg,
            format=format,
            **infer_kwargs,
        ).collect()
        raw = result.data.flatten()[0]
        raw_responses[key] = raw
        outcomes[key] = raw

    complete = all(key in outcomes and outcomes[key] for key in setting_keys)
    return {
        "outcomes": outcomes,
        "raw_responses": raw_responses,
        "complete": complete,
    }


def run_bell_grid_array(
    models: List[Dict[str, str]],
    word_pairs: List[Any],
    interpretation_prompt_fn: Callable,
    analysis_fn: Callable,
    setting_messages: Dict[str, str],
    param_grid: Optional[List[List]] = None,
    trials_per_point: int = 10,
    include_flipped: bool = True,
) -> List[Dict[str, Any]]:
    """Run a full Bell test grid sweep using NPCArray for parallel model inference.

    This is the NPCArray-powered equivalent of the grid sweep loop, but uses
    vectorized inference where possible.

    Parameters
    ----------
    models : list of dicts
        Model configs with "model" and "provider".
    word_pairs : list
        Word pair definitions.
    interpretation_prompt_fn : callable
        Function(sentence, word1, word2) -> str that builds the interpretation prompt.
        The caller provides this (prompts are NOT in qstk).
    analysis_fn : callable
        Function(interpretation, word_pair) -> dict that classifies A/B outcomes.
        The caller provides this.
    setting_messages : dict
        Mapping of setting key ("A", "A_prime", "B", "B_prime") to system messages.
    param_grid : list, optional
        Parameter grid [[temp, top_p, top_k], ...]. Uses defaults if None.
    trials_per_point : int
        Number of trials per grid point.
    include_flipped : bool
        Whether to test both word orderings.

    Returns
    -------
    list of trial result dicts suitable for DataFrame construction.
    """
    from .grid import get_param_grid, DEFAULT_PARAM_GRID

    all_results = []

    for model_cfg in models:
        model_name = model_cfg["model"]
        provider = model_cfg["provider"]
        grid = get_param_grid(provider, param_grid)

        # Create single-model array for this model
        arr = NPCArray.from_llms([model_name], providers=[provider])

        for wp in word_pairs:
            wp_label = f"{wp[0]['term']}/{wp[1]['term']}"
            flip_values = [False, True] if include_flipped else [False]

            for temp, top_p, top_k in grid:
                for flip in flip_values:
                    if flip:
                        word1, word2 = wp[1]["term"], wp[0]["term"]
                    else:
                        word1, word2 = wp[0]["term"], wp[1]["term"]

                    for trial_idx in range(trials_per_point):
                        sentence_template = random.choice([
                            "The {word1} was settled near the {word2}",
                            "The {word1} appeared right beside the {word2}",
                            "They noticed the {word1} close to the {word2}",
                        ])
                        sentence = sentence_template.format(word1=word1, word2=word2)
                        prompt = interpretation_prompt_fn(sentence, word1, word2)

                        infer_kwargs = {"temperature": temp, "max_tokens": 2000}
                        if top_p is not None:
                            infer_kwargs["top_p"] = top_p
                        if top_k is not None:
                            infer_kwargs["top_k"] = top_k

                        outcomes = {}
                        for setting_key in ["A", "A_prime", "B", "B_prime"]:
                            system_msg = setting_messages[setting_key]
                            try:
                                result = arr.infer(
                                    [prompt],
                                    system_message=system_msg,
                                    format="json",
                                    **infer_kwargs,
                                ).collect()
                                interp = result.data.flatten()[0]
                                if isinstance(interp, str):
                                    try:
                                        interp = json.loads(interp)
                                    except json.JSONDecodeError:
                                        continue
                                if interp:
                                    analysis = analysis_fn(interp, wp)
                                    if analysis:
                                        outcomes[setting_key] = analysis
                            except Exception as e:
                                print(f"  Error {setting_key}: {e}")

                        complete = all(k in outcomes for k in ["A", "A_prime", "B", "B_prime"])
                        chsh = {}
                        if complete:
                            chsh = compute_chsh_products(outcomes)

                        all_results.append({
                            "trial": trial_idx,
                            "model": model_name,
                            "word_pair": wp_label,
                            "first_word": word1,
                            "second_word": word2,
                            "flipped": flip,
                            "temperature": temp,
                            "top_p": top_p if top_p is not None else 0,
                            "top_k": top_k if top_k is not None else 0,
                            "complete": complete,
                            **{k: chsh.get(k) for k in ["A_B", "A_B_prime", "A_prime_B", "A_prime_B_prime"]},
                        })

    return all_results


def infer_and_classify(
    array: NPCArray,
    prompts: List[str],
    classify_fn: Callable,
    **infer_kwargs,
) -> List[Any]:
    """Infer across a model array and classify results.

    Convenience function: runs inference on all prompts across all models,
    then applies a classification function to each response.

    Parameters
    ----------
    array : NPCArray
        Model array.
    prompts : list of str
        Prompts to infer.
    classify_fn : callable
        Function(response) -> classification result.
    **infer_kwargs
        Passed to .infer().

    Returns
    -------
    list of classified results (flattened).
    """
    results = array.infer(prompts, **infer_kwargs).map(classify_fn).collect()
    return results.flatten()
