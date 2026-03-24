"""CHSH Bell test math: S-value, expectation values, product computations."""

import math
from collections import Counter
from typing import List, Dict, Optional

import numpy as np


def compute_chsh_products(outcomes: Dict[str, List[float]]) -> Dict[str, float]:
    """Compute CHSH product terms from measurement outcomes.

    Parameters
    ----------
    outcomes : dict
        Keys are setting labels ("A", "A_prime", "B", "B_prime"), values are
        outcome vectors (list of +1/-1 values or floats to be normalized).

    Returns
    -------
    dict with keys "A_B", "A_B_prime", "A_prime_B", "A_prime_B_prime"
    containing the dot products of the normalized outcome vectors.
    """
    def _norm(v):
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n > 0 else v

    required = ["A", "A_prime", "B", "B_prime"]
    if not all(k in outcomes for k in required):
        return {}

    A = _norm(outcomes["A"])
    Ap = _norm(outcomes["A_prime"])
    B = _norm(outcomes["B"])
    Bp = _norm(outcomes["B_prime"])

    return {
        "A_B": sum(a * b for a, b in zip(A, B)),
        "A_B_prime": sum(a * b for a, b in zip(A, Bp)),
        "A_prime_B": sum(a * b for a, b in zip(Ap, B)),
        "A_prime_B_prime": sum(a * b for a, b in zip(Ap, Bp)),
    }


def compute_chsh_products_binary(outcomes: Dict[str, int]) -> Dict[str, int]:
    """Compute CHSH product terms from binary (+1/-1) scalar outcomes.

    Parameters
    ----------
    outcomes : dict
        Keys are setting labels ("A", "A_prime", "B", "B_prime"),
        values are +1 or -1 integers.

    Returns
    -------
    dict with product terms as integers.
    """
    return {
        "A_B": outcomes["A"] * outcomes["B"],
        "A_B_prime": outcomes["A"] * outcomes["B_prime"],
        "A_prime_B": outcomes["A_prime"] * outcomes["B"],
        "A_prime_B_prime": outcomes["A_prime"] * outcomes["B_prime"],
    }


def calculate_s_value(expectation_values: Dict[str, float]) -> float:
    """Calculate CHSH S value from expectation values.

    S = E(A,B) - E(A,B') + E(A',B) + E(A',B')

    Classical bound: |S| <= 2
    Quantum (Tsirelson) bound: |S| <= 2*sqrt(2) ~ 2.828
    """
    return (
        expectation_values.get("A_B", 0.0)
        - expectation_values.get("A_B_prime", 0.0)
        + expectation_values.get("A_prime_B", 0.0)
        + expectation_values.get("A_prime_B_prime", 0.0)
    )


def check_violation(s_value: float, bound: float = 2.0) -> bool:
    """Check if S-value violates the classical CHSH bound."""
    return abs(s_value) > bound


def calculate_expectation_values_direct(
    all_trial_product_terms: List[Dict[str, float]],
) -> Dict[str, float]:
    """Calculate expectation values by simple averaging of product terms.

    This is the direct averaging method used in bell_test.py and bell_test_simple.py.

    Parameters
    ----------
    all_trial_product_terms : list of dicts
        Each dict has keys "A_B", "A_B_prime", "A_prime_B", "A_prime_B_prime"
        with numeric values.

    Returns
    -------
    dict of averaged expectation values.
    """
    if not all_trial_product_terms:
        return {"A_B": 0.0, "A_B_prime": 0.0, "A_prime_B": 0.0, "A_prime_B_prime": 0.0}

    sum_products = Counter()
    n = len(all_trial_product_terms)
    for trial in all_trial_product_terms:
        for key, val in trial.items():
            sum_products[key] += val

    standard_keys = ["A_B", "A_B_prime", "A_prime_B", "A_prime_B_prime"]
    return {k: sum_products.get(k, 0) / n for k in standard_keys}


def calculate_expectation_values_density_matrix(
    all_trial_product_terms: List[Dict[str, float]],
) -> Dict[str, float]:
    """Calculate expectation values using density matrix formalism.

    This is the density matrix method used in bell_test_grid_sweep.py.
    Constructs a mixed state density matrix from all trial product vectors,
    then computes expectation values of projection observables.

    Parameters
    ----------
    all_trial_product_terms : list of dicts
        Each dict has keys "A_B", "A_B_prime", "A_prime_B", "A_prime_B_prime"
        with float values.

    Returns
    -------
    dict of expectation values.
    """
    if not all_trial_product_terms:
        return {"A_B": 0.0, "A_B_prime": 0.0, "A_prime_B": 0.0, "A_prime_B_prime": 0.0}

    complete = [
        t for t in all_trial_product_terms
        if all(
            k in t and isinstance(t[k], (int, float))
            for k in ["A_B", "A_B_prime", "A_prime_B", "A_prime_B_prime"]
        )
    ]
    if not complete:
        return {"A_B": 0.0, "A_B_prime": 0.0, "A_prime_B": 0.0, "A_prime_B_prime": 0.0}

    quantum_states = []
    for trial in complete:
        state = [trial["A_B"], trial["A_B_prime"], trial["A_prime_B"], trial["A_prime_B_prime"]]
        norm = math.sqrt(sum(x * x for x in state))
        if norm > 0:
            quantum_states.append([x / norm for x in state])

    density_matrix = np.zeros((4, 4), dtype=complex)
    for state in quantum_states:
        state_array = np.array(state)
        density_matrix += np.outer(state_array, state_array.conj())
    if quantum_states:
        density_matrix /= len(quantum_states)

    observables = {
        "A_B": np.diag([1, 0, 0, 0]),
        "A_B_prime": np.diag([0, 1, 0, 0]),
        "A_prime_B": np.diag([0, 0, 1, 0]),
        "A_prime_B_prime": np.diag([0, 0, 0, 1]),
    }

    return {
        key: float(np.real(np.trace(density_matrix @ op))) * 4
        for key, op in observables.items()
    }
