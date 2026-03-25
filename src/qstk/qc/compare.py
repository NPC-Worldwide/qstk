"""Comparison utilities for quantum hardware vs LLM Bell test results.

Provides functions to align and compare S-values, expectation values,
and violation statistics from quantum circuits and LLM experiments
side-by-side.
"""

import numpy as np
from typing import Dict, Any, Optional, List

from .circuits import chsh_circuit, semantic_circuit
from .states import bell_state, werner_state
from .measure import chsh_s_value, chsh_expectation_values, chsh_expectation_values_density


def compare_quantum_llm(
    llm_expectation_values: Dict[str, float],
    quantum_state_type: str = "phi_plus",
    noise_p: Optional[float] = None,
    a0_angle: float = 0.0,
    a1_angle: float = np.pi / 2,
    b0_angle: float = np.pi / 4,
    b1_angle: float = 3 * np.pi / 4,
) -> Dict[str, Any]:
    """Compare LLM Bell test results against quantum mechanical predictions.

    Parameters
    ----------
    llm_expectation_values : dict
        Expectation values from LLM experiment with keys
        "A_B", "A_B_prime", "A_prime_B", "A_prime_B_prime".
    quantum_state_type : str
        Which Bell state for the quantum comparison.
    noise_p : float, optional
        If provided, uses a Werner state with this entanglement parameter
        instead of a pure Bell state.
    a0_angle, a1_angle, b0_angle, b1_angle : float
        Measurement angles for the quantum experiment.

    Returns
    -------
    dict with:
        - "llm_s": S-value from LLM
        - "quantum_s": S-value from quantum state
        - "classical_bound": 2.0
        - "tsirelson_bound": 2√2
        - "llm_violation": bool
        - "quantum_violation": bool
        - "llm_expectation_values": input LLM values
        - "quantum_expectation_values": quantum values
        - "expectation_deltas": per-pair differences
        - "s_delta": difference in S-values
        - "noise_p": Werner state parameter (if used)
        - "equivalent_werner_p": estimated Werner p that matches LLM S
    """
    from .operators import alice_operators, bob_operators

    # LLM S-value
    llm_ev = llm_expectation_values
    llm_s = (
        llm_ev.get("A_B", 0)
        - llm_ev.get("A_B_prime", 0)
        + llm_ev.get("A_prime_B", 0)
        + llm_ev.get("A_prime_B_prime", 0)
    )

    # Quantum S-value
    A0, A1 = alice_operators(a0_angle, a1_angle)
    B0, B1 = bob_operators(b0_angle, b1_angle)

    if noise_p is not None:
        bstate = bell_state(quantum_state_type)
        rho = werner_state(bstate, noise_p)
        q_ev = chsh_expectation_values_density(rho, A0, A1, B0, B1)
    else:
        bstate = bell_state(quantum_state_type)
        q_ev = chsh_expectation_values(bstate, A0, A1, B0, B1)

    quantum_s = (
        q_ev["A_B"]
        - q_ev["A_B_prime"]
        + q_ev["A_prime_B"]
        + q_ev["A_prime_B_prime"]
    )

    tsirelson = 2 * np.sqrt(2)

    # Per-pair deltas
    deltas = {}
    for key in ["A_B", "A_B_prime", "A_prime_B", "A_prime_B_prime"]:
        deltas[key] = llm_ev.get(key, 0) - q_ev.get(key, 0)

    # Estimate Werner p that would produce the LLM's S-value
    # For optimal CHSH operators, S(p) = p * 2√2
    # So p_equiv = |S_llm| / 2√2
    equiv_p = min(abs(llm_s) / tsirelson, 1.0)

    return {
        "llm_s": llm_s,
        "quantum_s": quantum_s,
        "classical_bound": 2.0,
        "tsirelson_bound": tsirelson,
        "llm_violation": abs(llm_s) > 2.0,
        "quantum_violation": abs(quantum_s) > 2.0,
        "llm_expectation_values": llm_ev,
        "quantum_expectation_values": q_ev,
        "expectation_deltas": deltas,
        "s_delta": llm_s - quantum_s,
        "noise_p": noise_p,
        "equivalent_werner_p": equiv_p,
    }


def sweep_werner_comparison(
    llm_s_value: float,
    p_values: Optional[List[float]] = None,
    state_type: str = "phi_plus",
) -> List[Dict[str, Any]]:
    """Sweep Werner noise parameter to find the quantum state matching an LLM S-value.

    Parameters
    ----------
    llm_s_value : float
        The S-value from an LLM experiment.
    p_values : list of float, optional
        Werner p values to test. Default: 0.0 to 1.0 in steps of 0.01.
    state_type : str
        Bell state type.

    Returns
    -------
    list of dicts with "p", "quantum_s", "delta" for each p.
    """
    if p_values is None:
        p_values = [i / 100 for i in range(101)]

    results = []
    for p in p_values:
        bstate = bell_state(state_type)
        rho = werner_state(bstate, p)
        qs = chsh_s_value(rho, is_density_matrix=True)
        results.append({
            "p": p,
            "quantum_s": qs,
            "delta": abs(llm_s_value - qs),
        })

    results.sort(key=lambda r: r["delta"])
    return results


def batch_compare(
    llm_results: List[Dict[str, float]],
    quantum_state_type: str = "phi_plus",
    noise_p: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare a batch of LLM trial results against a quantum reference.

    Parameters
    ----------
    llm_results : list of dicts
        Each dict has "A_B", "A_B_prime", "A_prime_B", "A_prime_B_prime".
    quantum_state_type : str
        Reference quantum state.
    noise_p : float, optional
        Werner noise parameter.

    Returns
    -------
    dict with summary statistics.
    """
    comparisons = [
        compare_quantum_llm(r, quantum_state_type, noise_p)
        for r in llm_results
    ]

    llm_s_values = [c["llm_s"] for c in comparisons]
    q_s = comparisons[0]["quantum_s"]

    return {
        "n_trials": len(comparisons),
        "quantum_s": q_s,
        "llm_s_mean": float(np.mean(llm_s_values)),
        "llm_s_std": float(np.std(llm_s_values)),
        "llm_s_min": float(np.min(llm_s_values)),
        "llm_s_max": float(np.max(llm_s_values)),
        "llm_violation_rate": sum(1 for c in comparisons if c["llm_violation"]) / len(comparisons),
        "mean_s_delta": float(np.mean([abs(c["s_delta"]) for c in comparisons])),
        "mean_equivalent_werner_p": float(np.mean([c["equivalent_werner_p"] for c in comparisons])),
        "comparisons": comparisons,
    }
