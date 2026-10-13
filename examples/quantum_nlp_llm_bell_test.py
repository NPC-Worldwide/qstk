"""Quantum NLP: LLM-driven Bell test with quantum circuit comparison.

This example shows the hybrid quantum-classical workflow:

  1. Define an ambiguous word pair and two persona settings for each party.
  2. (Optional) Use an LLM to generate +1/-1 interpretations for the
     four measurement settings A, A', B, B'.
  3. Compute the LLM S-value from those outcomes.
  4. Build the matching quantum state and run it through Qiskit Aer.
  5. Compare the LLM S-value to the quantum circuit S-value.

For the interview demo, the LLM is mocked out with deterministic outcomes
so the script runs without API keys. Replace the mock outcomes with a real
LLM call when you have npcpy configured.
"""

import numpy as np
from qstk.chsh import compute_chsh_products_binary, calculate_s_value, check_violation
from qstk.qc.compare import compare_quantum_llm, sweep_werner_comparison
from qstk.qc.qiskit_runner import run_qiskit_simulator


WORD_PAIR = ("bank", "bat")
MEANINGS = {
    "bank": ["financial institution", "river bank"],
    "bat": ["baseball bat", "flying animal"],
}


# Mock LLM outcomes: these would come from prompting an LLM with
# different personas and asking it to pick meaning A or B.
MOCK_LLM_OUTCOMES = {
    "A": 1,
    "A_prime": -1,
    "B": 1,
    "B_prime": -1,
}


def run_llm_bell_test():
    products = compute_chsh_products_binary(MOCK_LLM_OUTCOMES)
    ev = {
        "A_B": products["A_B"],
        "A_B_prime": products["A_B_prime"],
        "A_prime_B": products["A_prime_B"],
        "A_prime_B_prime": products["A_prime_B_prime"],
    }
    s_llm = calculate_s_value(ev)
    return s_llm, ev


def main():
    print("=" * 70)
    print("Quantum NLP: LLM-driven Bell test vs Qiskit quantum circuit")
    print("=" * 70)
    print()

    s_llm, ev = run_llm_bell_test()
    print(f"Word pair: {WORD_PAIR[0]} × {WORD_PAIR[1]}")
    print(f"LLM expectation values: {ev}")
    print(f"LLM S-value: {s_llm:+.4f}")
    print(f"LLM violation of |S|>2: {check_violation(s_llm)}")
    print()

    comparison = compare_quantum_llm(ev, quantum_state_type="phi_plus")
    print("Quantum reference (phi_plus):")
    print(f"  Quantum S = {comparison['quantum_s']:+.4f}")
    print(f"  Equivalent Werner p = {comparison['equivalent_werner_p']:.3f}")
    print()

    print("Running the quantum reference on Qiskit Aer...")
    qiskit_result = run_qiskit_simulator(
        state_type="phi_plus",
        n_shots=8192,
        seed=42,
    )
    print(f"  Qiskit Aer S = {qiskit_result.s_value:+.4f}")
    print(f"  Violation: {abs(qiskit_result.s_value) > 2.0}")
    print()

    print("=" * 70)
    print("Discussion point")
    print("=" * 70)
    print("This is the integration surface between LLM and quantum computing:")
    print("language models produce correlation hypotheses; quantum circuits")
    print("measure whether those correlations are classically explainable.")


if __name__ == "__main__":
    main()
