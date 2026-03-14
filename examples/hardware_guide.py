"""Quantum hardware guide — cost estimation, local testing, and cloud execution.

Demonstrates how to:
1. Estimate costs before running on real hardware
2. Run CHSH experiments on the local numpy simulator
3. Submit to IBM Quantum (requires qiskit-ibm-runtime + token)
4. Submit to Amazon Braket (requires amazon-braket-sdk + AWS credentials)
5. Compare results across backends
6. Track costs with the experiment log

Run: python examples/hardware_guide.py
"""

import numpy as np
from qstk.qc.hardware import (
    PRICING,
    estimate_cost,
    estimate_chsh_experiment_cost,
    print_cost_comparison,
    run_numpy,
    compare_backends,
    ExperimentLog,
    s_value_from_counts,
)


def demo_cost_estimation():
    """Show cost estimates for different backends and experiment sizes."""

    print("=" * 70)
    print("COST ESTIMATION — Know before you spend")
    print("=" * 70)

    # Basic comparison
    print_cost_comparison(n_shots=1024, n_circuits=4)

    # Scale up: 4 Bell states, 1024 shots each
    print("\n\nFull experiment: 4 Bell states x 4 CHSH settings x 1024 shots:")
    print(f"{'Backend':<25} {'Circuits':>10} {'Cost':>12}")
    print("-" * 50)
    for backend in ["numpy", "ibm_free", "ibm_paygo", "braket_rigetti",
                     "braket_iqm_garnet", "braket_ionq_forte"]:
        est = estimate_chsh_experiment_cost(backend, n_shots=1024, n_bell_states=4)
        print(f"{backend:<25} {est.n_circuits:>10} ${est.estimated_usd:>10.2f}")

    # Budget planning
    print("\n\nBudget planning — how many shots can you afford?")
    budgets = [0, 5, 20, 100]
    for budget in budgets:
        print(f"\n  Budget: ${budget}")
        for backend in ["ibm_free", "braket_rigetti", "braket_ionq_forte"]:
            p = PRICING[backend]
            if p["model"] == "per_minute" and p["rate"] == 0:
                print(f"    {backend}: ~10 min free / 28 days (enough for thousands of circuits)")
            elif p["model"] == "per_task_plus_shot":
                # How many 4-circuit experiments fit in budget?
                per_experiment = p["task"] * 4 + p["shot"] * 1024 * 4
                n_experiments = int(budget / per_experiment) if per_experiment > 0 else 0
                print(f"    {backend}: ~{n_experiments} experiments @ ${per_experiment:.2f} each")


def demo_numpy_simulator():
    """Run CHSH on the numpy simulator — always free, always available."""

    print("\n\n" + "=" * 70)
    print("NUMPY SIMULATOR — Free baseline")
    print("=" * 70)

    tsirelson = 2 * np.sqrt(2)

    # All 4 Bell states
    print("\nBell state CHSH S-values (exact + sampled):")
    print(f"{'State':<15} {'Exact S':>12} {'|S|>2':>8} {'Tsirelson %':>13}")
    print("-" * 50)

    for state in ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]:
        r = run_numpy(state, n_shots=8192, seed=42)
        frac = abs(r.s_value) / tsirelson * 100
        violation = "YES" if abs(r.s_value) > 2.0 else "no"
        print(f"{state:<15} {r.s_value:>+12.6f} {violation:>8} {frac:>12.1f}%")

    # Angle sweep on phi_plus
    print("\nAngle sweep — Bob's b1 from 0 to π (phi_plus):")
    print(f"{'b1 angle':>10} {'S-value':>12} {'|S|>2':>8}")
    print("-" * 32)
    for b1_deg in range(0, 181, 15):
        b1 = np.radians(b1_deg)
        r = run_numpy("phi_plus", n_shots=0, b1_angle=b1)
        violation = "YES" if abs(r.s_value) > 2.0 else "no"
        print(f"{b1_deg:>8}° {r.s_value:>+12.6f} {violation:>8}")


def demo_ibm_quantum():
    """Show how to run on IBM Quantum (requires setup)."""

    print("\n\n" + "=" * 70)
    print("IBM QUANTUM — Setup & Execution")
    print("=" * 70)

    print("""
    Setup (one-time):
    1. Create account at https://quantum.ibm.com
    2. Get your API token from the dashboard
    3. Install: pip install qiskit-ibm-runtime
    4. Save token:
       from qiskit_ibm_runtime import QiskitRuntimeService
       QiskitRuntimeService.save_account(
           channel="ibm_quantum",
           token="YOUR_TOKEN_HERE",
           overwrite=True,
       )

    Running an experiment:
    """)

    print("    from qstk.qc.hardware import run_ibm, ExperimentLog")
    print()
    print("    log = ExperimentLog('my_experiment.json')")
    print("    result = run_ibm(")
    print("        state_type='phi_plus',")
    print("        n_shots=1024,")
    print("        backend_name='ibm_brisbane',  # or None for least busy")
    print("        use_error_mitigation=True,")
    print("        log=log,")
    print("    )")
    print("    print(f'Hardware S-value: {result.s_value:.4f}')")
    print("    print(f'Theoretical S: {result.metadata[\"theoretical_s_value\"]:.4f}')")
    print("    log.print_summary()")

    # Check if qiskit-ibm-runtime is installed
    try:
        import qiskit_ibm_runtime
        print(f"\n    qiskit-ibm-runtime is installed (v{qiskit_ibm_runtime.__version__})")
        print("    Ready to run! Set your token with QiskitRuntimeService.save_account()")
    except ImportError:
        print("\n    qiskit-ibm-runtime NOT installed. Run: pip install qiskit-ibm-runtime")


def demo_braket():
    """Show how to run on Amazon Braket (requires setup)."""

    print("\n\n" + "=" * 70)
    print("AMAZON BRAKET — Setup & Execution")
    print("=" * 70)

    print("""
    Setup (one-time):
    1. AWS account with Braket access enabled
    2. Install: pip install amazon-braket-sdk
    3. Configure AWS CLI: aws configure
       (needs access key, secret key, region us-east-1)

    Running an experiment:
    """)

    print("    from qstk.qc.hardware import run_braket, ExperimentLog")
    print()
    print("    # Local simulator (free)")
    print("    result = run_braket(state_type='phi_plus', n_shots=1024, use_local=True)")
    print()
    print("    # Rigetti QPU (~$5 for 4 circuits x 1024 shots)")
    print("    result = run_braket(")
    print("        state_type='phi_plus',")
    print("        n_shots=1024,")
    print("        device_arn='arn:aws:braket:::device/qpu/rigetti/Ankaa-3',")
    print("    )")
    print()
    print("    # IonQ trapped ion (~$329 for 4 circuits x 1024 shots)")
    print("    result = run_braket(")
    print("        state_type='phi_plus',")
    print("        n_shots=1024,")
    print("        device_arn='arn:aws:braket:::device/qpu/ionq/Forte-1',")
    print("    )")

    # Check if braket is installed
    try:
        import braket
        print(f"\n    amazon-braket-sdk is installed")
        print("    Ready to run! Configure AWS credentials with: aws configure")
    except ImportError:
        print("\n    amazon-braket-sdk NOT installed. Run: pip install amazon-braket-sdk")


def demo_comparison():
    """Compare across backends."""

    print("\n\n" + "=" * 70)
    print("MULTI-BACKEND COMPARISON")
    print("=" * 70)

    print("\nRunning phi_plus CHSH on numpy (add 'ibm' or 'braket_local' when ready):\n")

    results = compare_backends(
        state_type="phi_plus",
        n_shots=4096,
        backends=["numpy"],
        seed=42,
    )

    print("""
    To compare with real hardware:

    results = compare_backends(
        state_type='phi_plus',
        n_shots=1024,
        backends=['numpy', 'ibm', 'braket_local'],
    )
    """)


def demo_paper_experiment():
    """Template for a QNLPAI 2026 paper experiment.

    The novel contribution: compare quantum hardware S-values against
    LLM semantic Bell test S-values. Nobody has done this yet.
    """

    print("\n\n" + "=" * 70)
    print("PAPER EXPERIMENT TEMPLATE — Quantum vs LLM S-value Comparison")
    print("=" * 70)

    print("""
    Experiment design for QNLPAI 2026:

    1. Encode word-sense correlations as quantum states
       - "bank" (financial / river) x "bat" (baseball / animal)
       - Amplitude encoding: |α₀₀|² = P(meaning_A1 ∧ meaning_B1), etc.
       - Run CHSH circuit on QPU → get hardware S-value

    2. Run same semantic task on LLMs
       - Give LLMs the ambiguous word pair with different persona prompts
       - Record interpretation correlations → compute LLM S-value

    3. Compare: does the quantum state encoding capture the same
       correlation structure that LLMs exhibit?

    Key predictions:
    - Entangled semantic states → S > 2 on QPU (quantum advantage region)
    - LLM correlations may also exceed classical bound (that's your 2025 result)
    - QPU noise degrades S (Werner state model) — calibrate via noise sweeps

    Budget estimate:
    """)

    # Real cost estimate for the experiment
    print("    Per word pair (4 Bell states x 1024 shots):")
    for backend in ["ibm_free", "ibm_paygo", "braket_rigetti", "braket_ionq_forte"]:
        est = estimate_chsh_experiment_cost(backend, n_shots=1024, n_bell_states=4)
        print(f"      {backend:<25} ${est.estimated_usd:>8.2f}  ({est.n_circuits} circuits)")

    print("\n    For 10 word pairs:")
    for backend in ["ibm_free", "ibm_paygo", "braket_rigetti"]:
        est = estimate_chsh_experiment_cost(backend, n_shots=1024, n_bell_states=4, n_angle_settings=10)
        print(f"      {backend:<25} ${est.estimated_usd:>8.2f}  ({est.n_circuits} circuits)")

    # Quick demo of semantic state encoding
    print("\n    Example semantic state encoding:")
    from qstk.qc import semantic_circuit

    # bank-bat: correlated meanings (financial-baseball more likely together)
    correlated = [0.6, 0.15, 0.15, 0.6]
    r = semantic_circuit(correlated, n_shots=4096)
    print(f"      Correlated:     amplitudes={correlated}  S={r['s_value']:.4f}  violation={r['violation']}")

    # Uniform: no correlation
    uniform = [0.5, 0.5, 0.5, 0.5]
    r = semantic_circuit(uniform, n_shots=4096)
    print(f"      Uniform:        amplitudes={uniform}  S={r['s_value']:.4f}  violation={r['violation']}")

    # Anti-correlated
    anti = [0.15, 0.6, 0.6, 0.15]
    r = semantic_circuit(anti, n_shots=4096)
    print(f"      Anti-correlated: amplitudes={anti}  S={r['s_value']:.4f}  violation={r['violation']}")

    print("""
    Code to run on real hardware:

    from qstk.qc.hardware import run_ibm, ExperimentLog

    log = ExperimentLog('qnlpai2026_results.json')

    # Encode semantic state as qubit amplitudes
    # (you'd compute these from word co-occurrence or LLM probabilities)
    amplitudes = [0.6, 0.15, 0.15, 0.6]  # correlated

    # For now: run the standard Bell state as a calibration
    result = run_ibm('phi_plus', n_shots=1024, log=log)
    print(f'Hardware S = {result.s_value:.4f}')
    print(f'Theory  S = {result.metadata["theoretical_s_value"]:.4f}')

    log.print_summary()
    """)


if __name__ == "__main__":
    demo_cost_estimation()
    demo_numpy_simulator()
    demo_ibm_quantum()
    demo_braket()
    demo_comparison()
    demo_paper_experiment()

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
    1. IBM Quantum (cheapest path to real hardware):
       pip install qiskit-ibm-runtime
       → Sign up at quantum.ibm.com → save token → run_ibm()

    2. Amazon Braket (more device choices):
       pip install amazon-braket-sdk
       → aws configure → run_braket()

    3. Start with ibm_free tier (10 min/month) for initial calibration
    4. Run semantic state encodings on QPU
    5. Compare with LLM S-values from your CHSH experiments
    """)
