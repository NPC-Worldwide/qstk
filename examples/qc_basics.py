"""qstk.qc basics — Bell states, CHSH violation, Werner noise, semantic states.

Run: python examples/qc_basics.py
"""

import numpy as np
from qstk.qc import (
    # State preparation
    bell_state, werner_state, parameterized_entangled_state, semantic_state,
    # Operators
    alice_operators, bob_operators, pauli_x, pauli_z,
    # Measurement
    chsh_s_value, chsh_expectation_values, measure_state,
    entanglement_entropy, concurrence, fidelity,
    # Circuits
    bell_circuit, chsh_circuit, semantic_circuit,
    # Comparison
    compare_quantum_llm, sweep_werner_comparison,
)


# ── 1. Bell States ──────────────────────────────────────────────────────

print("=" * 60)
print("1. THE FOUR BELL STATES")
print("=" * 60)

for name in ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]:
    psi = bell_state(name)
    print(f"\n  |{name}> = {np.round(psi, 3)}")
    print(f"    Concurrence:  {concurrence(psi):.3f}")   # 1.0 = maximally entangled
    print(f"    Ent. entropy: {entanglement_entropy(psi):.3f} bits")

    # Simulate measuring it 1000 times
    counts = measure_state(psi, n_shots=1000, seed=42)
    print(f"    Measurements: {counts}")


# ── 2. Bell Circuit (gate-level) ────────────────────────────────────────

print("\n" + "=" * 60)
print("2. BELL CIRCUIT (gate-by-gate)")
print("=" * 60)

result = bell_circuit("phi_plus", n_shots=2000, seed=7)
print(f"\n  Gates applied:  {result['circuit_ops']}")
print(f"  Final state:    {np.round(result['state'], 3)}")
print(f"  Counts (2000):  {result['counts']}")
print(f"  ~50/50 |00> and |11> as expected for |Phi+>")


# ── 3. CHSH Violation ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("3. CHSH BELL TEST")
print("=" * 60)
print("""
  Classical bound:  |S| <= 2
  Tsirelson bound:  |S| <= 2*sqrt(2) ~ 2.828
""")

# Run a full CHSH experiment on |Phi+>
chsh = chsh_circuit("phi_plus", n_shots=10000, seed=42)
print(f"  S-value:            {chsh['s_value']:.4f}")
print(f"  Violates classical: {chsh['violation']}")
print(f"  Tsirelson fraction: {chsh['tsirelson_fraction']:.4f}")
print(f"\n  Expectation values:")
for k, v in chsh["expectation_values"].items():
    print(f"    E({k}) = {v:+.4f}")

print(f"\n  Sampled measurement outcomes (10k shots each):")
for setting, data in chsh["measurement_counts"].items():
    print(f"    {setting}: +1={data['+1']}, -1={data['-1']}, mean={data['mean']:+.4f}")


# ── 4. How S-value depends on measurement angles ───────────────────────

print("\n" + "=" * 60)
print("4. ANGLE SWEEP — finding maximum violation")
print("=" * 60)

psi = bell_state("phi_plus")
print(f"\n  Fixing Alice at a0=0, a1=pi/2")
print(f"  Sweeping Bob's b1 angle:\n")

for b1_deg in range(0, 361, 30):
    b1 = np.radians(b1_deg)
    A0, A1 = alice_operators(0, np.pi / 2)
    B0, B1 = bob_operators(np.pi / 4, b1)
    s = chsh_s_value(psi, A0, A1, B0, B1)
    bar = "#" * int(abs(s) * 8)
    marker = " <-- MAX" if abs(abs(s) - 2 * np.sqrt(2)) < 0.01 else ""
    print(f"    b1={b1_deg:3d} deg  S={s:+.4f}  |{'':2s}{bar}{marker}")


# ── 5. Werner States — noise degrades violation ─────────────────────────

print("\n" + "=" * 60)
print("5. WERNER STATES — entanglement vs noise")
print("=" * 60)
print(f"\n  rho(p) = p*|Phi+><Phi+| + (1-p)*I/4")
print(f"  Violation threshold: p > 1/sqrt(2) ~ 0.707\n")

psi = bell_state("phi_plus")
for p in [1.0, 0.9, 0.8, 0.71, 0.707, 0.7, 0.5, 0.2, 0.0]:
    rho = werner_state(psi, p)
    s = chsh_s_value(rho, is_density_matrix=True)
    violation = "VIOLATES" if abs(s) > 2 else "classical"
    bar = "#" * int(abs(s) * 8)
    print(f"    p={p:.3f}  S={s:+.4f}  {violation:10s}  {bar}")


# ── 6. Tunable entanglement ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("6. PARAMETERIZED ENTANGLEMENT")
print("=" * 60)
print(f"\n  |psi(theta)> = cos(theta)|00> + sin(theta)|11>")
print(f"  theta=0 -> separable, theta=pi/4 -> maximally entangled\n")

for theta_deg in range(0, 91, 15):
    theta = np.radians(theta_deg)
    psi = parameterized_entangled_state(theta)
    c = concurrence(psi)
    s = chsh_s_value(psi)
    ent = entanglement_entropy(psi)
    print(f"    theta={theta_deg:2d} deg  C={c:.3f}  S={s:+.4f}  entropy={ent:.3f} bits")


# ── 7. Semantic State — word-sense CHSH ─────────────────────────────────

print("\n" + "=" * 60)
print('7. SEMANTIC STATE — "bank" x "bat" word sense CHSH')
print("=" * 60)
print("""
  Two ambiguous words, each with 2 meanings:
    bank: financial(0) / river(1)
    bat:  baseball(0) / animal(1)

  Basis: |bank_meaning, bat_meaning>
  If meanings are correlated -> entanglement -> CHSH violation
""")

# Correlated meanings: financial-baseball go together, river-animal go together
correlated = [0.7, 0.05, 0.05, 0.7]  # |00> and |11> dominate
result_c = semantic_circuit(correlated, seed=42)
print(f"  Correlated senses:   amps={correlated}")
print(f"    S={result_c['s_value']:+.4f}  violation={result_c['violation']}")
print(f"    concurrence={result_c['concurrence']:.3f}")

# Uncorrelated meanings: all equally likely
uniform = [0.5, 0.5, 0.5, 0.5]
result_u = semantic_circuit(uniform, seed=42)
print(f"\n  Uniform (no correlation): amps={uniform}")
print(f"    S={result_u['s_value']:+.4f}  violation={result_u['violation']}")
print(f"    concurrence={result_u['concurrence']:.3f}")

# Anti-correlated: financial-animal, river-baseball
anti = [0.05, 0.7, 0.7, 0.05]  # |01> and |10> dominate
result_a = semantic_circuit(anti, seed=42)
print(f"\n  Anti-correlated:     amps={anti}")
print(f"    S={result_a['s_value']:+.4f}  violation={result_a['violation']}")
print(f"    concurrence={result_a['concurrence']:.3f}")


# ── 8. Compare quantum prediction vs LLM results ───────────────────────

print("\n" + "=" * 60)
print("8. QUANTUM vs LLM COMPARISON")
print("=" * 60)
print("""
  Suppose an LLM Bell test produced these expectation values:
""")

llm_results = {
    "A_B": 0.62,
    "A_B_prime": -0.58,
    "A_prime_B": 0.55,
    "A_prime_B_prime": 0.49,
}

comp = compare_quantum_llm(llm_results)
print(f"  LLM expectation values: {llm_results}")
print(f"  LLM S-value:       {comp['llm_s']:.4f}  (violation: {comp['llm_violation']})")
print(f"  Quantum S-value:   {comp['quantum_s']:.4f}  (violation: {comp['quantum_violation']})")
print(f"  S delta:           {comp['s_delta']:.4f}")
print(f"  Equivalent Werner p: {comp['equivalent_werner_p']:.4f}")
print(f"    (LLM behaves like a Werner state with this much entanglement)")

# Find the best-matching Werner state
sweep = sweep_werner_comparison(comp["llm_s"])
best = sweep[0]
print(f"\n  Best Werner match: p={best['p']:.2f} gives S={best['quantum_s']:.4f} (delta={best['delta']:.4f})")


# ── 9. State fidelity ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("9. STATE FIDELITY")
print("=" * 60)

phi_plus = bell_state("phi_plus")
phi_minus = bell_state("phi_minus")
psi_plus = bell_state("psi_plus")
slightly_off = parameterized_entangled_state(np.pi / 4 - 0.1)

print(f"\n  F(Phi+, Phi+)       = {fidelity(phi_plus, phi_plus):.4f}")
print(f"  F(Phi+, Phi-)       = {fidelity(phi_plus, phi_minus):.4f}")
print(f"  F(Phi+, Psi+)       = {fidelity(phi_plus, psi_plus):.4f}")
print(f"  F(Phi+, slightly_off) = {fidelity(phi_plus, slightly_off):.4f}")


print("\n" + "=" * 60)
print("Done! All simulations use pure numpy — no quantum hardware needed.")
print("For real hardware, use to_qiskit() or to_cirq() on circuit_ops.")
print("=" * 60)
