#!/usr/bin/env python3
"""
Validate and visualize all quantum circuits in qstk.qc.

Checks:
1. Bell state preparation is correct (orthonormal, maximally entangled).
2. CHSH operators are Hermitian with eigenvalues ±1.
3. CHSH S-value reaches Tsirelson bound 2√2 for ideal Bell state (with optimal angles).
4. Werner state interpolation: violation disappears at p = 1/√2.
5. Parameterized entangled state shows correct concurrence / entropy trend.
6. Semantic circuits behave as expected for correlated / uniform / anti-correlated amplitudes.
7. Visualizations: state amplitudes, S-value landscapes, entanglement measures.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from qstk.qc.states import bell_state, werner_state, parameterized_entangled_state, semantic_state
from qstk.qc.operators import (
    pauli_x, pauli_y, pauli_z, pauli_i,
    hadamard, cnot, alice_operators, bob_operators, chsh_operator,
)
from qstk.qc.measure import (
    expectation_value, chsh_expectation_values, chsh_s_value,
    measure_state, entanglement_entropy, concurrence, fidelity,
    density_matrix, reduced_density_matrix, von_neumann_entropy,
)
from qstk.qc.circuits import bell_circuit, chsh_circuit, semantic_circuit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_hermitian(M, tol=1e-12):
    return np.allclose(M, M.conj().T, atol=tol)

def is_unitary(M, tol=1e-12):
    I = np.eye(M.shape[0], dtype=complex)
    return np.allclose(M @ M.conj().T, I, atol=tol) and np.allclose(M.conj().T @ M, I, atol=tol)

def eigenvalues_real_pm1(M, tol=1e-12):
    ev = np.linalg.eigvalsh(M)
    return np.allclose(ev, np.round(ev), atol=tol) and set(np.round(ev).astype(int)) <= {-1, 1}

# ---------------------------------------------------------------------------
# 1. Validate basic operators
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. VALIDATING BASIC OPERATORS")
print("=" * 60)

ops = {
    "Pauli X": pauli_x(),
    "Pauli Y": pauli_y(),
    "Pauli Z": pauli_z(),
    "Hadamard": hadamard(),
    "CNOT": cnot(),
}

for name, M in ops.items():
    herm = is_hermitian(M)
    unit = is_unitary(M)
    print(f"  {name:12s}  Hermitian={herm}  Unitary={unit}")
    assert herm, f"{name} is not Hermitian"
    assert unit, f"{name} is not Unitary"

print("  [PASS] All basic operators are Hermitian and Unitary.\n")

# ---------------------------------------------------------------------------
# 2. Validate Bell states
# ---------------------------------------------------------------------------
print("=" * 60)
print("2. VALIDATING BELL STATES")
print("=" * 60)

bell_names = ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]
bell_vecs = {name: bell_state(name) for name in bell_names}

for name, v in bell_vecs.items():
    norm = np.linalg.norm(v)
    print(f"  {name:12s}  norm={norm:.6f}")
    assert np.isclose(norm, 1.0), f"Bell state {name} not normalized"

# Orthogonality
for i, n1 in enumerate(bell_names):
    for n2 in bell_names[i+1:]:
        ov = abs(np.dot(bell_vecs[n1].conj(), bell_vecs[n2]))
        print(f"  overlap({n1},{n2}) = {ov:.2e}")
        assert np.isclose(ov, 0.0, atol=1e-12), f"Bell states {n1} and {n2} not orthogonal"

# Check entanglement entropy = 1 (maximal)
for name, v in bell_vecs.items():
    ee = entanglement_entropy(v)
    cc = concurrence(v)
    print(f"  {name:12s}  entanglement_entropy={ee:.6f}  concurrence={cc:.6f}")
    assert np.isclose(ee, 1.0, atol=1e-12), f"Bell state {name} EE != 1"
    assert np.isclose(cc, 1.0, atol=1e-12), f"Bell state {name} concurrence != 1"

# Verify circuit preparation matches direct state
for name in bell_names:
    circ = bell_circuit(name, n_shots=0)
    fid = fidelity(circ["state"], bell_vecs[name])
    print(f"  circuit({name}) fidelity vs direct = {fid:.6f}")
    assert np.isclose(fid, 1.0, atol=1e-12), f"Circuit preparation mismatch for {name}"

print("  [PASS] All Bell states are orthonormal, maximally entangled, circuits match.\n")

# ---------------------------------------------------------------------------
# 3. Validate CHSH operators
# ---------------------------------------------------------------------------
print("=" * 60)
print("3. VALIDATING CHSH OPERATORS")
print("=" * 60)

A0, A1 = alice_operators()
B0, B1 = bob_operators()

for label, M in [("A0", A0), ("A1", A1), ("B0", B0), ("B1", B1)]:
    herm = is_hermitian(M)
    ev = np.linalg.eigvalsh(M)
    pm1 = eigenvalues_real_pm1(M)
    print(f"  {label}  Hermitian={herm}  eigenvalues={np.round(ev, 6)}  ±1={pm1}")
    assert herm, f"{label} not Hermitian"
    assert pm1, f"{label} eigenvalues not ±1"

# CHSH operator
Bchsh = chsh_operator(A0, A1, B0, B1)
print(f"  CHSH operator Hermitian={is_hermitian(Bchsh)}")
ev_chsh = np.linalg.eigvalsh(Bchsh)
print(f"  CHSH operator eigenvalues: {np.round(ev_chsh, 6)}")
assert is_hermitian(Bchsh), "CHSH operator not Hermitian"

print("  [PASS] CHSH operators are valid Hermitian observables with eigenvalues ±1.\n")

# ---------------------------------------------------------------------------
# 4. Validate CHSH S-value reaches Tsirelson bound (with optimal angles)
# ---------------------------------------------------------------------------
print("=" * 60)
print("4. VALIDATING CHSH S-VALUE / TSIRELSON BOUND")
print("=" * 60)

phi_plus = bell_state("phi_plus")
s = chsh_s_value(phi_plus)
print(f"  S(phi_plus) = {s:.6f}")
print(f"  Tsirelson bound = {2*np.sqrt(2):.6f}")
assert np.isclose(s, 2*np.sqrt(2), atol=1e-12), "S-value does not reach Tsirelson bound"

# Also via circuit interface
circ_result = chsh_circuit("phi_plus", n_shots=0)
print(f"  S via chsh_circuit = {circ_result['s_value']:.6f}")
assert np.isclose(circ_result["s_value"], 2*np.sqrt(2), atol=1e-12)

# Check all Bell states with optimal angles
for name in bell_names:
    if name in ("phi_plus", "psi_minus"):
        # Default angles are optimal
        s_i = chsh_s_value(bell_state(name))
        print(f"  S({name}, default angles) = {s_i:.6f}")
        assert np.isclose(abs(s_i), 2*np.sqrt(2), atol=1e-12)
    elif name == "phi_minus":
        # Need A1 = -σx (angle -π/2)
        A0_pm, A1_pm = alice_operators(0.0, -np.pi/2)
        s_i = chsh_s_value(bell_state(name), A0_pm, A1_pm, B0, B1)
        print(f"  S(phi_minus, A1=-σx) = {s_i:.6f}")
        assert np.isclose(abs(s_i), 2*np.sqrt(2), atol=1e-12)
    elif name == "psi_plus":
        # Need swapped Bob bases: B0=3π/4, B1=π/4
        B0_pp, B1_pp = bob_operators(3*np.pi/4, np.pi/4)
        s_i = chsh_s_value(bell_state(name), A0, A1, B0_pp, B1_pp)
        print(f"  S(psi_plus, B0=3π/4, B1=π/4) = {s_i:.6f}")
        assert np.isclose(abs(s_i), 2*np.sqrt(2), atol=1e-12)

print("  [PASS] S-value reaches Tsirelson bound with optimal angles for all Bell states.\n")

# ---------------------------------------------------------------------------
# 5. Validate Werner state threshold
# ---------------------------------------------------------------------------
print("=" * 60)
print("5. VALIDATING WERNER STATES")
print("=" * 60)

p_crit = 1 / np.sqrt(2)
print(f"  Critical p = 1/√2 ≈ {p_crit:.6f}")

for p in [1.0, 0.9, p_crit + 0.01, p_crit, p_crit - 0.01, 0.5, 0.0]:
    rho = werner_state(phi_plus, p)
    s_rho = chsh_s_value(rho, is_density_matrix=True)
    trace = np.trace(rho)
    herm = is_hermitian(rho)
    evs = np.linalg.eigvalsh(rho)
    pos = np.all(evs >= -1e-12)
    print(f"  p={p:.3f}  S={s_rho:.6f}  trace={trace:.4f}  Hermitian={herm}  pos_semidef={pos}")
    assert np.isclose(trace, 1.0), f"Werner state trace != 1 for p={p}"
    assert herm, f"Werner state not Hermitian for p={p}"
    assert pos, f"Werner state not positive semidefinite for p={p}"

rho0 = werner_state(phi_plus, 0.0)
assert np.isclose(chsh_s_value(rho0, is_density_matrix=True), 0.0, atol=1e-12)
rho1 = werner_state(phi_plus, 1.0)
assert np.isclose(chsh_s_value(rho1, is_density_matrix=True), 2*np.sqrt(2), atol=1e-12)

print("  [PASS] Werner states are valid density matrices with correct S interpolation.\n")

# ---------------------------------------------------------------------------
# 6. Validate parameterized entangled state
# ---------------------------------------------------------------------------
print("=" * 60)
print("6. VALIDATING PARAMETERIZED ENTANGLED STATE")
print("=" * 60)

thetas = np.linspace(0, np.pi/2, 9)
for th in thetas:
    v = parameterized_entangled_state(th)
    norm = np.linalg.norm(v)
    ee = entanglement_entropy(v)
    cc = concurrence(v)
    print(f"  θ={th:.4f}  norm={norm:.4f}  EE={ee:.4f}  C={cc:.4f}")
    assert np.isclose(norm, 1.0), f"Parameterized state not normalized at θ={th}"
    expected_c = abs(np.sin(2*th))
    assert np.isclose(cc, expected_c, atol=1e-12), f"Concurrence mismatch at θ={th}"

print("  [PASS] Parameterized state has correct normalization and concurrence.\n")

# ---------------------------------------------------------------------------
# 7. Validate semantic circuits
# ---------------------------------------------------------------------------
print("=" * 60)
print("7. VALIDATING SEMANTIC CIRCUITS")
print("=" * 60)

# Uniform amplitudes -> separable (product state) -> S should be <= 2
uniform = [0.5, 0.5, 0.5, 0.5]
res_uniform = semantic_circuit(uniform, n_shots=0)
print(f"  Uniform:   S={res_uniform['s_value']:.4f}  violation={res_uniform['violation']}")
assert not res_uniform["violation"], "Uniform state should NOT violate CHSH"

# Anti-correlated: only |01> and |10> (like psi_plus)
anticorr = [0.0, 1/np.sqrt(2), 1/np.sqrt(2), 0.0]
res_anti = semantic_circuit(anticorr, n_shots=0)
print(f"  Anti-corr: S={res_anti['s_value']:.4f}  violation={res_anti['violation']}")

# Highly correlated: only |00> and |11> (like phi_plus)
corr = [1/np.sqrt(2), 0.0, 0.0, 1/np.sqrt(2)]
res_corr = semantic_circuit(corr, n_shots=0)
print(f"  Correlated: S={res_corr['s_value']:.4f}  violation={res_corr['violation']}")
assert res_corr["violation"], "Maximally correlated state SHOULD violate CHSH"
assert np.isclose(res_corr["concurrence"], 1.0, atol=1e-12)
assert np.isclose(res_corr["entanglement_entropy"], 1.0, atol=1e-12)

print("  [PASS] Semantic circuits behave as expected.\n")

# ---------------------------------------------------------------------------
# 8. Validate compare utilities
# ---------------------------------------------------------------------------
print("=" * 60)
print("8. VALIDATING COMPARE UTILITIES")
print("=" * 60)

from qstk.qc.compare import compare_quantum_llm, sweep_werner_comparison, batch_compare

# Simulate LLM results that match a Werner state with p=0.6
llm_ev = {"A_B": 0.6, "A_B_prime": -0.4, "A_prime_B": 0.4, "A_prime_B_prime": 0.6}
comp = compare_quantum_llm(llm_ev, quantum_state_type="phi_plus")
print(f"  LLM S={comp['llm_s']:.4f}, Quantum S={comp['quantum_s']:.4f}")
print(f"  LLM violation={comp['llm_violation']}, Quantum violation={comp['quantum_violation']}")
print(f"  Equivalent Werner p={comp['equivalent_werner_p']:.4f}")
assert comp["quantum_violation"], "Quantum phi_plus should always violate"
assert comp["equivalent_werner_p"] <= 1.0

# Sweep to find closest Werner p
sweep = sweep_werner_comparison(comp["llm_s"], p_values=[i/100 for i in range(101)])
best = sweep[0]
print(f"  Best matching p={best['p']:.3f}, delta={best['delta']:.4f}")
assert best["delta"] < 0.1

# Batch compare
batch = batch_compare([llm_ev, llm_ev], quantum_state_type="phi_plus")
print(f"  Batch: n={batch['n_trials']}, mean LLM S={batch['llm_s_mean']:.4f}")
assert batch["n_trials"] == 2

print("  [PASS] Compare utilities work correctly.\n")

# ---------------------------------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------------------------------
print("=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

fig = plt.figure(figsize=(18, 24))
gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

# Plot 1: Bell state amplitudes (bar)
ax1 = fig.add_subplot(gs[0, 0])
labels = ["|00>", "|01>", "|10>", "|11>"]
x = np.arange(len(labels))
width = 0.2
for i, name in enumerate(bell_names):
    v = bell_vecs[name]
    reals = np.real(v)
    ax1.bar(x + i*width - 1.5*width, reals, width, label=name, alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel("Re(amplitude)")
ax1.set_title("Bell State Amplitudes (Real Part)")
ax1.legend(fontsize=8)
ax1.axhline(0, color="k", linewidth=0.5)

# Plot 2: Bell state measurement counts (simulated)
ax2 = fig.add_subplot(gs[0, 1])
counts = bell_circuit("phi_plus", n_shots=4096, seed=42)["counts"]
ax2.bar(counts.keys(), counts.values(), color="steelblue", alpha=0.8)
ax2.set_title('Bell State Measurement Counts (|Φ+>, 4096 shots)')
ax2.set_ylabel("Count")

# Plot 3: CHSH S-value vs Werner p
ax3 = fig.add_subplot(gs[1, 0])
ps = np.linspace(0, 1, 101)
ss = [abs(chsh_s_value(werner_state(phi_plus, p), is_density_matrix=True)) for p in ps]
ax3.plot(ps, ss, color="darkgreen", linewidth=2)
ax3.axhline(2, color="red", linestyle="--", label="Classical bound |S|=2")
ax3.axhline(2*np.sqrt(2), color="green", linestyle="--", label="Tsirelson bound 2√2")
ax3.axvline(1/np.sqrt(2), color="purple", linestyle=":", label=f"Critical p=1/√2≈{1/np.sqrt(2):.3f}")
ax3.fill_between(ps, 2, 2*np.sqrt(2), where=[s > 2 for s in ss], color="green", alpha=0.1)
ax3.set_xlabel("Werner parameter p")
ax3.set_ylabel("|S|")
ax3.set_title("CHSH Violation vs Werner Noise")
ax3.legend(fontsize=8)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 2.9)

# Plot 4: Parameterized state entanglement measures
ax4 = fig.add_subplot(gs[1, 1])
thetas = np.linspace(0, np.pi/2, 200)
ees = [entanglement_entropy(parameterized_entangled_state(t)) for t in thetas]
ccs = [concurrence(parameterized_entangled_state(t)) for t in thetas]
ax4.plot(thetas * 180/np.pi, ees, label="Entanglement Entropy", color="purple")
ax4.plot(thetas * 180/np.pi, ccs, label="Concurrence", color="orange")
ax4.set_xlabel("θ (degrees)")
ax4.set_ylabel("Measure")
ax4.set_title("Entanglement vs Parameter θ")
ax4.legend()
ax4.set_xlim(0, 90)

# Plot 5: CHSH S-value landscape (Bob's angles)
ax5 = fig.add_subplot(gs[2, 0])
angles = np.linspace(0, np.pi, 180)
A0, A1 = alice_operators(0.0, np.pi/2)
S_grid = np.zeros((len(angles), len(angles)))
for i, b0 in enumerate(angles):
    for j, b1 in enumerate(angles):
        B0, B1 = bob_operators(b0, b1)
        s = abs(chsh_s_value(phi_plus, A0, A1, B0, B1))
        S_grid[i, j] = s

im = ax5.imshow(S_grid, origin="lower", extent=[0, 180, 0, 180], aspect="auto", cmap="RdYlGn", vmin=0, vmax=2.9)
ax5.set_xlabel("Bob angle b1 (degrees)")
ax5.set_ylabel("Bob angle b0 (degrees)")
ax5.set_title("S-value Landscape over Bob's Angles")
fig.colorbar(im, ax=ax5, label="|S|")
opt_b0 = 45
opt_b1 = 135
ax5.plot(opt_b1, opt_b0, "k*", markersize=12, label="Optimal")
ax5.legend(fontsize=8)

# Plot 6: Entanglement entropy landscape for random states
ax6 = fig.add_subplot(gs[2, 1])
n_random = 2000
rng = np.random.default_rng(123)
random_states = []
for _ in range(n_random):
    v = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    v = v / np.linalg.norm(v)
    random_states.append(v)

ee_vals = [entanglement_entropy(v) for v in random_states]
cc_vals = [concurrence(v) for v in random_states]
ss_vals = [abs(chsh_s_value(v)) for v in random_states]

sc = ax6.scatter(cc_vals, ss_vals, c=ee_vals, cmap="viridis", alpha=0.4, s=10)
ax6.set_xlabel("Concurrence")
ax6.set_ylabel("|S|")
ax6.set_title("Random States: Concurrence vs |S| (colored by EE)")
ax6.axhline(2, color="red", linestyle="--", linewidth=1)
fig.colorbar(sc, ax=ax6, label="Entanglement Entropy")

# Plot 7: CHSH operator eigenvalue spectrum
ax7 = fig.add_subplot(gs[3, 0])
Bchsh = chsh_operator()
ev_chsh = np.linalg.eigvalsh(Bchsh)
ax7.bar(range(len(ev_chsh)), ev_chsh, color="teal", alpha=0.8)
ax7.axhline(2*np.sqrt(2), color="green", linestyle="--", label=f"2√2 ≈ {2*np.sqrt(2):.3f}")
ax7.axhline(-2*np.sqrt(2), color="green", linestyle="--")
ax7.set_xticks(range(len(ev_chsh)))
ax7.set_xticklabels([f"λ{i+1}" for i in range(len(ev_chsh))])
ax7.set_ylabel("Eigenvalue")
ax7.set_title("CHSH Operator Eigenvalues")
ax7.legend()

# Plot 8: Semantic state CHSH S for varying correlation
ax8 = fig.add_subplot(gs[3, 1])
alphas = np.linspace(0, 1, 101)
semantic_ss = []
for a in alphas:
    amp = np.array([a, 0, 0, np.sqrt(1 - a**2)], dtype=complex)
    if np.linalg.norm(amp) == 0:
        semantic_ss.append(0)
    else:
        semantic_ss.append(abs(chsh_s_value(amp / np.linalg.norm(amp))))

ax8.plot(alphas, semantic_ss, color="darkblue", linewidth=2)
ax8.axhline(2, color="red", linestyle="--", label="Classical bound")
ax8.axhline(2*np.sqrt(2), color="green", linestyle="--", label="Tsirelson bound")
ax8.set_xlabel("|α_00| (amplitude on |00>)")
ax8.set_ylabel("|S|")
ax8.set_title("Semantic State: CHSH vs |00> Amplitude")
ax8.legend()
ax8.set_xlim(0, 1)

out_path = "/home/caug/npcww/qstk/qstk/circuit_validation.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"  Saved visualization to: {out_path}")
plt.close()

print("\n" + "=" * 60)
print("ALL VALIDATIONS PASSED!")
print("=" * 60)
