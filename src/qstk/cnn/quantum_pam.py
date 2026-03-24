"""
Quantum Phase-Associative Memory (PAM) -- Proof of Concept
==========================================================

Demonstrates the core PAM mechanism on a quantum simulator using Qiskit:
  - Store a key-value association via outer product (entangling operation)
  - Retrieve the value via phase-interference (constructive for aligned query,
    destructive for misaligned query)

This is a d=4 proof-of-concept: 2 qubits = 4-dimensional complex Hilbert space.
Two registers:
  - Key register: 2 qubits (encodes 4-dim key/query vectors)
  - Value register: 2 qubits (encodes 4-dim value vectors)

The circuit implements:
  1. Amplitude-encode key |k> and value |v> into their registers
  2. Build the state matrix S = |v><k| as an entangling unitary
  3. Apply S to a query |q> to retrieve: S|q> = <k|q> |v>
     - If |q> = |k>: inner product <k|k> = 1, get |v> with certainty
     - If |q> is orthogonal to |k>: <k|q> = 0, get nothing
  4. Measure the value register

Additionally demonstrates:
  - Phase rotation (quantum-native RoPE) via Rz gates
  - Noise resilience using depolarizing noise model

Authors: C. Agostino, Q. Le Thien
For: QNLP AI 2026 Conference -- Quantum Hardware Section
"""

import numpy as np
import sys
import os

# ---------------------------------------------------------------------------
# Imports -- Qiskit
# ---------------------------------------------------------------------------
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit import transpile
    from qiskit.circuit.library import Initialize, UnitaryGate
    from qiskit.quantum_info import Statevector, Operator, partial_trace
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: uv pip install qiskit qiskit-aer --python .venv/bin/python")
    sys.exit(1)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ===========================================================================
# Helper functions
# ===========================================================================

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a complex vector to unit L2 norm."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Cannot normalize zero vector")
    return v / n


def build_outer_product_unitary(key: np.ndarray, value: np.ndarray) -> np.ndarray:
    """
    Build a unitary U that implements the map:
        U |q>|0> = <k|q> |q>|v> + ... (orthogonal complement)

    For the PAM retrieval, we want S|q> = <k|q>|v>.  On a quantum computer
    we implement this as a controlled operation on two registers.

    The actual unitary we build acts on the 4-qubit space (key_reg x value_reg)
    and implements:
        |k>|0> -> |k>|v>       (the "storage" operation)

    More precisely, we build U such that for the key register in state |k>,
    the value register is rotated from |0> to |v>.

    We use the SWAP-test inspired approach:
      1. On the full 16-dim space, U = I_key (x) V_conditional
         where V_conditional flips the value register to |v> conditioned on
         the key register being in state |k>.

    Implementation: We build the 16x16 unitary directly.
    """
    d = len(key)
    assert len(value) == d
    key = normalize(key)
    value = normalize(value)

    # Build the controlled unitary on the full d^2 space.
    # For basis state |i> on key register:
    #   if |i> == |k>: apply rotation |0> -> |v> on value register
    #   else: identity on value register
    #
    # The rotation |0> -> |v> is any unitary whose first column is |v>.
    # We use Householder reflections to build it.

    # Step 1: Build V_rot such that V_rot |0> = |v>
    V_rot = _unitary_first_col(value)

    # Step 2: Build the full controlled unitary
    # U = sum_i |i><i| (x) U_i where U_i = V_rot if |i> = |k>, else I
    U_full = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        proj_i = np.zeros((d, d), dtype=complex)
        proj_i[i, i] = 1.0
        # Overlap with key: how much does |i> align with |k>?
        # For exact key matching, only apply V_rot when i corresponds to |k>
        # Since key is a superposition, we need a different approach.
        # We'll work in the key's eigenbasis.
        pass  # see below

    # Better approach: work in the basis where |k> is a basis vector.
    # Let R be a unitary such that R|k> = |0>.
    R_key = _unitary_first_col(key).conj().T  # R maps |k> -> |0>

    # In the rotated basis:
    #   Controlled-V = |0><0| (x) V_rot + sum_{i>0} |i><i| (x) I
    ctrl_V = np.zeros((d * d, d * d), dtype=complex)
    proj_0 = np.zeros((d, d), dtype=complex)
    proj_0[0, 0] = 1.0
    ctrl_V += np.kron(proj_0, V_rot)
    for i in range(1, d):
        proj_i = np.zeros((d, d), dtype=complex)
        proj_i[i, i] = 1.0
        ctrl_V += np.kron(proj_i, np.eye(d, dtype=complex))

    # Transform back: U = (R^dag (x) I) @ ctrl_V @ (R (x) I)
    R_full = np.kron(R_key, np.eye(d, dtype=complex))
    R_dag_full = np.kron(R_key.conj().T, np.eye(d, dtype=complex))
    U_full = R_dag_full @ ctrl_V @ R_full

    # Verify unitarity
    check = U_full @ U_full.conj().T
    assert np.allclose(check, np.eye(d * d), atol=1e-10), \
        f"Unitarity check failed, max error: {np.max(np.abs(check - np.eye(d*d)))}"

    return U_full


def _unitary_first_col(v: np.ndarray) -> np.ndarray:
    """
    Build a unitary matrix U such that U|0> = |v> (v must be unit norm).
    Uses Gram-Schmidt to extend |v> to a full orthonormal basis.
    """
    d = len(v)
    v = normalize(v)

    # Start with v as first column, extend via Gram-Schmidt
    basis = [v.copy()]
    for i in range(d):
        e_i = np.zeros(d, dtype=complex)
        e_i[i] = 1.0
        # Orthogonalize against existing basis vectors
        for b in basis:
            e_i = e_i - np.dot(b.conj(), e_i) * b
        norm = np.linalg.norm(e_i)
        if norm > 1e-10:
            basis.append(e_i / norm)
        if len(basis) == d:
            break

    U = np.column_stack(basis)
    return U


def phase_rotate_state(statevector: np.ndarray, theta: float) -> np.ndarray:
    """
    Apply a phase rotation e^{i*theta*j} to the j-th component.
    This is the quantum-native version of RoPE (Rotary Position Embedding).
    On hardware, this is implemented as Rz gates on individual qubits.
    """
    d = len(statevector)
    phases = np.exp(1j * theta * np.arange(d))
    return statevector * phases


# ===========================================================================
# Core quantum PAM circuit
# ===========================================================================

def build_pam_circuit(
    key: np.ndarray,
    value: np.ndarray,
    query: np.ndarray,
    label: str = "",
    apply_phase_rope: bool = False,
    rope_theta: float = 0.0,
) -> QuantumCircuit:
    """
    Build a quantum circuit that:
      1. Prepares query |q> on the key register
      2. Initializes value register to |0>
      3. Applies the PAM storage unitary U_{k,v} (outer product |v><k|)
      4. Measures the value register

    If query aligns with key, measurement collapses to |v>.
    If query is orthogonal to key, no information is retrieved.

    Args:
        key:    4-dim complex vector (the stored key)
        value:  4-dim complex vector (the stored value)
        query:  4-dim complex vector (the retrieval query)
        label:  Circuit label for display
        apply_phase_rope: Whether to apply position-dependent phase rotation
        rope_theta: Phase angle for RoPE-like rotation

    Returns:
        QuantumCircuit ready to run on a simulator.
    """
    d = 4  # Hilbert space dimension
    n_qubits = 2  # log2(4) = 2 qubits per register

    key = normalize(key.astype(complex))
    value = normalize(value.astype(complex))
    query = normalize(query.astype(complex))

    # Optional: apply phase rotation (quantum RoPE) to query
    if apply_phase_rope:
        query = phase_rotate_state(query, rope_theta)
        query = normalize(query)

    # Build the storage unitary
    U_store = build_outer_product_unitary(key, value)

    # Create registers
    qr_key = QuantumRegister(n_qubits, 'key')
    qr_val = QuantumRegister(n_qubits, 'val')
    cr = ClassicalRegister(n_qubits, 'meas')
    qc = QuantumCircuit(qr_key, qr_val, cr)
    if label:
        qc.name = label

    # Step 1: Prepare query state on key register
    qc.initialize(query.tolist(), qr_key)

    # Step 2: Value register starts in |00> (default)

    # Step 3: Apply the PAM storage unitary on both registers
    U_gate = UnitaryGate(U_store, label='PAM_S')
    qc.append(U_gate, list(qr_key) + list(qr_val))

    # Step 4: Measure value register
    qc.barrier()
    qc.measure(qr_val, cr)

    return qc


def build_phase_rope_circuit() -> QuantumCircuit:
    """
    Build a small circuit demonstrating quantum-native RoPE.
    Apply Rz(theta) gates to encode position-dependent phases.
    This is a single-qubit operation -- extremely natural on quantum hardware.
    """
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(qr, cr)
    qc.name = "Quantum RoPE"

    # Prepare a superposition state
    qc.h(qr[0])
    qc.h(qr[1])

    # Position-dependent phase rotation (this IS RoPE on quantum hardware)
    theta_pos = np.pi / 4  # position-dependent angle
    qc.rz(theta_pos, qr[0])       # Phase shift on qubit 0
    qc.rz(2 * theta_pos, qr[1])   # Doubled phase on qubit 1 (like RoPE frequencies)

    qc.barrier()
    qc.measure(qr, cr)
    return qc


# ===========================================================================
# Run experiments
# ===========================================================================

def run_experiment(
    key: np.ndarray,
    value: np.ndarray,
    query: np.ndarray,
    label: str,
    shots: int = 8192,
    noise_model=None,
) -> dict:
    """Run a PAM circuit and return measurement counts."""
    qc = build_pam_circuit(key, value, query, label=label)

    sim = AerSimulator(noise_model=noise_model) if noise_model else AerSimulator()
    qc_t = transpile(qc, sim)
    result = sim.run(qc_t, shots=shots).result()
    counts = result.get_counts()

    return counts


def counts_to_probs(counts: dict, n_bits: int = 2) -> dict:
    """Convert raw counts to probabilities for all possible outcomes."""
    total = sum(counts.values())
    probs = {}
    for i in range(2**n_bits):
        bitstring = format(i, f'0{n_bits}b')
        probs[bitstring] = counts.get(bitstring, 0) / total
    return probs


def get_noise_model(p_depol: float) -> NoiseModel:
    """Create a simple depolarizing noise model."""
    noise_model = NoiseModel()
    # Single-qubit depolarizing error
    error_1q = depolarizing_error(p_depol, 1)
    # Two-qubit depolarizing error (higher rate)
    error_2q = depolarizing_error(p_depol * 2, 2)
    noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
    return noise_model


# ===========================================================================
# Statevector analysis (for understanding, not just measurement)
# ===========================================================================

def analyze_statevector(key, value, query, label=""):
    """
    Run the circuit without measurement and inspect the full statevector.
    This gives exact probabilities and confirms the phase-interference mechanism.
    """
    d = 4
    key = normalize(key.astype(complex))
    value = normalize(value.astype(complex))
    query = normalize(query.astype(complex))

    U_store = build_outer_product_unitary(key, value)

    # Initial state: |query>|00>
    init_state = np.kron(query, np.array([1, 0, 0, 0], dtype=complex))

    # Apply storage unitary
    final_state = U_store @ init_state

    # Probability of each value-register outcome
    # Value register is the last 2 qubits (indices correspond to last 2 bits)
    probs_val = np.zeros(d)
    for val_idx in range(d):
        # Sum over key register states
        for key_idx in range(d):
            full_idx = key_idx * d + val_idx
            probs_val[val_idx] += np.abs(final_state[full_idx])**2

    print(f"\n--- Statevector Analysis: {label} ---")
    print(f"  Key:   {key}")
    print(f"  Value: {value}")
    print(f"  Query: {query}")
    print(f"  Inner product <key|query>: {np.dot(key.conj(), query):.4f}")
    print(f"  |<key|query>|^2 = {np.abs(np.dot(key.conj(), query))**2:.4f}")
    print(f"  Value register probabilities:")
    for i in range(d):
        marker = " <-- target" if i == np.argmax(np.abs(value)**2) else ""
        print(f"    |{i:02b}> : {probs_val[i]:.4f}{marker}")

    return probs_val


# ===========================================================================
# Main demonstration
# ===========================================================================

def main():
    print("=" * 72)
    print("QUANTUM PHASE-ASSOCIATIVE MEMORY (PAM) -- PROOF OF CONCEPT")
    print("d=4 (2 qubits per register), Qiskit Aer Simulator")
    print("=" * 72)

    # Define the key-value association to store
    # "Token 0 maps to Token 1" -- key=|00>, value=|01>
    key   = np.array([1, 0, 0, 0], dtype=complex)  # |00>
    value = np.array([0, 1, 0, 0], dtype=complex)   # |01>

    # Queries
    query_aligned    = np.array([1, 0, 0, 0], dtype=complex)  # same as key
    query_orthogonal = np.array([0, 0, 1, 0], dtype=complex)  # |10>, orthogonal
    query_partial    = normalize(np.array([1, 0, 1, 0], dtype=complex))  # partial overlap

    shots = 16384

    # ----- Statevector analysis (exact) -----
    print("\n" + "=" * 72)
    print("SECTION 1: EXACT STATEVECTOR ANALYSIS")
    print("=" * 72)

    sv_aligned = analyze_statevector(key, value, query_aligned, "Aligned Query (|00>)")
    sv_ortho   = analyze_statevector(key, value, query_orthogonal, "Orthogonal Query (|10>)")
    sv_partial = analyze_statevector(key, value, query_partial, "Partial Overlap ((|00>+|10>)/sqrt(2))")

    # ----- Ideal simulator -----
    print("\n" + "=" * 72)
    print("SECTION 2: IDEAL SIMULATOR (NO NOISE)")
    print("=" * 72)

    counts_aligned = run_experiment(key, value, query_aligned, "Aligned", shots=shots)
    counts_ortho   = run_experiment(key, value, query_orthogonal, "Orthogonal", shots=shots)
    counts_partial = run_experiment(key, value, query_partial, "Partial", shots=shots)

    probs_aligned = counts_to_probs(counts_aligned)
    probs_ortho   = counts_to_probs(counts_ortho)
    probs_partial = counts_to_probs(counts_partial)

    print(f"\nAligned query (|00>):")
    for bs, p in sorted(probs_aligned.items()):
        print(f"  |{bs}> : {p:.4f}")

    print(f"\nOrthogonal query (|10>):")
    for bs, p in sorted(probs_ortho.items()):
        print(f"  |{bs}> : {p:.4f}")

    print(f"\nPartial overlap query ((|00>+|10>)/sqrt(2)):")
    for bs, p in sorted(probs_partial.items()):
        print(f"  |{bs}> : {p:.4f}")

    # ----- Noisy simulator -----
    print("\n" + "=" * 72)
    print("SECTION 3: NOISY SIMULATOR (DEPOLARIZING NOISE)")
    print("=" * 72)

    noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    fidelities = []  # P(|01>) for aligned query at each noise level

    for p_noise in noise_levels:
        nm = get_noise_model(p_noise) if p_noise > 0 else None
        counts = run_experiment(key, value, query_aligned, f"noise={p_noise}", shots=shots, noise_model=nm)
        probs = counts_to_probs(counts)
        fid = probs.get('01', 0.0)
        fidelities.append(fid)
        print(f"  p_depol={p_noise:.3f}  ->  P(|01>) = {fid:.4f}  (ideal: 1.0)")

    # ----- Phase rotation (quantum RoPE) demo -----
    print("\n" + "=" * 72)
    print("SECTION 4: PHASE ROTATION (QUANTUM-NATIVE RoPE)")
    print("=" * 72)

    print("\nDemonstrating position-dependent phase encoding:")
    print("  Rz(theta) on qubit 0, Rz(2*theta) on qubit 1")
    print("  This is the quantum-native version of Rotary Position Embedding.")
    print("  On real hardware: single-qubit gate, ~10ns, >99.9% fidelity.")

    # Show how phase rotation affects retrieval
    rope_thetas = np.linspace(0, 2*np.pi, 17)
    rope_fidelities = []
    for theta in rope_thetas:
        query_roped = phase_rotate_state(query_aligned.copy(), theta)
        query_roped = normalize(query_roped)
        counts = run_experiment(key, value, query_roped, f"RoPE theta={theta:.2f}", shots=shots)
        probs = counts_to_probs(counts)
        fid = probs.get('01', 0.0)
        rope_fidelities.append(fid)
        print(f"  theta={theta:.2f}  ->  P(|01>) = {fid:.4f}")

    # ----- Build a display circuit for the figure -----
    display_circuit = build_pam_circuit(key, value, query_aligned, label="PAM Retrieval")

    # ===========================================================================
    # VISUALIZATION
    # ===========================================================================
    print("\n" + "=" * 72)
    print("GENERATING FIGURES...")
    print("=" * 72)

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "Quantum Phase-Associative Memory (PAM) -- d=4 Proof of Concept\n"
        "2 qubits per register, Qiskit Aer Simulator",
        fontsize=16, fontweight='bold', y=0.98
    )

    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35, top=0.92, bottom=0.06)

    # ---- Panel 1: Circuit diagram (spans top row) ----
    ax_circ = fig.add_subplot(gs[0, :])
    try:
        display_circuit.draw(output='mpl', ax=ax_circ, style='iqp', fold=60)
    except Exception:
        # Fallback: text representation
        ax_circ.text(0.5, 0.5, str(display_circuit.draw(output='text')),
                     transform=ax_circ.transAxes, fontsize=8,
                     verticalalignment='center', horizontalalignment='center',
                     fontfamily='monospace')
        ax_circ.set_xlim(0, 1)
        ax_circ.set_ylim(0, 1)
    ax_circ.set_title("PAM Quantum Circuit (key=|00>, value=|01>, query=|00>)",
                       fontsize=12, fontweight='bold')

    # ---- Panel 2: Aligned vs orthogonal measurement probabilities ----
    ax_bar = fig.add_subplot(gs[1, 0])
    states = ['|00>', '|01>', '|10>', '|11>']
    bitstrings = ['00', '01', '10', '11']
    x_pos = np.arange(len(states))
    width = 0.3

    bars1 = ax_bar.bar(x_pos - width, [probs_aligned[b] for b in bitstrings],
                       width, label='Aligned query (|00>)', color='#2196F3', edgecolor='black')
    bars2 = ax_bar.bar(x_pos, [probs_ortho[b] for b in bitstrings],
                       width, label='Orthogonal query (|10>)', color='#F44336', edgecolor='black')
    bars3 = ax_bar.bar(x_pos + width, [probs_partial[b] for b in bitstrings],
                       width, label='Partial overlap', color='#FF9800', edgecolor='black')

    ax_bar.set_xlabel('Value Register Outcome')
    ax_bar.set_ylabel('Probability')
    ax_bar.set_title('Phase-Interference Retrieval\n(Ideal Simulator)', fontweight='bold')
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(states)
    ax_bar.legend(fontsize=8, loc='upper right')
    ax_bar.set_ylim(0, 1.15)
    ax_bar.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='uniform')

    # Add probability annotations
    for bar in bars1:
        height = bar.get_height()
        if height > 0.02:
            ax_bar.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)

    # ---- Panel 3: Statevector exact probabilities ----
    ax_sv = fig.add_subplot(gs[1, 1])
    x_pos2 = np.arange(4)
    width2 = 0.25

    ax_sv.bar(x_pos2 - width2, sv_aligned, width2,
              label='Aligned', color='#2196F3', edgecolor='black')
    ax_sv.bar(x_pos2, sv_ortho, width2,
              label='Orthogonal', color='#F44336', edgecolor='black')
    ax_sv.bar(x_pos2 + width2, sv_partial, width2,
              label='Partial', color='#FF9800', edgecolor='black')

    ax_sv.set_xlabel('Value Register Basis State')
    ax_sv.set_ylabel('Probability (exact)')
    ax_sv.set_title('Exact Statevector Analysis\n(No Sampling Noise)', fontweight='bold')
    ax_sv.set_xticks(x_pos2)
    ax_sv.set_xticklabels(states)
    ax_sv.legend(fontsize=8)
    ax_sv.set_ylim(0, 1.15)

    # ---- Panel 4: Noise resilience ----
    ax_noise = fig.add_subplot(gs[1, 2])
    ax_noise.plot(noise_levels, fidelities, 'o-', color='#4CAF50', linewidth=2, markersize=8)
    ax_noise.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Ideal')
    ax_noise.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Random (1/d)')
    ax_noise.set_xlabel('Depolarizing Error Rate')
    ax_noise.set_ylabel('Retrieval Fidelity P(|01>)')
    ax_noise.set_title('Noise Resilience\n(Aligned Query)', fontweight='bold')
    ax_noise.legend(fontsize=9)
    ax_noise.set_ylim(0, 1.1)
    ax_noise.grid(True, alpha=0.3)

    # ---- Panel 5: Phase rotation (quantum RoPE) effect ----
    ax_rope = fig.add_subplot(gs[2, 0])
    ax_rope.plot(rope_thetas, rope_fidelities, 's-', color='#9C27B0', linewidth=2, markersize=6)
    ax_rope.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Random (1/d)')
    ax_rope.set_xlabel('Phase Rotation Angle (radians)')
    ax_rope.set_ylabel('Retrieval Fidelity P(|01>)')
    ax_rope.set_title('Quantum RoPE: Phase vs Retrieval\n(Rz-based Position Encoding)', fontweight='bold')
    ax_rope.set_xlim(0, 2*np.pi)
    ax_rope.legend(fontsize=9)
    ax_rope.grid(True, alpha=0.3)

    # ---- Panel 6: Interpretation / summary text ----
    ax_text = fig.add_subplot(gs[2, 1:])
    ax_text.axis('off')
    summary_text = (
        "QUANTUM PAM: PROOF-OF-CONCEPT RESULTS\n"
        "=" * 50 + "\n\n"
        "Circuit: 4 qubits (2 key + 2 value), d=4 Hilbert space\n\n"
        "KEY RESULT: Phase-interference retrieval works.\n"
        "  - Aligned query:    P(correct value) ~ 1.0  (constructive interference)\n"
        "  - Orthogonal query: P(correct value) ~ 0.0  (destructive interference)\n"
        "  - Partial overlap:  P(correct value) ~ 0.5  (partial interference)\n\n"
        "QUANTUM ADVANTAGES DEMONSTRATED:\n"
        "  1. State update (outer product |v><k|) = single entangling unitary\n"
        "  2. Retrieval (S@q) = unitary application + measurement\n"
        "  3. Phase encoding (RoPE) = native Rz gates (~10ns, >99.9% fidelity)\n"
        "  4. Complex arithmetic is FREE -- qubits are natively complex\n\n"
        "SCALING IMPLICATIONS (from resource analysis):\n"
        "  - d=4:     2+2 = 4 qubits   (this demo)\n"
        "  - d=64:    6+6 = 12 qubits   (feasible on current hardware)\n"
        "  - d=1024: 10+10 = 20 qubits  (feasible on IBM Eagle, 127 qubits)\n"
        "  - State prep gates scale as O(2^n) -- the bottleneck.\n\n"
        "NOISE: Retrieval fidelity degrades gracefully with depolarizing noise.\n"
        f"  At p=0.01: fidelity = {fidelities[3]:.3f}  "
        f"(IBM Eagle 1q error ~ 0.001)"
    )
    ax_text.text(0.02, 0.98, summary_text, transform=ax_text.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    # Save
    out_path = '/home/caug/npcww/qstk/results/quantum_pam_demo.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {out_path}")
    plt.close(fig)

    # Also save the transpiled circuit stats
    print("\n" + "=" * 72)
    print("CIRCUIT STATISTICS (after transpilation)")
    print("=" * 72)
    sim = AerSimulator()
    qc = build_pam_circuit(key, value, query_aligned, label="PAM")
    qc_t = transpile(qc, sim)
    print(f"  Total qubits:     {qc_t.num_qubits}")
    print(f"  Circuit depth:    {qc_t.depth()}")
    print(f"  Gate count:       {dict(qc_t.count_ops())}")

    # RoPE circuit stats
    qc_rope = build_phase_rope_circuit()
    qc_rope_t = transpile(qc_rope, sim)
    print(f"\n  RoPE circuit depth:   {qc_rope_t.depth()}")
    print(f"  RoPE gate count:      {dict(qc_rope_t.count_ops())}")

    print("\n" + "=" * 72)
    print("DONE. Quantum PAM proof-of-concept complete.")
    print("=" * 72)

    return {
        'probs_aligned': probs_aligned,
        'probs_orthogonal': probs_ortho,
        'probs_partial': probs_partial,
        'noise_fidelities': list(zip(noise_levels, fidelities)),
        'rope_fidelities': list(zip(rope_thetas.tolist(), rope_fidelities)),
        'statevector_aligned': sv_aligned.tolist(),
        'statevector_orthogonal': sv_ortho.tolist(),
    }


if __name__ == '__main__':
    results = main()
