"""
Quantum PAM on Simulated IBM Hardware (Brisbane-class noise)
============================================================

Since no IBM Quantum API token is configured on this machine, we simulate
what real hardware execution would look like using two complementary approaches:

  1. GenericBackendV2 -- Qiskit's built-in fake backend with noise parameters
     sampled from historical IBM hardware distributions (T1, T2, gate errors,
     readout errors, coupling map constraints).

  2. Custom Brisbane-class noise model -- hand-calibrated from published IBM
     Brisbane specifications: T1~200us, T2~150us, 1q error ~3e-4, 2q error ~8e-3,
     readout error ~1.2%, heavy-hex coupling map.

Both approaches transpile the circuit to the backend's native gate set (ECR/CX,
SX, RZ, X) and coupling map, then simulate with realistic noise.

The script compares:
  - Ideal simulator (no noise, no hardware constraints)
  - GenericBackendV2 with noise (IBM-calibrated defaults)
  - Custom Brisbane noise model (thermal relaxation + readout errors)

QUBIT ORDERING FIX
===================
The PAM unitary is built in math convention where the state index is
key_idx * d + val_idx (key = MSBs, val = LSBs). Qiskit's UnitaryGate
treats qubit 0 in the qubit list as the LSB of the matrix index. Therefore,
to match the math convention, we append the unitary with qubit order
[val_0, val_1, key_0, key_1] rather than [key_0, key_1, val_0, val_1].
This ensures that the value register occupies the LSBs of the unitary index,
matching the kron(key, val) structure.

HOW TO RUN ON REAL IBM HARDWARE
===============================
If you have an IBM Quantum account:

  1. pip install qiskit-ibm-runtime
  2. from qiskit_ibm_runtime import QiskitRuntimeService
     QiskitRuntimeService.save_account(
         channel="ibm_quantum",
         token="YOUR_IBM_QUANTUM_TOKEN",
         set_as_default=True,
     )
  3. service = QiskitRuntimeService()
     backend = service.least_busy(min_num_qubits=4, operational=True)
  4. from qiskit_ibm_runtime import SamplerV2
     sampler = SamplerV2(mode=backend)
     job = sampler.run([transpiled_circuit], shots=8192)
     result = job.result()

  Free tier backends (as of 2026):
    - ibm_brisbane (127 qubits, Eagle r3, ECR basis)
    - ibm_kyoto (127 qubits, Eagle r3)
    - ibm_osaka (127 qubits, Eagle r3)

Authors: C. Agostino, Q. Le Thien
For: QNLP AI 2026 Conference -- Quantum Hardware Section
"""

import numpy as np
import sys
import os
import time
import warnings

# Suppress Qiskit noise model composition warnings
warnings.filterwarnings('ignore', message='.*all-qubit error already exists.*')

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit import transpile
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        thermal_relaxation_error,
        ReadoutError,
    )
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: uv pip install qiskit qiskit-aer")
    sys.exit(1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import helpers from the existing module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_pam import (
    normalize,
    build_outer_product_unitary,
    _unitary_first_col,
    phase_rotate_state,
)


# ===========================================================================
# Corrected PAM circuit builder
# ===========================================================================

def build_pam_circuit(
    key: np.ndarray,
    value: np.ndarray,
    query: np.ndarray,
    label: str = "",
) -> QuantumCircuit:
    """
    Build a PAM circuit with correct Qiskit qubit ordering.

    The outer-product unitary U uses kron(key_space, val_space) internally,
    so the state index is key_idx * d + val_idx. Qiskit's UnitaryGate maps
    the FIRST qubit in the list to the LSB of the matrix index. Therefore
    we must pass val qubits first, then key qubits, so that val occupies
    the low-order bits.

    Args:
        key:   4-dim complex vector (stored key)
        value: 4-dim complex vector (stored value)
        query: 4-dim complex vector (retrieval query)
        label: Circuit name

    Returns:
        QuantumCircuit with measurement on the value register.
    """
    d = 4
    n_qubits = 2  # per register

    key = normalize(key.astype(complex))
    value = normalize(value.astype(complex))
    query = normalize(query.astype(complex))

    U_store = build_outer_product_unitary(key, value)

    qr_key = QuantumRegister(n_qubits, 'key')
    qr_val = QuantumRegister(n_qubits, 'val')
    cr = ClassicalRegister(n_qubits, 'meas')
    qc = QuantumCircuit(qr_key, qr_val, cr)
    if label:
        qc.name = label

    # Prepare query on key register
    qc.initialize(query.tolist(), qr_key)

    # Apply PAM storage unitary with correct qubit ordering:
    # val qubits first (LSBs), then key qubits (MSBs)
    U_gate = UnitaryGate(U_store, label='PAM_S')
    qc.append(U_gate, list(qr_val) + list(qr_key))

    # Measure value register
    qc.barrier()
    qc.measure(qr_val, cr)

    return qc


def counts_to_probs(counts: dict, n_bits: int = 2) -> dict:
    """Convert raw counts to probabilities for all possible outcomes."""
    total = sum(counts.values())
    probs = {}
    for i in range(2**n_bits):
        bitstring = format(i, f'0{n_bits}b')
        probs[bitstring] = counts.get(bitstring, 0) / total
    return probs


def analyze_statevector(key, value, query, label=""):
    """
    Exact statevector analysis (math convention, no circuit).
    Returns probabilities for each value-register basis state.
    """
    d = 4
    key = normalize(key.astype(complex))
    value = normalize(value.astype(complex))
    query = normalize(query.astype(complex))

    U = build_outer_product_unitary(key, value)
    init = np.kron(query, np.array([1, 0, 0, 0], dtype=complex))
    final = U @ init

    probs_val = np.zeros(d)
    for val_idx in range(d):
        for key_idx in range(d):
            full_idx = key_idx * d + val_idx
            probs_val[val_idx] += np.abs(final[full_idx])**2

    overlap = np.dot(key.conj(), query)
    print(f"\n--- Statevector: {label} ---")
    print(f"  <key|query> = {overlap:.4f},  |<k|q>|^2 = {np.abs(overlap)**2:.4f}")
    for i in range(d):
        marker = " <-- target" if i == np.argmax(np.abs(value)**2) else ""
        print(f"    |{i:02b}> : {probs_val[i]:.4f}{marker}")

    return probs_val


# ===========================================================================
# Brisbane-class hardware parameters (from IBM published specs, 2024-2025)
# ===========================================================================

BRISBANE_PARAMS = {
    "name": "ibm_brisbane (simulated)",
    "num_qubits": 127,
    "processor": "Eagle r3",
    "T1_us": 200.0,
    "T2_us": 150.0,
    "1q_error": 2.8e-4,
    "2q_error": 7.6e-3,
    "1q_duration_ns": 60,
    "2q_duration_ns": 660,
    "readout_error": 0.012,
    "readout_duration_ns": 1120,
    "basis_gates": ["ecr", "id", "rz", "sx", "x"],
}


def build_brisbane_coupling_map_subset(n_qubits: int = 8) -> CouplingMap:
    """
    Build a subset of IBM Brisbane's heavy-hex coupling map.

    Heavy-hex topology has degree-2 and degree-3 nodes. This mimics the key
    constraint: not all qubits are connected, so the transpiler must insert
    SWAP gates to route 2-qubit operations between non-adjacent qubits.
    """
    if n_qubits == 4:
        edges = [[0, 1], [1, 2], [2, 3]]
    elif n_qubits == 8:
        # Ladder graph (approximates a heavy-hex fragment)
        edges = [
            [0, 1], [2, 3], [4, 5], [6, 7],  # rungs
            [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7],  # rails
        ]
    else:
        edges = [[i, i+1] for i in range(n_qubits - 1)]
    bidirectional = edges + [[b, a] for a, b in edges]
    return CouplingMap(bidirectional)


def build_brisbane_noise_model(
    n_qubits: int = 8,
    params: dict = None,
) -> NoiseModel:
    """
    Build a physically realistic noise model approximating IBM Brisbane.

    The noise has three components, each modeling different physics:

    1. THERMAL RELAXATION (T1/T2): During every gate, qubits undergo
       spontaneous decay. T1 = energy relaxation (|1> -> |0>, amplitude
       damping). T2 = phase coherence loss (dephasing). In superconducting
       transmons, T1 ~ 200us and T2 ~ 150us for Brisbane-class devices.
       The error per gate depends on gate_duration / T1.

    2. READOUT ERRORS: Asymmetric confusion matrix. Measuring |1> as |0>
       (p_meas_1_given_0) differs from measuring |0> as |1> (p_meas_0_given_1).
       Brisbane readout error ~ 1.2%.

    3. DEPOLARIZING (residual): Captures coherent errors, cross-talk, and
       other imperfections not modeled by thermal relaxation. Added on top
       of thermal relaxation for 2-qubit gates.
    """
    if params is None:
        params = BRISBANE_PARAMS

    noise_model = NoiseModel()

    T1 = params["T1_us"] * 1e3    # ns
    T2 = params["T2_us"] * 1e3    # ns
    t_1q = params["1q_duration_ns"]
    t_2q = params["2q_duration_ns"]

    # Single-qubit gate errors
    error_1q = thermal_relaxation_error(T1, T2, t_1q)
    noise_model.add_all_qubit_quantum_error(error_1q, ['sx', 'x', 'id'])

    # Two-qubit gate errors (each qubit relaxes independently)
    error_2q_a = thermal_relaxation_error(T1, T2, t_2q)
    error_2q_b = thermal_relaxation_error(T1, T2, t_2q)
    error_2q = error_2q_a.expand(error_2q_b)
    noise_model.add_all_qubit_quantum_error(error_2q, ['ecr', 'cx'])

    # Readout confusion matrix (asymmetric)
    p_err = params["readout_error"]
    readout_err = ReadoutError([
        [1 - p_err * 0.4, p_err * 0.4],
        [p_err * 1.6, 1 - p_err * 1.6],
    ])
    for qubit in range(n_qubits):
        noise_model.add_readout_error(readout_err, [qubit])

    # Extra depolarizing on 2-qubit gates (cross-talk, coherent errors)
    extra_2q = depolarizing_error(params["2q_error"] * 0.3, 2)
    noise_model.add_all_qubit_quantum_error(extra_2q, ['ecr', 'cx'])

    return noise_model


# ===========================================================================
# Backend runners
# ===========================================================================

def run_on_ideal(qc: QuantumCircuit, shots: int = 16384) -> dict:
    """Run on ideal (noiseless) Aer simulator."""
    sim = AerSimulator()
    qc_t = transpile(qc, sim)
    result = sim.run(qc_t, shots=shots).result()
    return result.get_counts()


def run_on_generic_backend(
    qc: QuantumCircuit,
    shots: int = 16384,
    seed: int = 42,
) -> tuple:
    """
    Run on GenericBackendV2 with IBM-calibrated noise defaults.
    Returns (counts, transpiled_circuit, backend).
    """
    coupling = build_brisbane_coupling_map_subset(8)
    backend = GenericBackendV2(
        num_qubits=8,
        basis_gates=["ecr", "id", "rz", "sx", "x"],
        coupling_map=coupling,
        seed=seed,
        noise_info=True,
    )
    qc_t = transpile(qc, backend, optimization_level=2, seed_transpiler=seed)
    job = backend.run(qc_t, shots=shots)
    counts = job.result().get_counts()
    return counts, qc_t, backend


def run_on_brisbane_noise(
    qc: QuantumCircuit,
    shots: int = 16384,
    seed: int = 42,
) -> tuple:
    """
    Run on AerSimulator with Brisbane-class thermal relaxation noise.
    Returns (counts, transpiled_circuit, noise_model).
    """
    n_qubits = 8
    coupling = build_brisbane_coupling_map_subset(n_qubits)
    noise_model = build_brisbane_noise_model(n_qubits)

    sim = AerSimulator(
        noise_model=noise_model,
        coupling_map=coupling,
        basis_gates=BRISBANE_PARAMS["basis_gates"] + ["reset", "delay", "measure"],
    )
    qc_t = transpile(
        qc, sim,
        basis_gates=BRISBANE_PARAMS["basis_gates"],
        coupling_map=coupling,
        optimization_level=2,
        seed_transpiler=seed,
    )
    result = sim.run(qc_t, shots=shots, seed_simulator=seed).result()
    counts = result.get_counts()
    return counts, qc_t, noise_model


def circuit_stats(qc: QuantumCircuit) -> dict:
    """Extract circuit statistics after transpilation."""
    ops = dict(qc.count_ops())
    n_cx = ops.get('cx', 0) + ops.get('ecr', 0)
    n_1q = sum(v for k, v in ops.items() if k in ['sx', 'x', 'rz', 'id'])
    return {
        'depth': qc.depth(),
        'num_qubits': qc.num_qubits,
        'gate_count': ops,
        'n_cx': n_cx,
        'n_1q': n_1q,
        'total_gates': sum(v for k, v in ops.items()
                          if k not in ['measure', 'barrier', 'reset', 'delay']),
    }


def estimate_fidelity(stats: dict, params: dict = None) -> float:
    """
    Pessimistic fidelity estimate: F ~ prod(1 - e_i)^n_i.
    Assumes independent errors, no error mitigation.
    Real hardware with error mitigation would outperform this estimate.
    """
    if params is None:
        params = BRISBANE_PARAMS
    n_1q = stats['n_1q']
    n_cx = stats['n_cx']
    n_meas = stats['gate_count'].get('measure', 0)
    f = (1 - params['1q_error'])**n_1q
    f *= (1 - params['2q_error'])**n_cx
    f *= (1 - params['readout_error'])**n_meas
    return f


# ===========================================================================
# Main experiment
# ===========================================================================

def main():
    print("=" * 78)
    print("QUANTUM PAM -- SIMULATED HARDWARE EXECUTION")
    print("Brisbane-class IBM Eagle r3 noise (127-qubit processor)")
    print("=" * 78)

    # ---- Setup ----
    key   = np.array([1, 0, 0, 0], dtype=complex)   # |00>
    value = np.array([0, 1, 0, 0], dtype=complex)    # |01>

    query_aligned    = np.array([1, 0, 0, 0], dtype=complex)
    query_orthogonal = np.array([0, 0, 1, 0], dtype=complex)
    query_partial    = normalize(np.array([1, 0, 1, 0], dtype=complex))

    shots = 16384
    target_bitstring = '01'  # value = |01> -> Qiskit measures "01"

    queries = [
        ("Aligned (|00>)", query_aligned),
        ("Orthogonal (|10>)", query_orthogonal),
        ("Partial overlap", query_partial),
    ]

    # ---- Statevector (exact, for reference) ----
    print("\n" + "=" * 78)
    print("EXACT STATEVECTOR ANALYSIS (mathematical, no circuit)")
    print("=" * 78)
    sv_aligned = analyze_statevector(key, value, query_aligned, "Aligned")
    sv_ortho = analyze_statevector(key, value, query_orthogonal, "Orthogonal")
    sv_partial = analyze_statevector(key, value, query_partial, "Partial")

    # ---- Check for real hardware ----
    print("\n--- Checking for IBM Quantum access ---")
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backends = service.backends(min_num_qubits=4, operational=True)
        if backends:
            print(f"  Found {len(backends)} available backends!")
    except Exception as e:
        print(f"  No IBM Quantum access: {e}")
        print("  Using simulated hardware noise instead.")

    # ---- Run experiments ----
    results = {}
    transpiled_circuits = {}

    for label, query in queries:
        qc = build_pam_circuit(key, value, query, label=label)

        print(f"\n{'='*60}")
        print(f"QUERY: {label}")
        print(f"{'='*60}")

        # 1) Ideal
        t0 = time.time()
        counts_ideal = run_on_ideal(qc, shots=shots)
        t_ideal = time.time() - t0
        probs_ideal = counts_to_probs(counts_ideal)
        print(f"\n  [Ideal Simulator] ({t_ideal:.2f}s)")
        for bs in ['00', '01', '10', '11']:
            marker = " <-- target" if bs == target_bitstring and probs_ideal[bs] > 0.4 else ""
            print(f"    |{bs}> : {probs_ideal[bs]:.4f}{marker}")

        # 2) GenericBackendV2
        t0 = time.time()
        counts_generic, qc_generic, _ = run_on_generic_backend(qc, shots=shots)
        t_generic = time.time() - t0
        probs_generic = counts_to_probs(counts_generic)
        stats_generic = circuit_stats(qc_generic)
        fid_generic = estimate_fidelity(stats_generic)
        print(f"\n  [GenericBackendV2 w/ noise] ({t_generic:.2f}s)")
        print(f"    Depth: {stats_generic['depth']}, "
              f"ECR: {stats_generic['n_cx']}, "
              f"1Q: {stats_generic['n_1q']}, "
              f"Est.fidelity: {fid_generic:.4f}")
        for bs in ['00', '01', '10', '11']:
            print(f"    |{bs}> : {probs_generic[bs]:.4f}")

        # 3) Brisbane noise model
        t0 = time.time()
        counts_bris, qc_bris, _ = run_on_brisbane_noise(qc, shots=shots)
        t_bris = time.time() - t0
        probs_bris = counts_to_probs(counts_bris)
        stats_bris = circuit_stats(qc_bris)
        fid_bris = estimate_fidelity(stats_bris)
        print(f"\n  [Brisbane Noise Model] ({t_bris:.2f}s)")
        print(f"    Depth: {stats_bris['depth']}, "
              f"ECR: {stats_bris['n_cx']}, "
              f"1Q: {stats_bris['n_1q']}, "
              f"Est.fidelity: {fid_bris:.4f}")
        for bs in ['00', '01', '10', '11']:
            print(f"    |{bs}> : {probs_bris[bs]:.4f}")

        results[label] = {
            'ideal': probs_ideal,
            'generic': probs_generic,
            'brisbane': probs_bris,
            'stats_generic': stats_generic,
            'stats_brisbane': stats_bris,
            'fid_generic': fid_generic,
            'fid_brisbane': fid_bris,
        }
        transpiled_circuits[label] = {
            'generic': qc_generic,
            'brisbane': qc_bris,
        }

    display_qc = transpiled_circuits["Aligned (|00>)"]['brisbane']
    stats_display = circuit_stats(display_qc)

    # ---- Noise scaling sweep ----
    print("\n" + "=" * 78)
    print("NOISE SCALING SWEEP (Brisbane noise x multiplier)")
    print("=" * 78)

    noise_scales = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    sweep_fids = []
    qc_sweep = build_pam_circuit(key, value, query_aligned, label="sweep")

    for scale in noise_scales:
        if scale == 0.0:
            counts = run_on_ideal(qc_sweep, shots=shots)
        else:
            sp = BRISBANE_PARAMS.copy()
            sp["T1_us"] = BRISBANE_PARAMS["T1_us"] / scale
            sp["T2_us"] = BRISBANE_PARAMS["T2_us"] / scale
            sp["2q_error"] = min(BRISBANE_PARAMS["2q_error"] * scale, 0.5)
            sp["readout_error"] = min(BRISBANE_PARAMS["readout_error"] * scale, 0.5)

            nm = build_brisbane_noise_model(8, sp)
            coupling = build_brisbane_coupling_map_subset(8)
            sim = AerSimulator(
                noise_model=nm,
                coupling_map=coupling,
                basis_gates=BRISBANE_PARAMS["basis_gates"] + ["reset", "delay", "measure"],
            )
            qc_t = transpile(
                qc_sweep, sim,
                basis_gates=BRISBANE_PARAMS["basis_gates"],
                coupling_map=coupling,
                optimization_level=2,
                seed_transpiler=42,
            )
            result = sim.run(qc_t, shots=shots, seed_simulator=42).result()
            counts = result.get_counts()

        probs = counts_to_probs(counts)
        fid = probs.get(target_bitstring, 0.0)
        sweep_fids.append(fid)
        print(f"  scale={scale:.2f}  ->  P(|{target_bitstring}>) = {fid:.4f}")

    # ===========================================================================
    # VISUALIZATION
    # ===========================================================================
    print("\n" + "=" * 78)
    print("GENERATING FIGURE...")
    print("=" * 78)

    fig = plt.figure(figsize=(22, 20))
    fig.patch.set_facecolor('#fafafa')
    fig.suptitle(
        "Quantum Phase-Associative Memory (PAM) on Simulated IBM Hardware\n"
        "d=4 circuit transpiled to Brisbane-class Eagle r3 noise profile",
        fontsize=16, fontweight='bold', y=0.98, color='#1a1a2e',
    )

    gs = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35,
                  top=0.93, bottom=0.04, left=0.07, right=0.97)

    bitstrings = ['00', '01', '10', '11']
    states = ['|00>', '|01>', '|10>', '|11>']
    colors_b = {'ideal': '#2196F3', 'generic': '#FF9800', 'brisbane': '#F44336'}

    # ---- Panel 0: Transpiled circuit ----
    ax_circ = fig.add_subplot(gs[0, :])
    try:
        display_qc.draw(output='mpl', ax=ax_circ, style='iqp', fold=80)
        ax_circ.set_title(
            f"Transpiled Circuit for Brisbane (aligned query)\n"
            f"Depth: {stats_display['depth']}  |  "
            f"ECR/CX: {stats_display['n_cx']}  |  "
            f"1Q gates: {stats_display['n_1q']}  |  "
            f"Total: {stats_display['total_gates']}",
            fontsize=11, fontweight='bold', color='#333'
        )
    except Exception as e:
        ax_circ.text(0.02, 0.95,
                     f"Transpiled Circuit Statistics (Brisbane target)\n"
                     f"{'='*55}\n"
                     f"  Depth:       {stats_display['depth']}\n"
                     f"  ECR/CX:      {stats_display['n_cx']}\n"
                     f"  1Q gates:    {stats_display['n_1q']}\n"
                     f"  Total gates: {stats_display['total_gates']}\n"
                     f"  Gate ops:    {stats_display['gate_count']}\n\n"
                     f"  (Graphical render error: {e})",
                     transform=ax_circ.transAxes, fontsize=9,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
        ax_circ.set_xlim(0, 1); ax_circ.set_ylim(0, 1); ax_circ.axis('off')

    # ---- Panels 1-3: Per-query measurement results ----
    for idx, (label, query) in enumerate(queries):
        ax = fig.add_subplot(gs[1, idx])
        r = results[label]

        x_pos = np.arange(len(states))
        w = 0.25

        bars_i = ax.bar(x_pos - w, [r['ideal'][b] for b in bitstrings],
                        w, label='Ideal', color=colors_b['ideal'],
                        edgecolor='black', linewidth=0.5, alpha=0.9)
        bars_g = ax.bar(x_pos, [r['generic'][b] for b in bitstrings],
                        w, label='GenericBackendV2', color=colors_b['generic'],
                        edgecolor='black', linewidth=0.5, alpha=0.9)
        bars_b = ax.bar(x_pos + w, [r['brisbane'][b] for b in bitstrings],
                        w, label='Brisbane noise', color=colors_b['brisbane'],
                        edgecolor='black', linewidth=0.5, alpha=0.9)

        ax.set_xlabel('Value Register Outcome', fontsize=9)
        ax.set_ylabel('Probability', fontsize=9)
        ax.set_title(f'{label}', fontweight='bold', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(states)
        ax.legend(fontsize=7, loc='upper right')
        ax.set_ylim(0, 1.15)
        ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

        # Annotate probabilities on ideal bars
        for bar in bars_i:
            h = bar.get_height()
            if h > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=7,
                        color=colors_b['ideal'], fontweight='bold')

        # Annotate the target for aligned query
        if "Aligned" in label:
            target_idx = bitstrings.index(target_bitstring)
            ax.annotate('TARGET\nVALUE', xy=(target_idx - w, r['ideal'][target_bitstring]),
                        xytext=(target_idx + 1, 0.85), fontsize=8, fontweight='bold',
                        color='green',
                        arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    # ---- Panel 4: Retrieval fidelity comparison ----
    ax_fid = fig.add_subplot(gs[2, 0])
    query_short = ['Aligned', 'Orthogonal', 'Partial']

    fid_i = [results[l]['ideal'][target_bitstring] for l, _ in queries]
    fid_g = [results[l]['generic'][target_bitstring] for l, _ in queries]
    fid_b = [results[l]['brisbane'][target_bitstring] for l, _ in queries]

    x_q = np.arange(3)
    w = 0.25
    ax_fid.bar(x_q - w, fid_i, w, label='Ideal', color=colors_b['ideal'],
               edgecolor='black', linewidth=0.5)
    ax_fid.bar(x_q, fid_g, w, label='GenericBackendV2', color=colors_b['generic'],
               edgecolor='black', linewidth=0.5)
    ax_fid.bar(x_q + w, fid_b, w, label='Brisbane noise', color=colors_b['brisbane'],
               edgecolor='black', linewidth=0.5)

    ax_fid.set_xlabel('Query Type')
    ax_fid.set_ylabel(f'P(|{target_bitstring}>) -- Retrieval Prob.')
    ax_fid.set_title(f'Value Retrieval: P(correct = |{target_bitstring}>)', fontweight='bold')
    ax_fid.set_xticks(x_q)
    ax_fid.set_xticklabels(query_short)
    ax_fid.legend(fontsize=8)
    ax_fid.set_ylim(0, 1.15)
    ax_fid.axhline(y=0.25, color='gray', linestyle='--', alpha=0.4)

    for i, (fi, fg, fb) in enumerate(zip(fid_i, fid_g, fid_b)):
        for val, offset, col in [(fi, -w, colors_b['ideal']),
                                  (fg, 0, colors_b['generic']),
                                  (fb, w, colors_b['brisbane'])]:
            if val > 0.02:
                ax_fid.text(i + offset, val + 0.02, f'{val:.2f}',
                            ha='center', va='bottom', fontsize=7,
                            color=col, fontweight='bold')

    # ---- Panel 5: Noise scaling sweep ----
    ax_sweep = fig.add_subplot(gs[2, 1])
    ax_sweep.plot(noise_scales, sweep_fids, 'o-', color='#E91E63',
                  linewidth=2.5, markersize=8,
                  markeredgecolor='black', markeredgewidth=0.5)
    ax_sweep.axhline(y=1.0, color='#2196F3', linestyle='--', alpha=0.5, label='Ideal (P=1.0)')
    ax_sweep.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Random (1/d=0.25)')
    ax_sweep.axvline(x=1.0, color='#F44336', linestyle=':', alpha=0.7, label='Brisbane nominal')
    ax_sweep.fill_between([0.8, 1.2], [0, 0], [1.15, 1.15],
                          alpha=0.08, color='red')
    ax_sweep.set_xlabel('Noise Scale (1.0 = Brisbane nominal)')
    ax_sweep.set_ylabel(f'Retrieval Fidelity P(|{target_bitstring}>)')
    ax_sweep.set_title('Noise Scaling: Brisbane-class\n(aligned query)', fontweight='bold')
    ax_sweep.legend(fontsize=8, loc='upper right')
    ax_sweep.set_ylim(0, 1.15)
    ax_sweep.grid(True, alpha=0.3)

    # ---- Panel 6: Circuit depth comparison ----
    ax_depth = fig.add_subplot(gs[2, 2])
    depth_labels = ['Aligned', 'Orthogonal', 'Partial']
    depths_i, depths_g, depths_b = [], [], []

    for label, query in queries:
        qc_tmp = build_pam_circuit(key, value, query, label="tmp")
        qc_it = transpile(qc_tmp, AerSimulator())
        depths_i.append(qc_it.depth())
        depths_g.append(results[label]['stats_generic']['depth'])
        depths_b.append(results[label]['stats_brisbane']['depth'])

    x_d = np.arange(3)
    w_d = 0.25
    ax_depth.bar(x_d - w_d, depths_i, w_d, label='Ideal (Aer)', color=colors_b['ideal'],
                 edgecolor='black', linewidth=0.5)
    ax_depth.bar(x_d, depths_g, w_d, label='GenericBackendV2', color=colors_b['generic'],
                 edgecolor='black', linewidth=0.5)
    ax_depth.bar(x_d + w_d, depths_b, w_d, label='Brisbane', color=colors_b['brisbane'],
                 edgecolor='black', linewidth=0.5)

    ax_depth.set_xlabel('Query Type')
    ax_depth.set_ylabel('Circuit Depth (after transpilation)')
    ax_depth.set_title('Transpiled Circuit Depth\n(deeper = more noise exposure)', fontweight='bold')
    ax_depth.set_xticks(x_d)
    ax_depth.set_xticklabels(depth_labels, fontsize=9)
    ax_depth.legend(fontsize=8)
    ax_depth.grid(True, alpha=0.2, axis='y')

    # ---- Panel 7: Summary + how-to ----
    ax_sum = fig.add_subplot(gs[3, :])
    ax_sum.axis('off')

    ra = results["Aligned (|00>)"]
    ro = results["Orthogonal (|10>)"]
    rp = results["Partial overlap"]

    ortho_noise = max(ro['brisbane'][target_bitstring], 1e-6)
    snr_ideal = ra['ideal'][target_bitstring] / max(ro['ideal'][target_bitstring], 1e-6)
    snr_generic = ra['generic'][target_bitstring] / max(ro['generic'][target_bitstring], 1e-6)
    snr_brisbane = ra['brisbane'][target_bitstring] / ortho_noise

    summary_left = (
        "HARDWARE SIMULATION RESULTS\n"
        "=" * 52 + "\n\n"
        f"Retrieval P(|{target_bitstring}>) -- correct value:\n"
        f"                Ideal   GenericV2  Brisbane\n"
        f"  Aligned:      {ra['ideal'][target_bitstring]:.4f}    {ra['generic'][target_bitstring]:.4f}     {ra['brisbane'][target_bitstring]:.4f}\n"
        f"  Orthogonal:   {ro['ideal'][target_bitstring]:.4f}    {ro['generic'][target_bitstring]:.4f}     {ro['brisbane'][target_bitstring]:.4f}\n"
        f"  Partial:      {rp['ideal'][target_bitstring]:.4f}    {rp['generic'][target_bitstring]:.4f}     {rp['brisbane'][target_bitstring]:.4f}\n\n"
        f"Signal-to-noise (aligned/orthogonal):\n"
        f"  Ideal:      {snr_ideal:.1f}x\n"
        f"  GenericV2:  {snr_generic:.1f}x\n"
        f"  Brisbane:   {snr_brisbane:.1f}x\n\n"
        f"Transpiled circuit (Brisbane):\n"
        f"  Depth:     {ra['stats_brisbane']['depth']}\n"
        f"  ECR gates: {ra['stats_brisbane']['n_cx']}\n"
        f"  1Q gates:  {ra['stats_brisbane']['n_1q']}\n"
        f"  Est. fid:  {ra['fid_brisbane']:.4f}\n"
    )

    summary_right = (
        "HOW TO RUN ON REAL IBM HARDWARE\n"
        "=" * 52 + "\n\n"
        "1. Create free account: quantum.ibm.com\n"
        "2. pip install qiskit-ibm-runtime\n"
        "3. Save your API token:\n"
        "   from qiskit_ibm_runtime import (\n"
        "       QiskitRuntimeService)\n"
        "   QiskitRuntimeService.save_account(\n"
        '       channel="ibm_quantum",\n'
        '       token="YOUR_TOKEN")\n'
        "4. Run on real hardware:\n"
        "   service = QiskitRuntimeService()\n"
        "   backend = service.least_busy(\n"
        "       min_num_qubits=4)\n"
        "   from qiskit_ibm_runtime import SamplerV2\n"
        "   sampler = SamplerV2(mode=backend)\n"
        "   job = sampler.run([qc_transpiled])\n\n"
        "Free backends (2026):\n"
        "  ibm_brisbane, ibm_kyoto, ibm_osaka\n"
        "  (127q Eagle r3, ECR basis gates)\n"
    )

    ax_sum.text(0.01, 0.98, summary_left, transform=ax_sum.transAxes,
                fontsize=8.5, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#e3f2fd', alpha=0.9,
                          edgecolor='#1565C0', linewidth=1.5))

    ax_sum.text(0.52, 0.98, summary_right, transform=ax_sum.transAxes,
                fontsize=8.5, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3e0', alpha=0.9,
                          edgecolor='#E65100', linewidth=1.5))

    # ---- Save ----
    out_dir = '/home/caug/npcww/qstk/results'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'quantum_pam_hardware.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\nFigure saved to: {out_path}")
    plt.close(fig)

    # ---- Final report ----
    print("\n" + "=" * 78)
    print("HARDWARE SIMULATION REPORT")
    print("=" * 78)
    print(f"\nBrisbane noise parameters:")
    for k, v in BRISBANE_PARAMS.items():
        print(f"  {k}: {v}")
    print(f"\nRetrieval fidelity P(|{target_bitstring}>) under Brisbane noise:")
    print(f"  Aligned:    {ra['brisbane'][target_bitstring]:.4f} (ideal: {ra['ideal'][target_bitstring]:.4f})")
    print(f"  Orthogonal: {ro['brisbane'][target_bitstring]:.4f} (ideal: {ro['ideal'][target_bitstring]:.4f})")
    print(f"  Partial:    {rp['brisbane'][target_bitstring]:.4f} (ideal: {rp['ideal'][target_bitstring]:.4f})")
    print(f"\nSignal-to-noise ratio (aligned / orthogonal P(|{target_bitstring}>)):")
    print(f"  Ideal:      {snr_ideal:.1f}x")
    print(f"  GenericV2:  {snr_generic:.1f}x")
    print(f"  Brisbane:   {snr_brisbane:.1f}x")

    surviving = snr_brisbane > 2.0
    print(f"\nConclusion: Phase-interference retrieval {'SURVIVES' if surviving else 'is degraded under'}")
    print(f"  realistic Brisbane-class hardware noise.")
    if surviving:
        print(f"  The aligned-vs-orthogonal discrimination remains clear ({snr_brisbane:.1f}x).")
        print(f"  PAM mechanism is hardware-viable for d=4.")
    else:
        print(f"  SNR of {snr_brisbane:.1f}x indicates significant noise impact at this circuit depth.")
        print(f"  Recommendations:")
        print(f"    - Dynamical decoupling (DD sequences during idle times)")
        print(f"    - Readout error mitigation (M3, TREX)")
        print(f"    - Zero-noise extrapolation (ZNE)")
        print(f"    - Pauli twirling for 2-qubit gates")
        print(f"    - Optimized qubit mapping (Mapomatic)")

    return results


if __name__ == '__main__':
    results = main()
