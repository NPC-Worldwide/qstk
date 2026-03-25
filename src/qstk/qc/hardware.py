"""Quantum hardware backends — IBM Quantum and Amazon Braket runners.

Provides a unified interface for submitting CHSH/Bell circuits to:
- NumPy statevector simulation (always available, free)
- IBM Quantum via qiskit-ibm-runtime (free tier: 10 min/28 days)
- Amazon Braket (IonQ, Rigetti, IQM — pay-per-shot)

All runners return a standardized HardwareResult for easy comparison.

Each CHSH experiment runs 4 circuits (one per measurement setting pair),
computes expectation values from raw bitstring counts, and derives S.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

import numpy as np

from .circuits import bell_circuit, chsh_circuit, to_qiskit
from .measure import chsh_s_value, measure_state


# ═══════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HardwareResult:
    """Standardized result from any quantum backend."""
    backend: str               # "numpy", "ibm_<name>", "braket_<device>"
    backend_type: str          # "simulator", "trapped_ion", "superconducting"
    counts: Dict[str, Dict[str, int]]  # per-setting counts: {"A_B": {"00": 512, ...}, ...}
    n_shots: int
    s_value: float             # CHSH S-value computed from counts
    expectation_values: Dict[str, float]  # E(Ai,Bj) per setting
    execution_time_s: float    # wall-clock time
    cost_estimate_usd: float   # estimated cost
    metadata: Dict[str, Any]   # backend-specific info (job_id, etc.)
    timestamp: float


@dataclass
class CostEstimate:
    """Cost estimate for a quantum experiment."""
    backend: str
    n_shots: int
    n_circuits: int
    estimated_usd: float
    pricing_model: str         # "free", "per_minute", "per_shot", "per_task_plus_shot"
    notes: str


# ═══════════════════════════════════════════════════════════════════
# Cost Estimation
# ═══════════════════════════════════════════════════════════════════

# Pricing data (as of early 2026)
PRICING = {
    # IBM Quantum
    "ibm_free": {"model": "per_minute", "rate": 0.0, "limit": "10 min / 28 days"},
    "ibm_paygo": {"model": "per_minute", "rate": 96.0},
    "ibm_flex": {"model": "per_minute", "rate": 48.0, "minimum": 30000},

    # Amazon Braket (per_task=$0.30 + per_shot varies)
    "braket_rigetti": {"model": "per_task_plus_shot", "task": 0.30, "shot": 0.00090},
    "braket_iqm_garnet": {"model": "per_task_plus_shot", "task": 0.30, "shot": 0.00145},
    "braket_iqm_emerald": {"model": "per_task_plus_shot", "task": 0.30, "shot": 0.00160},
    "braket_ionq_forte": {"model": "per_task_plus_shot", "task": 0.30, "shot": 0.08000},
    "braket_quera": {"model": "per_task_plus_shot", "task": 0.30, "shot": 0.01000},

    # Simulators
    "numpy": {"model": "free", "rate": 0.0},
    "braket_sv1": {"model": "per_minute", "rate": 0.075, "minimum_seconds": 3},
}


def estimate_cost(
    backend: str,
    n_shots: int = 1024,
    n_circuits: int = 4,
    estimated_runtime_s: float = 10.0,
) -> CostEstimate:
    """Estimate cost for running an experiment on a given backend.

    Parameters
    ----------
    backend : str
        Backend key from PRICING dict.
    n_shots : int
        Shots per circuit.
    n_circuits : int
        Number of circuits (4 for CHSH = one per measurement setting).
    estimated_runtime_s : float
        Estimated QPU runtime in seconds (for per-minute pricing).

    Returns
    -------
    CostEstimate dataclass.
    """
    if backend not in PRICING:
        return CostEstimate(backend, n_shots, n_circuits, 0.0, "unknown",
                            f"Unknown backend '{backend}'. Known: {list(PRICING.keys())}")

    p = PRICING[backend]
    model = p["model"]

    if model == "free":
        return CostEstimate(backend, n_shots, n_circuits, 0.0, model, "Free (local simulation)")

    elif model == "per_minute":
        minutes = estimated_runtime_s / 60
        cost = minutes * p["rate"]
        limit = p.get("limit", "")
        notes = f"${p['rate']}/min QPU time"
        if limit:
            notes += f" (free tier: {limit})"
        return CostEstimate(backend, n_shots, n_circuits, cost, model, notes)

    elif model == "per_task_plus_shot":
        task_cost = p["task"] * n_circuits
        shot_cost = p["shot"] * n_shots * n_circuits
        total = task_cost + shot_cost
        return CostEstimate(backend, n_shots, n_circuits, total, model,
                            f"${p['task']}/task + ${p['shot']}/shot. "
                            f"Tasks: {n_circuits} x ${p['task']} = ${task_cost:.2f}, "
                            f"Shots: {n_circuits} x {n_shots} x ${p['shot']} = ${shot_cost:.2f}")

    return CostEstimate(backend, n_shots, n_circuits, 0.0, "unknown", "")


def estimate_chsh_experiment_cost(
    backend: str,
    n_shots: int = 1024,
    n_bell_states: int = 1,
    n_angle_settings: int = 1,
) -> CostEstimate:
    """Estimate cost for a full CHSH experiment.

    A CHSH experiment requires 4 circuits (one per operator pair).
    Multiply by number of Bell states and angle settings tested.
    """
    n_circuits = 4 * n_bell_states * n_angle_settings
    # Rough runtime estimate: ~2s per circuit on IBM, ~5s on trapped ion
    if "ionq" in backend:
        est_runtime = n_circuits * 5
    elif "ibm" in backend:
        est_runtime = n_circuits * 2
    else:
        est_runtime = n_circuits * 1

    return estimate_cost(backend, n_shots, n_circuits, est_runtime)


def print_cost_comparison(n_shots: int = 1024, n_circuits: int = 4):
    """Print a cost comparison table across all backends."""
    print(f"\nCost estimates for {n_circuits} circuits x {n_shots} shots:\n")
    print(f"{'Backend':<25} {'Cost':>10} {'Model':<25} Notes")
    print("-" * 90)
    for backend in sorted(PRICING.keys()):
        est = estimate_cost(backend, n_shots, n_circuits)
        print(f"{backend:<25} ${est.estimated_usd:>8.2f} {est.pricing_model:<25} {est.notes}")


# ═══════════════════════════════════════════════════════════════════
# Job Tracking
# ═══════════════════════════════════════════════════════════════════

class ExperimentLog:
    """Track costs and results across multiple hardware runs."""

    def __init__(self, log_path: str = "qstk_hardware_log.json"):
        self.log_path = Path(log_path)
        self.entries: List[Dict] = []
        if self.log_path.exists():
            with open(self.log_path) as f:
                self.entries = json.load(f)

    def log_result(self, result: HardwareResult):
        self.entries.append(asdict(result))
        self._save()

    def _save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.entries, f, indent=2, default=str)

    def total_cost(self) -> float:
        return sum(e.get("cost_estimate_usd", 0) for e in self.entries)

    def summary(self) -> Dict[str, Any]:
        if not self.entries:
            return {"n_runs": 0, "total_cost": 0.0}
        by_backend = {}
        for e in self.entries:
            b = e["backend"]
            if b not in by_backend:
                by_backend[b] = {"runs": 0, "cost": 0.0, "total_shots": 0}
            by_backend[b]["runs"] += 1
            by_backend[b]["cost"] += e.get("cost_estimate_usd", 0)
            by_backend[b]["total_shots"] += e.get("n_shots", 0)
        return {
            "n_runs": len(self.entries),
            "total_cost": self.total_cost(),
            "by_backend": by_backend,
        }

    def print_summary(self):
        s = self.summary()
        print(f"\nExperiment Log: {s['n_runs']} runs, ${s['total_cost']:.2f} total")
        for backend, info in s.get("by_backend", {}).items():
            print(f"  {backend}: {info['runs']} runs, {info['total_shots']} shots, ${info['cost']:.2f}")


# ═══════════════════════════════════════════════════════════════════
# Computing S-value from raw measurement counts
# ═══════════════════════════════════════════════════════════════════

# CHSH measurement settings: (alice_angle, bob_angle)
DEFAULT_CHSH_SETTINGS = {
    "A_B":             (0.0, np.pi / 4),
    "A_B_prime":       (0.0, 3 * np.pi / 4),
    "A_prime_B":       (np.pi / 2, np.pi / 4),
    "A_prime_B_prime": (np.pi / 2, 3 * np.pi / 4),
}


def expectation_from_counts(counts: Dict[str, int]) -> float:
    """Compute ⟨AB⟩ from bitstring counts.

    Mapping: |0⟩ → +1, |1⟩ → -1.
    Product outcome: |00⟩,|11⟩ → +1; |01⟩,|10⟩ → -1.

    Parameters
    ----------
    counts : dict
        Bitstring counts like {"00": 512, "01": 12, "10": 8, "11": 492}.

    Returns
    -------
    float : expectation value in [-1, +1].
    """
    agree = counts.get("00", 0) + counts.get("11", 0)
    disagree = counts.get("01", 0) + counts.get("10", 0)
    total = agree + disagree
    if total == 0:
        return 0.0
    return (agree - disagree) / total


def s_value_from_counts(
    all_counts: Dict[str, Dict[str, int]],
) -> tuple:
    """Compute CHSH S-value from per-setting measurement counts.

    Parameters
    ----------
    all_counts : dict
        Keys are setting labels ("A_B", "A_B_prime", "A_prime_B", "A_prime_B_prime"),
        values are bitstring count dicts.

    Returns
    -------
    (s_value, expectation_values) tuple.
    """
    ev = {}
    for label in ["A_B", "A_B_prime", "A_prime_B", "A_prime_B_prime"]:
        ev[label] = expectation_from_counts(all_counts[label])

    s = ev["A_B"] - ev["A_B_prime"] + ev["A_prime_B"] + ev["A_prime_B_prime"]
    return s, ev


# ═══════════════════════════════════════════════════════════════════
# Backend Runners
# ═══════════════════════════════════════════════════════════════════

def run_numpy(
    state_type: str = "phi_plus",
    n_shots: int = 1024,
    a0_angle: float = 0.0,
    a1_angle: float = np.pi / 2,
    b0_angle: float = np.pi / 4,
    b1_angle: float = 3 * np.pi / 4,
    seed: Optional[int] = None,
) -> HardwareResult:
    """Run CHSH experiment on numpy statevector simulator (always free).

    Returns both the exact S-value and shot-sampled S-value.
    """
    t0 = time.time()
    result = chsh_circuit(state_type, a0_angle, a1_angle, b0_angle, b1_angle,
                          n_shots=n_shots, seed=seed)
    elapsed = time.time() - t0

    return HardwareResult(
        backend="numpy",
        backend_type="simulator",
        counts=result["measurement_counts"],
        n_shots=n_shots,
        s_value=result["s_value"],
        expectation_values=result["expectation_values"],
        execution_time_s=elapsed,
        cost_estimate_usd=0.0,
        metadata={
            "state_type": state_type,
            "seed": seed,
            "exact_s_value": result["s_value"],
            "angles": {
                "a0": a0_angle, "a1": a1_angle,
                "b0": b0_angle, "b1": b1_angle,
            },
        },
        timestamp=time.time(),
    )


def _build_qiskit_chsh_circuits(
    state_type: str = "phi_plus",
    a0_angle: float = 0.0,
    a1_angle: float = np.pi / 2,
    b0_angle: float = np.pi / 4,
    b1_angle: float = 3 * np.pi / 4,
):
    """Build 4 Qiskit circuits for CHSH measurement settings.

    Each circuit: Bell state prep → Ry(-θ_alice) on q0 → Ry(-θ_bob) on q1 → measure.

    To measure operator M(θ) = cos(θ)σ_z + sin(θ)σ_x in the Z basis,
    apply Ry(-θ) before measurement.

    Returns
    -------
    dict mapping setting label → (QuantumCircuit, alice_angle, bob_angle)
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import HGate, XGate, ZGate, CXGate, RYGate

    settings = {
        "A_B":             (a0_angle, b0_angle),
        "A_B_prime":       (a0_angle, b1_angle),
        "A_prime_B":       (a1_angle, b0_angle),
        "A_prime_B_prime": (a1_angle, b1_angle),
    }

    circuits = {}
    for label, (alice_angle, bob_angle) in settings.items():
        qc = QuantumCircuit(2, 2)

        # Bell state preparation
        qc.h(0)
        qc.cx(0, 1)

        # Phase gates for different Bell states
        if state_type in ("phi_minus", "Φ-"):
            qc.z(0)
        elif state_type in ("psi_plus", "Ψ+"):
            qc.x(1)
        elif state_type in ("psi_minus", "Ψ-"):
            qc.z(0)
            qc.x(1)

        # Measurement basis rotations: Ry(-θ) before Z-measurement
        if abs(alice_angle) > 1e-12:
            qc.ry(-alice_angle, 0)
        if abs(bob_angle) > 1e-12:
            qc.ry(-bob_angle, 1)

        # Measure
        qc.measure(0, 0)
        qc.measure(1, 1)

        circuits[label] = qc

    return circuits


def _build_braket_chsh_circuits(
    state_type: str = "phi_plus",
    a0_angle: float = 0.0,
    a1_angle: float = np.pi / 2,
    b0_angle: float = np.pi / 4,
    b1_angle: float = 3 * np.pi / 4,
):
    """Build 4 Braket circuits for CHSH measurement settings.

    Returns
    -------
    dict mapping setting label → braket.circuits.Circuit
    """
    from braket.circuits import Circuit as BraketCircuit

    settings = {
        "A_B":             (a0_angle, b0_angle),
        "A_B_prime":       (a0_angle, b1_angle),
        "A_prime_B":       (a1_angle, b0_angle),
        "A_prime_B_prime": (a1_angle, b1_angle),
    }

    circuits = {}
    for label, (alice_angle, bob_angle) in settings.items():
        bc = BraketCircuit()

        # Bell state preparation
        bc.h(0)
        bc.cnot(0, 1)

        # Phase gates for different Bell states
        if state_type in ("phi_minus", "Φ-"):
            bc.z(0)
        elif state_type in ("psi_plus", "Ψ+"):
            bc.x(1)
        elif state_type in ("psi_minus", "Ψ-"):
            bc.z(0)
            bc.x(1)

        # Measurement basis rotations: Ry(-θ) before Z-measurement
        if abs(alice_angle) > 1e-12:
            bc.ry(0, -alice_angle)
        if abs(bob_angle) > 1e-12:
            bc.ry(1, -bob_angle)

        circuits[label] = bc

    return circuits


def _counts_from_qiskit_result(result, circuit_index: int) -> Dict[str, int]:
    """Extract bitstring counts from a Qiskit SamplerV2 result.

    Handles the modern Qiskit result format where bits may be returned
    as classical register arrays.
    """
    pub_result = result[circuit_index]

    # Try modern SamplerV2 format
    try:
        # SamplerV2 returns BitArray objects
        data = pub_result.data
        # Get the classical register (could be 'meas' or 'c')
        for attr in ("meas", "c", "c0"):
            if hasattr(data, attr):
                bit_array = getattr(data, attr)
                return dict(bit_array.get_counts())
        # Fallback: try first available register
        if hasattr(data, '__iter__'):
            for key in data:
                return dict(data[key].get_counts())
    except (AttributeError, TypeError):
        pass

    # Fallback for older result formats
    try:
        counts = pub_result.data.get_counts()
        return dict(counts)
    except (AttributeError, TypeError):
        pass

    raise ValueError(f"Could not extract counts from Qiskit result at index {circuit_index}")


def run_ibm(
    state_type: str = "phi_plus",
    n_shots: int = 1024,
    a0_angle: float = 0.0,
    a1_angle: float = np.pi / 2,
    b0_angle: float = np.pi / 4,
    b1_angle: float = 3 * np.pi / 4,
    backend_name: Optional[str] = None,
    use_error_mitigation: bool = True,
    log: Optional[ExperimentLog] = None,
) -> HardwareResult:
    """Run CHSH experiment on IBM Quantum hardware.

    Builds 4 circuits (one per measurement setting), submits them as a batch,
    and computes S-value from the measured bitstring counts.

    Requires: pip install qiskit-ibm-runtime
    Auth: QiskitRuntimeService.save_account(token="YOUR_TOKEN")

    Parameters
    ----------
    state_type : str
        Bell state type.
    n_shots : int
        Shots per circuit.
    a0_angle, a1_angle, b0_angle, b1_angle : float
        Measurement angles.
    backend_name : str, optional
        Specific backend (e.g. "ibm_brisbane"). None = least busy.
    use_error_mitigation : bool
        Use built-in error mitigation (resilience_level=1).
    log : ExperimentLog, optional
        Log to track costs.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler import generate_preset_pass_manager

    service = QiskitRuntimeService()

    if backend_name:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=2)

    print(f"Using IBM backend: {backend.name}")

    # Build the 4 CHSH measurement circuits
    circuits = _build_qiskit_chsh_circuits(
        state_type, a0_angle, a1_angle, b0_angle, b1_angle
    )
    labels = list(circuits.keys())  # preserve order

    # Transpile all circuits to backend ISA
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa_circuits = [pm.run(circuits[label]) for label in labels]

    # Submit all 4 circuits as a batch
    t0 = time.time()
    sampler = SamplerV2(mode=backend)
    if use_error_mitigation:
        sampler.options.resilience_level = 1

    pubs = [(circ,) for circ in isa_circuits]
    job = sampler.run(pubs, shots=n_shots)
    print(f"Job submitted: {job.job_id()}")
    result = job.result()
    elapsed = time.time() - t0

    # Extract counts per setting
    all_counts = {}
    for i, label in enumerate(labels):
        all_counts[label] = _counts_from_qiskit_result(result, i)

    # Compute S-value from hardware counts
    s_value, ev = s_value_from_counts(all_counts)

    # Also get theoretical S for comparison
    numpy_result = chsh_circuit(state_type, a0_angle, a1_angle, b0_angle, b1_angle, n_shots=0)

    hw_result = HardwareResult(
        backend=f"ibm_{backend.name}",
        backend_type="superconducting",
        counts=all_counts,
        n_shots=n_shots,
        s_value=s_value,
        expectation_values=ev,
        execution_time_s=elapsed,
        cost_estimate_usd=0.0,  # free tier; compute from elapsed if PAYG
        metadata={
            "job_id": job.job_id(),
            "backend_name": backend.name,
            "state_type": state_type,
            "error_mitigation": use_error_mitigation,
            "theoretical_s_value": numpy_result["s_value"],
            "transpiled_depths": [c.depth() for c in isa_circuits],
            "angles": {
                "a0": a0_angle, "a1": a1_angle,
                "b0": b0_angle, "b1": b1_angle,
            },
        },
        timestamp=time.time(),
    )

    if log:
        log.log_result(hw_result)

    return hw_result


def run_braket(
    state_type: str = "phi_plus",
    n_shots: int = 1024,
    a0_angle: float = 0.0,
    a1_angle: float = np.pi / 2,
    b0_angle: float = np.pi / 4,
    b1_angle: float = 3 * np.pi / 4,
    device_arn: Optional[str] = None,
    use_local: bool = False,
    log: Optional[ExperimentLog] = None,
) -> HardwareResult:
    """Run CHSH experiment on Amazon Braket.

    Builds 4 circuits (one per measurement setting), submits them,
    and computes S-value from the measured bitstring counts.

    Requires: pip install amazon-braket-sdk
    Auth: aws configure (AWS CLI)

    Parameters
    ----------
    state_type : str
        Bell state type.
    n_shots : int
        Shots per circuit.
    a0_angle, a1_angle, b0_angle, b1_angle : float
        Measurement angles.
    device_arn : str, optional
        Device ARN. None uses local simulator.
        Examples:
        - "arn:aws:braket:::device/qpu/rigetti/Ankaa-3"
        - "arn:aws:braket:::device/qpu/ionq/Forte-1"
        - "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    use_local : bool
        Force local simulator (free, no AWS charges).
    log : ExperimentLog, optional
        Log to track costs.
    """
    from braket.devices import LocalSimulator

    # Build the 4 CHSH measurement circuits
    circuits = _build_braket_chsh_circuits(
        state_type, a0_angle, a1_angle, b0_angle, b1_angle
    )

    if use_local or device_arn is None:
        device = LocalSimulator()
        backend_label = "braket_local"
        backend_type = "simulator"
        cost = 0.0
    else:
        from braket.aws import AwsDevice
        device = AwsDevice(device_arn)
        if "rigetti" in device_arn.lower():
            backend_label = "braket_rigetti"
            backend_type = "superconducting"
        elif "ionq" in device_arn.lower():
            backend_label = "braket_ionq"
            backend_type = "trapped_ion"
        elif "iqm" in device_arn.lower():
            backend_label = "braket_iqm"
            backend_type = "superconducting"
        else:
            backend_label = f"braket_{device_arn.split('/')[-1]}"
            backend_type = "unknown"
        est = estimate_cost(backend_label, n_shots, 4)
        cost = est.estimated_usd

    # Run all 4 circuits
    t0 = time.time()
    all_counts = {}
    for label, bc in circuits.items():
        task = device.run(bc, shots=n_shots)
        result = task.result()
        all_counts[label] = dict(result.measurement_counts)
    elapsed = time.time() - t0

    # Compute S-value from hardware counts
    s_value, ev = s_value_from_counts(all_counts)

    # Theoretical S for comparison
    numpy_result = chsh_circuit(state_type, a0_angle, a1_angle, b0_angle, b1_angle, n_shots=0)

    hw_result = HardwareResult(
        backend=backend_label,
        backend_type=backend_type,
        counts=all_counts,
        n_shots=n_shots,
        s_value=s_value,
        expectation_values=ev,
        execution_time_s=elapsed,
        cost_estimate_usd=cost,
        metadata={
            "state_type": state_type,
            "device_arn": device_arn or "local",
            "theoretical_s_value": numpy_result["s_value"],
            "angles": {
                "a0": a0_angle, "a1": a1_angle,
                "b0": b0_angle, "b1": b1_angle,
            },
        },
        timestamp=time.time(),
    )

    if log:
        log.log_result(hw_result)

    return hw_result


# ═══════════════════════════════════════════════════════════════════
# Multi-backend comparison
# ═══════════════════════════════════════════════════════════════════

def compare_backends(
    state_type: str = "phi_plus",
    n_shots: int = 1024,
    backends: Optional[List[str]] = None,
    seed: int = 42,
) -> List[HardwareResult]:
    """Run the same experiment across multiple backends and compare.

    Parameters
    ----------
    state_type : str
        Bell state to prepare.
    n_shots : int
        Shots per measurement setting.
    backends : list of str, optional
        Which backends to run. Default: just numpy.
        Options: "numpy", "ibm", "braket_local", "braket_<arn>"
    seed : int
        Seed for numpy simulator.

    Returns
    -------
    List of HardwareResult, one per backend.
    """
    if backends is None:
        backends = ["numpy"]

    results = []
    for backend in backends:
        print(f"\nRunning on {backend}...")
        try:
            if backend == "numpy":
                r = run_numpy(state_type, n_shots, seed=seed)
            elif backend == "ibm":
                r = run_ibm(state_type, n_shots)
            elif backend == "braket_local":
                r = run_braket(state_type, n_shots, use_local=True)
            elif backend.startswith("braket_arn:"):
                arn = backend.replace("braket_", "")
                r = run_braket(state_type, n_shots, device_arn=arn)
            else:
                print(f"  Unknown backend: {backend}")
                continue
            results.append(r)
            print(f"  S-value: {r.s_value:.4f}")
            print(f"  Time: {r.execution_time_s:.2f}s")
            print(f"  Cost: ${r.cost_estimate_usd:.4f}")
            for label, ev in r.expectation_values.items():
                print(f"  E({label}): {ev:.4f}")
        except ImportError as e:
            print(f"  MISSING DEPENDENCY: {e}")
            print(f"  Install the required package to use this backend.")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary comparison
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"{'Backend':<25} {'S-value':>10} {'|S|>2':>8} {'Time':>8} {'Cost':>10}")
        print(f"{'='*60}")
        for r in results:
            violation = "YES" if abs(r.s_value) > 2.0 else "no"
            print(f"{r.backend:<25} {r.s_value:>10.4f} {violation:>8} "
                  f"{r.execution_time_s:>7.2f}s ${r.cost_estimate_usd:>8.4f}")

    return results
