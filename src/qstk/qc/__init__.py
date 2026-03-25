"""Quantum Computing submodule for QSTK.

Provides routines for preparing Bell states, constructing CHSH operators,
running measurements on quantum simulators and real quantum hardware,
encoding semantic states into qubit representations, and comparing
quantum hardware results against LLM semantic Bell test results.

Backends supported:
- NumPy/SciPy (statevector simulation, always available)
- Qiskit (IBM Quantum hardware + Aer simulators)
- Cirq (Google quantum hardware + simulators)
"""

from .states import (
    bell_state,
    computational_basis,
    semantic_state,
    werner_state,
    parameterized_entangled_state,
)
from .operators import (
    pauli_x,
    pauli_y,
    pauli_z,
    pauli_i,
    rotation_y,
    rotation_z,
    hadamard,
    cnot,
    alice_operators,
    bob_operators,
    chsh_operator,
    measurement_operator,
    commutator,
    commutator_norm,
    random_hermitian,
)
from .measure import (
    expectation_value,
    expectation_value_density,
    chsh_expectation_values,
    chsh_expectation_values_density,
    chsh_s_value,
    measure_state,
    density_matrix,
    reduced_density_matrix,
    von_neumann_entropy,
    entanglement_entropy,
    concurrence,
    fidelity,
)
from .circuits import (
    bell_circuit,
    chsh_circuit,
    semantic_circuit,
    to_qiskit,
    to_cirq,
)
from .compare import (
    compare_quantum_llm,
    sweep_werner_comparison,
    batch_compare,
)
from .hardware import (
    HardwareResult,
    CostEstimate,
    ExperimentLog,
    PRICING,
    estimate_cost,
    estimate_chsh_experiment_cost,
    print_cost_comparison,
    expectation_from_counts,
    s_value_from_counts,
    run_numpy,
    run_ibm,
    run_braket,
    compare_backends,
)
