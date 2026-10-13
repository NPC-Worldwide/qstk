# Quantum Semantic Toolkit and Library (QSTKL)

**Author:** Christopher Agostino  
**Company:** NPC Worldwide  
**Contact:** info@npcworldwi.de

## Overview

QSTKL is a small Python toolkit for quantum NLP experiments. It provides building blocks for tasks that connect language models to small-qubit quantum computing:

- encode semantic ambiguity or topic correlations as quantum states
- run those states through simulators (NumPy, Qiskit Aer) or real QPUs (IBM Quantum, Amazon Braket)
- drive LLM experiments with persona variation and parameter sweeps
- analyze results with statistical helpers

The package is intentionally lightweight. The core simulator is pure NumPy. Optional integrations are loaded only when installed.

## Installation

```bash
pip install qstkl
```

Or clone and install locally:

```bash
git clone https://github.com/npc-worldwide/qstk.git
cd qstk
pip install -e .
```

## Requirements

- Python 3.10+
- numpy
- npcpy>=1.3.33

Optional:
- qiskit >= 1.0, qiskit-aer, qiskit-ibm-runtime
- amazon-braket-sdk
- pandas
- matplotlib

## Modules

### `qstk.qc` — Quantum Computing

Bell-state preparation, measurement, operator construction, and circuit export to Qiskit/Cirq. Includes a NumPy statevector simulator and optional runners for IBM Quantum and Amazon Braket.

```python
from qstk.qc import bell_state, chsh_circuit, semantic_circuit, to_qiskit

psi = bell_state("phi_plus")
result = chsh_circuit("phi_plus", n_shots=4096)
print(result["s_value"], result["violation"])
```

### `qstk.cnn` — Complex-Valued Semantic Models

Pure-NumPy complex neural network layers, state-space models, and attention-style probes. Optional PyTorch autograd training is available.

```python
from qstk.cnn import layers, model, pam, ssm
```

### `qstk.chsh` — Bell-Test Statistics

Compute expectation values and S-values from LLM or quantum trial outcomes.

```python
from qstk.chsh import calculate_s_value, check_violation
```

### `qstk.personas` — Persona Generation

Random personas and system prompts for LLM experiments.

```python
from qstk.personas import generate_persona, get_persona_prompt

persona = generate_persona()
system = get_persona_prompt(persona)
```

### `qstk.arrays` — Vectorized LLM Inference

NPCArray helpers for running prompts across models and parameters in parallel.

```python
from qstk.arrays import create_bell_meshgrid

grid = create_bell_meshgrid(
    models=["gemma3:12b"], providers=["ollama"],
    temperatures=[0.2, 1.0, 1.8],
)
```

### `qstk.grid`, `qstk.results`, `qstk.statistics`

Parameter sweep construction, CSV logging/aggregation, and combinatorial significance tests.

## Examples

```bash
pip install matplotlib qiskit qiskit-aer
python examples/qc_basics.py          # quantum circuit walkthrough
python examples/chsh_experiments.py     # classical vs quantum statistics
python examples/hardware_guide.py       # backend cost estimation
python examples/qc_interactive.py       # interactive sliders
python docs/generate_images.py          # regenerate figures
```

## A Word-Sense Encoding Experiment

Map ambiguous word meanings to a 2-qubit state and measure correlations:

```python
from qstk.qc import semantic_circuit

# bank (financial / river) x bat (baseball / animal)
correlated = [0.7, 0.05, 0.05, 0.7]
result = semantic_circuit(correlated, n_shots=4096)
print(f"S = {result['s_value']:.3f}, violation = {result['violation']}")
```

## License

MIT
