"""qstk.cnn — Complex-valued Neural Networks

Native complex128/complex64 building blocks for quantum semantic models.
Core is pure numpy. Optional torch backend for autograd.

Two modes of use:
  1. BUILD complex-valued models from scratch (PAM, phase attention, etc.)
  2. PROBE existing LLMs by projecting their embeddings into complex space

Modules:
  - layers: ComplexLinear, ComplexEmbed, ComplexNorm, modReLU, CGU
  - pam: Phase-Associative Memory (matrix state, conjugate retrieval)
  - rope: Complex rotary position encoding (natural phase rotation)
  - optim: Adam for complex parameters (Wirtinger gradients)
  - probe: Project real-valued LLM embeddings into complex space,
           analyze phase structure, measure entanglement
  - model: Ready-to-use complex language models (CharPAM w/ analytical grads)
  - autograd: CharPAM with torch autograd for exact gradients (requires torch)

Example:
    >>> from qstk.cnn import ComplexLinear, PAMLayer, ComplexEmbed
    >>> embed = ComplexEmbed(vocab_size=256, dim=64)
    >>> pam = PAMLayer(dim=64, heads=4, d_head=16)
    >>> x = embed(token_ids)          # native complex128
    >>> y = pam(x)                    # phase-associative memory
    >>> logits = np.abs(y @ embed.W.conj().T)  # magnitude as logits

    >>> from qstk.cnn.probe import embed_to_complex, phase_coherence
    >>> z = embed_to_complex(real_embeddings)   # project to complex space
    >>> s = phase_coherence(z)                  # measure phase structure
"""

from .layers import (
    ComplexLinear,
    ComplexEmbed,
    ComplexNorm,
    mod_relu,
    complex_gated_unit,
    complex_randn,
    complex_glorot,
)
from .pam import PAMLayer
from .rope import complex_rope, make_freqs
from .optim import ComplexAdam
from .model import CharPAM, ComplexProbe

# Torch autograd backend (optional — only if torch installed)
try:
    from .autograd import CharPAM as CharPAMAutograd
except ImportError:
    pass

from .probe import (
    embed_to_complex,
    phase_coherence,
    phase_clusters,
    interference_score,
    semantic_entanglement,
)

from .operators import (
    transition_operator,
    extract_operators,
    operator_diversity,
    operator_spectrum,
    trajectory_coherence,
    compare_temperature_regimes,
    creativity_correlation,
)
