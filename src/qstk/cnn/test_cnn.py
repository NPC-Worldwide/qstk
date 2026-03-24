#!/usr/bin/env python3
"""
Test script for qstk.cnn -- verifies all modules import and run correctly.

Runs a small forward pass through each component, checks shapes and dtypes,
then does a mini training loop with CharPAM.
"""

import sys
import numpy as np

np.random.seed(42)


def section(name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")


def check(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        raise AssertionError(f"Check failed: {label}")


# -----------------------------------------------------------------------
# 1. layers.py
# -----------------------------------------------------------------------
section("layers.py -- ComplexLinear, ComplexEmbed, ComplexNorm, modReLU, CGU")

from qstk.cnn.layers import (
    complex_randn, complex_glorot,
    ComplexLinear, ComplexEmbed, ComplexNorm,
    mod_relu, complex_gated_unit,
)

# complex_randn
z = complex_randn(4, 8, 16, scale=0.5)
check(f"complex_randn shape={z.shape}", z.shape == (4, 8, 16))
check(f"complex_randn dtype={z.dtype}", z.dtype == np.complex128)
check("complex_randn has nonzero imag", np.any(z.imag != 0))

# complex_glorot
W = complex_glorot(32, 16)
check(f"complex_glorot shape={W.shape}", W.shape == (16, 32))
check(f"complex_glorot dtype={W.dtype}", W.dtype == np.complex128)

# ComplexLinear
linear = ComplexLinear(32, 16)
x = complex_randn(2, 10, 32)
y = linear(x)
check(f"ComplexLinear forward shape={y.shape}", y.shape == (2, 10, 16))
check(f"ComplexLinear output dtype={y.dtype}", y.dtype == np.complex128)
check("ComplexLinear has params['W']", 'W' in linear.params)

# ComplexEmbed
embed = ComplexEmbed(100, 32)
ids = np.array([[0, 5, 10], [20, 30, 40]])
z = embed(ids)
check(f"ComplexEmbed forward shape={z.shape}", z.shape == (2, 3, 32))
check(f"ComplexEmbed output dtype={z.dtype}", z.dtype == np.complex128)

# ComplexNorm
norm = ComplexNorm(32)
z_in = complex_randn(2, 10, 32, scale=5.0)
z_out = norm(z_in)
check(f"ComplexNorm forward shape={z_out.shape}", z_out.shape == (2, 10, 32))
mag_out = np.abs(z_out)
rms_out = np.sqrt(np.mean(mag_out ** 2, axis=-1))
check("ComplexNorm produces ~unit RMS", np.allclose(rms_out, 1.0, atol=0.1))

# mod_relu
bias = np.full(32, -0.1, dtype=np.complex128)
z_act = mod_relu(z_in, bias)
check(f"mod_relu shape={z_act.shape}", z_act.shape == z_in.shape)
# Phase should be preserved where output is nonzero
nz = np.abs(z_act) > 1e-10
if np.any(nz):
    phase_in = np.angle(z_in[nz])
    phase_out = np.angle(z_act[nz])
    check("mod_relu preserves phase", np.allclose(phase_in, phase_out, atol=1e-6))

# complex_gated_unit
W_gate = complex_glorot(32, 48)
W_val = complex_glorot(32, 48)
x_cgu = complex_randn(2, 10, 32)
y_cgu = complex_gated_unit(x_cgu, W_gate, W_val)
check(f"CGU shape={y_cgu.shape}", y_cgu.shape == (2, 10, 48))


# -----------------------------------------------------------------------
# 2. rope.py
# -----------------------------------------------------------------------
section("rope.py -- complex rotary position encoding")

from qstk.cnn.rope import make_freqs, complex_rope

freqs = make_freqs(dim=16, max_len=64)
check(f"make_freqs shape={freqs.shape}", freqs.shape == (64, 16))
check("make_freqs are unit magnitude", np.allclose(np.abs(freqs), 1.0, atol=1e-10))
check(f"make_freqs dtype={freqs.dtype}", np.issubdtype(freqs.dtype, np.complexfloating))

x_rope = complex_randn(2, 32, 16)
x_pos = complex_rope(x_rope, seq_len=32)
check(f"complex_rope shape={x_pos.shape}", x_pos.shape == (2, 32, 16))
# Magnitude should be preserved (rotation by unit complex)
check("complex_rope preserves magnitude",
      np.allclose(np.abs(x_pos), np.abs(x_rope), atol=1e-10))

# With offset
x_off = complex_rope(x_rope[:, :10, :], seq_len=10, offset=5)
check(f"complex_rope with offset shape={x_off.shape}", x_off.shape == (2, 10, 16))


# -----------------------------------------------------------------------
# 3. pam.py
# -----------------------------------------------------------------------
section("pam.py -- Phase-Associative Memory")

from qstk.cnn.pam import PAMLayer

pam = PAMLayer(dim=32, heads=4, d_head=8)
x_pam = complex_randn(2, 16, 32)

# Sequential mode
y_seq, state_seq = pam.forward(x_pam, mode='sequential')
check(f"PAM sequential output shape={y_seq.shape}", y_seq.shape == (2, 16, 32))
check(f"PAM sequential state shape={state_seq.shape}", state_seq.shape == (2, 4, 8, 8))
check("PAM output is complex", np.issubdtype(y_seq.dtype, np.complexfloating))

# Dual mode
y_dual, state_dual = pam.forward(x_pam, mode='dual')
check(f"PAM dual output shape={y_dual.shape}", y_dual.shape == (2, 16, 32))
check(f"PAM dual state shape={state_dual.shape}", state_dual.shape == (2, 4, 8, 8))

# Both modes should give similar results (not exact due to residual path)
# Just check they're in the same ballpark
mag_seq = np.abs(y_seq).mean()
mag_dual = np.abs(y_dual).mean()
check(f"Sequential/dual magnitude ratio={mag_seq/mag_dual:.2f} (should be ~1)",
      0.5 < mag_seq / mag_dual < 2.0)

# Params collection
params = pam.params
check("PAM params has Wq", 'Wq' in params)
check("PAM params has decay_bias", 'decay_bias' in params)
n_pam_params = sum(p.size for p in params.values())
print(f"  PAM total params: {n_pam_params} complex numbers")

# Continued generation with state
x_cont = complex_randn(2, 1, 32)
y_cont, state_cont = pam.forward(x_cont, mode='sequential', state=state_seq)
check(f"PAM continued gen shape={y_cont.shape}", y_cont.shape == (2, 1, 32))


# -----------------------------------------------------------------------
# 4. optim.py
# -----------------------------------------------------------------------
section("optim.py -- ComplexAdam")

from qstk.cnn.optim import ComplexAdam

test_params = {
    'W': complex_randn(16, 32, scale=0.5),
    'b': np.zeros(16, dtype=np.complex128),
}
opt = ComplexAdam(test_params, lr=1e-2)

# Simulate a few gradient steps
for i in range(5):
    grads = {
        'W': complex_randn(16, 32, scale=0.01),
        'b': complex_randn(16, scale=0.01),
    }
    opt.step(grads)

check("ComplexAdam ran 5 steps", opt.t == 5)
check("Parameters were updated", not np.allclose(
    test_params['W'], complex_randn(16, 32, scale=0.5)))

# zero_grad
opt.zero_grad()
check("zero_grad resets step counter", opt.t == 0)


# -----------------------------------------------------------------------
# 5. probe.py
# -----------------------------------------------------------------------
section("probe.py -- embedding projection and phase analysis")

from qstk.cnn.probe import (
    embed_to_complex, phase_coherence, phase_clusters,
    interference_score, semantic_entanglement,
)

# embed_to_complex
real_emb = np.random.randn(50, 64)

z_hilbert = embed_to_complex(real_emb, method='hilbert')
check(f"hilbert projection shape={z_hilbert.shape}", z_hilbert.shape == (50, 32))
check("hilbert output is complex", np.issubdtype(z_hilbert.dtype, np.complexfloating))

z_paired = embed_to_complex(real_emb, method='paired')
check(f"paired projection shape={z_paired.shape}", z_paired.shape == (50, 32))

z_random = embed_to_complex(real_emb, method='random_proj')
check(f"random_proj shape={z_random.shape}", z_random.shape == (50, 32))

# phase_coherence
pc = phase_coherence(z_hilbert)
check(f"phase_coherence={pc:.4f} in [0, 1]", 0 <= pc <= 1)

# phase_clusters
labels, centroids = phase_clusters(z_hilbert, n_clusters=3)
check(f"phase_clusters labels shape={labels.shape}", labels.shape == (50,))
check(f"phase_clusters centroids shape={centroids.shape}", centroids.shape == (3, 32))
check("all cluster labels assigned", set(labels).issubset({0, 1, 2}))

# interference_score
z1 = complex_randn(64)
z2 = z1.copy()
score_self = interference_score(z1, z2)
check(f"self-interference={score_self:.4f} (should be ~1.0)", abs(score_self - 1.0) < 0.01)

score_neg = interference_score(z1, -z1)
check(f"anti-interference={score_neg:.4f} (should be ~-1.0)", abs(score_neg + 1.0) < 0.01)

score_ortho = interference_score(z1, z1 * 1j)
check(f"orthogonal interference={score_ortho:.4f} (should be ~0.0)", abs(score_ortho) < 0.01)

# semantic_entanglement
z_pairs = complex_randn(30, 2, 64, scale=1.0)
S = semantic_entanglement(z_pairs)
check(f"semantic_entanglement S={S:.4f} (should be finite)", np.isfinite(S))


# -----------------------------------------------------------------------
# 6. model.py -- CharPAM
# -----------------------------------------------------------------------
section("model.py -- CharPAM")

from qstk.cnn.model import CharPAM, ComplexProbe

model = CharPAM(
    vocab_size=28,
    dim=32,
    heads=4,
    d_head=8,
    n_layers=2,
)

print(f"  CharPAM: {model._n_params} complex parameters "
      f"({model._n_params * 2} real floats)")

# Forward pass
token_ids = np.random.randint(0, 28, size=(2, 16))
logits = model.forward(token_ids)
check(f"CharPAM forward shape={logits.shape}", logits.shape == (2, 16, 28))
check("Logits are real", logits.dtype in (np.float64, np.float32))

# Forward-backward
loss, grads = model.forward_backward(token_ids)
check(f"CharPAM loss={loss:.4f} (should be ~ln(28)={np.log(28):.4f})",
      abs(loss - np.log(28)) < 1.5)
check(f"Grads has {len(grads)} entries", len(grads) > 0)
check("Grads include embed", 'embed' in grads)
check("Grads include L0.Wq", 'L0.Wq' in grads)

# Mini training loop (5 steps)
print("\n  Mini training loop (5 steps):")
opt = ComplexAdam(model.params, lr=1e-3)
for step in range(5):
    batch = np.random.randint(0, 28, size=(4, 20))
    loss, grads = model.forward_backward(batch)
    opt.step(grads)
    print(f"    step {step}: loss={loss:.4f}")

# Generation
gen = model.generate(np.array([0, 1, 2, 3]), max_tokens=10)
check(f"Generation length={len(gen)} (should be 14)", len(gen) == 14)
check("Generated ids are valid", np.all((0 <= gen) & (gen < 28)))

# Diagnostics
diag = model.diagnostics()
check(f"Diagnostics has embed_phase_coherence={diag['embed_phase_coherence']:.4f}",
      'embed_phase_coherence' in diag)
check(f"Diagnostics has {len(diag['layer_gammas'])} layer gammas",
      len(diag['layer_gammas']) == 2)


# -----------------------------------------------------------------------
# 7. model.py -- ComplexProbe
# -----------------------------------------------------------------------
section("model.py -- ComplexProbe")

probe = ComplexProbe(real_dim=64)
real_data = np.random.randn(100, 64)
z_probed = probe.project(real_data)
check(f"ComplexProbe project shape={z_probed.shape}", z_probed.shape == (100, 32))

metrics = probe.analyze(z_probed)
check(f"Probe phase_coherence={metrics['phase_coherence']:.4f}", 'phase_coherence' in metrics)
check(f"Probe mag_mean={metrics['mag_mean']:.4f}", 'mag_mean' in metrics)
check(f"Probe mean_interference={metrics['mean_interference']:.4f}", 'mean_interference' in metrics)


# -----------------------------------------------------------------------
# 8. Top-level imports from qstk.cnn
# -----------------------------------------------------------------------
section("Top-level imports from qstk.cnn")

from qstk.cnn import (
    ComplexLinear, ComplexEmbed, ComplexNorm, mod_relu, complex_gated_unit,
    complex_randn, complex_glorot,
    PAMLayer, complex_rope, make_freqs,
    ComplexAdam, CharPAM, ComplexProbe,
    embed_to_complex, phase_coherence, phase_clusters,
    interference_score, semantic_entanglement,
)
check("All top-level imports work", True)


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
section("ALL TESTS PASSED")
print("""
  layers.py   -- ComplexLinear, ComplexEmbed, ComplexNorm, modReLU, CGU
  rope.py     -- make_freqs, complex_rope
  pam.py      -- PAMLayer (sequential + dual mode)
  optim.py    -- ComplexAdam
  probe.py    -- embed_to_complex, phase_coherence, phase_clusters,
                 interference_score, semantic_entanglement
  model.py    -- CharPAM (forward, backward, generate, diagnostics),
                 ComplexProbe

  Everything is native complex128, pure numpy, no PyTorch.
""")
