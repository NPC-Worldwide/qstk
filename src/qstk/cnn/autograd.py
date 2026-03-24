"""
Hybrid autograd for complex-valued PAM language models.

The idea: keep the beautiful numpy complex128 representation for storage,
inspection, and inference -- but delegate gradient computation to PyTorch's
autograd engine, which handles Wirtinger calculus exactly through every
operation (normalization, modReLU, PAM recurrence, complex matmul).

The manual analytical gradients in model.py approximate several tricky
Jacobians (d(z/|z|)/dz through normalization, phase-path through gating).
PyTorch computes them exactly, giving cleaner training signal at scale.

Classes:
    ComplexParameter  -- wraps numpy complex128 <-> torch.complex128
    CharPAM           -- same API as model.CharPAM, torch backward, numpy everything else
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .layers import ComplexEmbed, ComplexNorm, complex_randn, complex_glorot


# ---------------------------------------------------------------------------
# Numpy <-> Torch bridge
# ---------------------------------------------------------------------------

class ComplexParameter:
    """Bridge between numpy complex128 and torch complex128 with autograd.

    Stores the canonical copy as a numpy array. On demand, converts to a
    torch tensor with requires_grad=True for automatic differentiation,
    then converts gradients back to numpy.

    Args:
        data: numpy complex128 array of any shape.

    Example:
        >>> p = ComplexParameter(np.ones((4, 4), dtype=np.complex128))
        >>> t = p.to_torch()     # torch.complex128, requires_grad=True
        >>> # ... do torch ops, call .backward() ...
        >>> grad_np = t.grad.numpy()  # back to numpy
    """

    def __init__(self, data: np.ndarray):
        self.data = data.astype(np.complex128)

    def to_torch(self, requires_grad: bool = True) -> torch.Tensor:
        """Convert to torch tensor, optionally with gradient tracking."""
        t = torch.from_numpy(self.data.copy())
        if requires_grad:
            t.requires_grad_(True)
        return t

    def numpy(self) -> np.ndarray:
        """Return the numpy array (canonical copy)."""
        return self.data

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size


# ---------------------------------------------------------------------------
# Torch-native forward pass operations
# ---------------------------------------------------------------------------

def _torch_complex_norm(z: torch.Tensor, scale: torch.Tensor,
                        eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization on magnitude, phase preserved. All torch ops.

    z:     [..., dim] complex128
    scale: [dim] complex128 (abs used)
    """
    mag = z.abs()  # [..., dim]
    rms = (mag.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    phase = z / (mag + 1e-8)
    return phase * (mag / rms) * scale.abs()


def _torch_mod_relu(z: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Phase-preserving activation: threshold magnitude, keep phase.

    z:    [..., dim] complex128
    bias: [dim] complex128 (real part used as threshold)
    """
    mag = z.abs()
    activated = F.relu(mag + bias.real)
    phase = z / (mag + 1e-8)
    return phase * activated


def _torch_softmax(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax along last axis (real-valued input)."""
    return F.softmax(x, dim=-1)


# ---------------------------------------------------------------------------
# CharPAM (autograd)
# ---------------------------------------------------------------------------

class CharPAM:
    """Character-level PAM language model with hybrid numpy/torch execution.

    Architecture matches CharPAM exactly:
        tokens -> ComplexEmbed -> [PAMLayer x N] -> output_norm -> logits

    Storage and inference: pure numpy complex128.
    Training (forward_backward): torch autograd for exact gradients.

    Args:
        vocab_size: Number of tokens (characters).
        dim:        Complex embedding dimension.
        heads:      Number of PAM heads.
        d_head:     Dimension per head.
        n_layers:   Number of PAM layers.
        decay_init: Initial decay bias (default -2.0).
        residual_scale: Scale for residual connections.

    Example:
        >>> model = CharPAM(65, dim=32, heads=4, d_head=8, n_layers=2)
        >>> ids = np.array([[10, 20, 30, 40, 50]])
        >>> logits = model.forward(ids)          # pure numpy
        >>> loss, grads = model.forward_backward(ids)  # torch autograd
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        heads: int,
        d_head: int,
        n_layers: int,
        decay_init: float = -2.0,
        residual_scale: float = 0.5,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.heads = heads
        self.d_head = d_head
        self.n_layers = n_layers
        self.residual_scale = residual_scale

        # All parameters stored as numpy complex128
        self._params: Dict[str, np.ndarray] = {}

        # Embedding
        self._params['embed'] = complex_randn(
            vocab_size, dim, scale=0.1, dtype=np.complex128
        )

        # Per-layer parameters
        inner_dim = heads * d_head
        for i in range(n_layers):
            pfx = f'L{i}'
            self._params[f'{pfx}.norm_scale'] = np.ones(dim, dtype=np.complex128)
            self._params[f'{pfx}.Wq'] = complex_glorot(dim, inner_dim, dtype=np.complex128)
            self._params[f'{pfx}.Wk'] = complex_glorot(dim, inner_dim, dtype=np.complex128)
            self._params[f'{pfx}.Wv'] = complex_glorot(dim, inner_dim, dtype=np.complex128)
            self._params[f'{pfx}.Wo'] = complex_glorot(inner_dim, dim, dtype=np.complex128) * 0.5
            self._params[f'{pfx}.decay_bias'] = np.full(heads, decay_init, dtype=np.complex128)
            self._params[f'{pfx}.modrelu_bias'] = np.full(dim, -0.05, dtype=np.complex128)

        # Output norm
        self._params['out_norm_scale'] = np.ones(dim, dtype=np.complex128)

        # Count parameters
        self._n_params = sum(p.size for p in self._params.values())

    @property
    def params(self) -> Dict[str, np.ndarray]:
        """All learnable parameters as a flat dict of numpy arrays."""
        return self._params

    @params.setter
    def params(self, p: Dict[str, np.ndarray]) -> None:
        self._params = p

    # ------------------------------------------------------------------
    # Pure numpy forward (for inference / generation)
    # ------------------------------------------------------------------

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Forward pass in pure numpy. tokens -> real logits.

        Args:
            token_ids: Integer array [B, T].

        Returns:
            Real array [B, T, vocab_size].
        """
        p = self._params
        B, T = token_ids.shape

        z = p['embed'][token_ids]  # [B, T, dim]

        for i in range(self.n_layers):
            pfx = f'L{i}'

            # Pre-norm
            mag = np.abs(z)
            rms = np.sqrt(np.mean(mag ** 2, axis=-1, keepdims=True) + 1e-6)
            ns = np.abs(p[f'{pfx}.norm_scale'])
            h = (z / (mag + 1e-8)) * (mag / rms) * ns

            # QKV projections
            Wq, Wk, Wv, Wo = (
                p[f'{pfx}.Wq'], p[f'{pfx}.Wk'],
                p[f'{pfx}.Wv'], p[f'{pfx}.Wo'],
            )
            q = (h @ Wq.T).reshape(B, T, self.heads, self.d_head)
            k = (h @ Wk.T).reshape(B, T, self.heads, self.d_head)
            v = (h @ Wv.T).reshape(B, T, self.heads, self.d_head)

            # QK normalization to unit magnitude
            q = q / (np.abs(q) + 1e-8) * (self.d_head ** -0.5)
            k = k / (np.abs(k) + 1e-8)

            # Decay
            gamma = 1.0 / (1.0 + np.exp(-p[f'{pfx}.decay_bias'].real))

            # Sequential PAM
            S = np.zeros((B, self.heads, self.d_head, self.d_head),
                         dtype=np.complex128)
            y_heads = np.zeros((B, T, self.heads, self.d_head),
                               dtype=np.complex128)
            for t in range(T):
                outer = v[:, t, :, :, None] * k[:, t, :, None, :].conj()
                S = S * gamma[None, :, None, None] + outer
                y_heads[:, t] = np.einsum('bhij,bhj->bhi', S, q[:, t])

            y = y_heads.reshape(B, T, self.dim) @ Wo.T

            # modReLU
            mag_y = np.abs(y)
            activated = np.maximum(0.0, mag_y + p[f'{pfx}.modrelu_bias'].real)
            phase_y = y / (mag_y + 1e-8)
            y_act = phase_y * activated

            z = z + self.residual_scale * y_act

        # Output norm
        mag_z = np.abs(z)
        rms_z = np.sqrt(np.mean(mag_z ** 2, axis=-1, keepdims=True) + 1e-6)
        out_s = np.abs(p['out_norm_scale'])
        z_normed = (z / (mag_z + 1e-8)) * (mag_z / rms_z) * out_s

        # Tied embedding readout
        logits = (z_normed @ p['embed'].conj().T).real
        return logits

    # ------------------------------------------------------------------
    # Torch-backed forward_backward (exact autograd)
    # ------------------------------------------------------------------

    def forward_backward(
        self,
        token_ids: np.ndarray,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Forward pass + cross-entropy loss + exact backward via torch autograd.

        Converts numpy params to torch tensors, runs the forward pass using
        native torch complex128 ops (which mirror the numpy ones exactly),
        calls .backward(), and extracts gradients back to numpy.

        Args:
            token_ids: Integer array [B, T].

        Returns:
            (loss, grads) where grads maps parameter names to numpy arrays.
        """
        B, T = token_ids.shape
        p = self._params

        # ---- Convert all params to torch tensors with grad ----
        torch_params: Dict[str, torch.Tensor] = {}
        for name, arr in p.items():
            torch_params[name] = torch.from_numpy(arr.copy()).requires_grad_(True)

        # Token ids as torch (long, no grad)
        tids = torch.from_numpy(token_ids).long()

        # ---- Forward pass in torch ----
        # Embedding lookup
        embed = torch_params['embed']
        z = embed[tids]  # [B, T, dim]

        for i in range(self.n_layers):
            pfx = f'L{i}'

            # Pre-norm
            norm_scale = torch_params[f'{pfx}.norm_scale']
            h = _torch_complex_norm(z, norm_scale)

            # QKV
            Wq = torch_params[f'{pfx}.Wq']
            Wk = torch_params[f'{pfx}.Wk']
            Wv = torch_params[f'{pfx}.Wv']
            Wo = torch_params[f'{pfx}.Wo']

            q = (h @ Wq.T).reshape(B, T, self.heads, self.d_head)
            k = (h @ Wk.T).reshape(B, T, self.heads, self.d_head)
            v = (h @ Wv.T).reshape(B, T, self.heads, self.d_head)

            # QK normalization to unit magnitude (pure phase)
            q = q / (q.abs() + 1e-8) * (self.d_head ** -0.5)
            k = k / (k.abs() + 1e-8)

            # Decay: gamma = sigmoid(decay_bias.real)
            decay_bias = torch_params[f'{pfx}.decay_bias']
            gamma = torch.sigmoid(decay_bias.real)  # [heads]

            # Sequential PAM recurrence
            # We unroll the loop so autograd can track every step.
            S = torch.zeros(B, self.heads, self.d_head, self.d_head,
                            dtype=torch.complex128)
            y_list = []
            for t in range(T):
                v_t = v[:, t]   # [B, heads, d_head]
                k_t = k[:, t]   # [B, heads, d_head]
                q_t = q[:, t]   # [B, heads, d_head]

                # outer(v_t, conj(k_t)): [B, H, d, d]
                outer = v_t.unsqueeze(-1) * k_t.conj().unsqueeze(-2)
                S = S * gamma[None, :, None, None] + outer

                # Retrieval: y_t = S @ q_t
                y_t = torch.einsum('bhij,bhj->bhi', S, q_t)
                y_list.append(y_t)

            y_heads = torch.stack(y_list, dim=1)  # [B, T, heads, d_head]

            # Merge heads + output projection
            y = y_heads.reshape(B, T, self.heads * self.d_head) @ Wo.T

            # modReLU
            modrelu_bias = torch_params[f'{pfx}.modrelu_bias']
            y_act = _torch_mod_relu(y, modrelu_bias)

            # Residual
            z = z + self.residual_scale * y_act

        # Output norm
        out_scale = torch_params['out_norm_scale']
        z_normed = _torch_complex_norm(z, out_scale)

        # Tied embedding readout: logits = Re(z_normed @ embed^H)
        logits = (z_normed @ embed.conj().T).real  # [B, T, V]

        # ---- Cross-entropy loss (next-token prediction) ----
        logits_shift = logits[:, :-1].contiguous()       # [B, T-1, V]
        targets = tids[:, 1:].contiguous()                # [B, T-1]

        # Reshape for F.cross_entropy: (N, C) and (N,)
        loss = F.cross_entropy(
            logits_shift.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )

        # ---- Backward ----
        loss.backward()

        # ---- Extract gradients back to numpy ----
        grads: Dict[str, np.ndarray] = {}
        for name, tensor in torch_params.items():
            if tensor.grad is not None:
                grads[name] = tensor.grad.detach().numpy().copy()
            else:
                grads[name] = np.zeros_like(p[name])

        return float(loss.item()), grads

    # ------------------------------------------------------------------
    # Generation (pure numpy, same as CharPAM)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> np.ndarray:
        """Autoregressive generation in pure numpy.

        Args:
            prompt_ids: 1D integer array.
            max_tokens: Tokens to generate.
            temperature: Sampling temperature.

        Returns:
            1D integer array: prompt + generated tokens.
        """
        gen_ids = prompt_ids.copy()

        for _ in range(max_tokens):
            ctx = gen_ids[None, :]
            logits = self.forward(ctx)
            next_logits = logits[0, -1]

            x = next_logits / max(temperature, 0.01)
            x = x - x.max()
            e = np.exp(x)
            probs = e / (e.sum() + 1e-10)

            next_id = np.random.choice(self.vocab_size, p=probs)
            gen_ids = np.append(gen_ids, next_id)

        return gen_ids

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> Dict[str, object]:
        """Phase coherence, gammas, parameter norms."""
        p = self._params
        embed = p['embed']

        # Embedding phase coherence
        phases = np.angle(embed)
        mean_phasor = np.mean(np.exp(1j * phases), axis=0)
        pc = float(np.mean(np.abs(mean_phasor)))

        mags = np.abs(embed)

        gammas = []
        for i in range(self.n_layers):
            db = p[f'L{i}.decay_bias']
            g = 1.0 / (1.0 + np.exp(-db.real))
            gammas.append(g)

        norms = {}
        for k, v in p.items():
            norms[k] = float(np.sqrt(np.sum(np.abs(v) ** 2)))

        return {
            'embed_phase_coherence': pc,
            'embed_mag_mean': float(mags.mean()),
            'embed_mag_std': float(mags.std()),
            'layer_gammas': gammas,
            'param_norms': norms,
        }
