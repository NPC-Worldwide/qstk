"""
Ready-to-use complex-valued language models -- pure numpy.

Classes:
    CharPAM      -- character-level PAM language model
    ComplexProbe  -- project real-valued model embeddings to complex space
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .layers import ComplexEmbed, ComplexNorm, complex_randn
from .pam import PAMLayer
from .probe import embed_to_complex, phase_coherence


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-10)


class CharPAM:
    """Character-level Phase-Associative Memory language model.

    Architecture:
        tokens -> ComplexEmbed -> [PAMLayer x N] -> output_norm -> logits

    Logits are computed via tied embedding readout:
        logits = Re(z_normed @ embed^H)
    The real part of the complex inner product gives phase-interference
    based scoring: tokens whose phase aligns with the prediction get
    high logits (constructive), misaligned ones get low (destructive).

    Args:
        vocab_size: Number of tokens (characters).
        dim:        Complex embedding dimension.
        heads:      Number of PAM heads.
        d_head:     Dimension per head.
        n_layers:   Number of PAM layers.
        decay_init: Initial decay bias (default -2.0).
        dtype:      np.complex128 or np.complex64.

    Example:
        >>> model = CharPAM(vocab_size=128, dim=32, heads=4, d_head=8, n_layers=2)
        >>> ids = np.array([[72, 101, 108, 108, 111]])  # "Hello"
        >>> logits = model.forward(ids)
        >>> logits.shape  # (1, 5, 128)
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
        dtype: np.dtype = np.complex128,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.heads = heads
        self.d_head = d_head
        self.n_layers = n_layers
        self.residual_scale = residual_scale
        self.dtype = dtype

        # Embedding (tied weights for input and output)
        self.embed = ComplexEmbed(vocab_size, dim, dtype=dtype)

        # PAM layers
        self.layers: List[PAMLayer] = [
            PAMLayer(dim, heads, d_head, decay_init=decay_init, dtype=dtype)
            for _ in range(n_layers)
        ]

        # Output norm
        self.output_norm = ComplexNorm(dim, dtype=dtype)

        # Count parameters
        total = self.embed.params['W'].size + self.output_norm.params['scale'].size
        for layer in self.layers:
            for v in layer.params.values():
                total += v.size
        self._n_params = total

    @property
    def params(self) -> Dict[str, np.ndarray]:
        """Collect all learnable parameters into a flat dict."""
        p: Dict[str, np.ndarray] = {}
        p['embed'] = self.embed.params['W']
        for i, layer in enumerate(self.layers):
            for k, v in layer.params.items():
                p[f'L{i}.{k}'] = v
        p['out_norm_scale'] = self.output_norm.params['scale']
        return p

    @params.setter
    def params(self, p: Dict[str, np.ndarray]) -> None:
        """Set parameters from a flat dict."""
        self.embed.params['W'] = p['embed']
        for i, layer in enumerate(self.layers):
            lp = {}
            for k in layer.params:
                lp[k] = p[f'L{i}.{k}']
            layer.params = lp
        self.output_norm.params['scale'] = p['out_norm_scale']

    def forward(
        self,
        token_ids: np.ndarray,
        mode: str = 'sequential',
    ) -> np.ndarray:
        """Forward pass: tokens -> logits.

        Args:
            token_ids: Integer array, shape [B, T].
            mode:      'sequential' or 'dual' (passed to PAM layers).

        Returns:
            Real array of logits, shape [B, T, vocab_size].
        """
        z = self.embed(token_ids)  # [B, T, dim] complex

        for layer in self.layers:
            z, _ = layer.forward(z, mode=mode, residual_scale=self.residual_scale)

        # Output norm
        z_normed = self.output_norm(z)

        # Tied embedding readout: logits = Re(z_normed @ embed^H)
        logits = (z_normed @ self.embed.params['W'].conj().T).real
        return logits

    def forward_backward(
        self,
        token_ids: np.ndarray,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Forward pass + cross-entropy loss + analytical backprop.

        Next-token prediction: predict token_ids[:, t+1] from position t.

        Args:
            token_ids: Integer array, shape [B, T].

        Returns:
            Tuple of:
              - loss: Scalar cross-entropy loss.
              - grads: Dict mapping parameter names to gradient arrays.

        Note:
            This implements the full backward pass with analytical gradients
            through the sequential PAM. No autograd dependency.
        """
        B, T = token_ids.shape
        p = self.params

        # ==================== FORWARD ====================
        z0 = p['embed'][token_ids]  # [B, T, dim]

        layer_data = []
        z = z0

        for li in range(self.n_layers):
            pfx = f'L{li}'
            # Pre-norm
            mag = np.abs(z)
            rms = np.sqrt(np.mean(mag ** 2, axis=-1, keepdims=True) + 1e-6)
            ns = np.abs(p[f'{pfx}.norm_scale'])
            h = (z / (mag + 1e-8)) * (mag / rms) * ns

            # QKV projections
            Wq = p[f'{pfx}.Wq']
            Wk = p[f'{pfx}.Wk']
            Wv = p[f'{pfx}.Wv']
            Wo = p[f'{pfx}.Wo']

            q_raw = (h @ Wq.T).reshape(B, T, self.heads, self.d_head)
            k_raw = (h @ Wk.T).reshape(B, T, self.heads, self.d_head)
            v = (h @ Wv.T).reshape(B, T, self.heads, self.d_head)

            # QK normalization
            q_mag = np.abs(q_raw)
            k_mag = np.abs(k_raw)
            q = q_raw / (q_mag + 1e-8)
            k = k_raw / (k_mag + 1e-8)

            scale = self.d_head ** -0.5
            q = q * scale

            gamma = 1.0 / (1.0 + np.exp(-p[f'{pfx}.decay_bias'].real))

            # Sequential PAM
            S_all = np.zeros((B, T, self.heads, self.d_head, self.d_head), dtype=self.dtype)
            S = np.zeros((B, self.heads, self.d_head, self.d_head), dtype=self.dtype)
            y_heads = np.zeros((B, T, self.heads, self.d_head), dtype=self.dtype)

            for t in range(T):
                outer = v[:, t, :, :, None] * k[:, t, :, None, :].conj()
                S = S * gamma[None, :, None, None] + outer
                S_all[:, t] = S
                y_heads[:, t] = np.einsum('bhij,bhj->bhi', S, q[:, t])

            y_pre = y_heads.reshape(B, T, self.dim)
            y_post = y_pre @ Wo.T

            # modReLU
            mag_y = np.abs(y_post)
            bias_r = p[f'{pfx}.modrelu_bias'].real
            gate = (mag_y + bias_r > 0).astype(np.float64)
            activated = np.maximum(0.0, mag_y + bias_r)
            phase_y = y_post / (mag_y + 1e-8)
            y_act = phase_y * activated

            z_next = z + self.residual_scale * y_act

            layer_data.append({
                'z_in': z, 'h': h, 'q': q, 'k': k, 'v': v,
                'q_raw': q_raw, 'k_raw': k_raw,
                'S_all': S_all, 'gamma': gamma,
                'y_pre': y_pre, 'y_post': y_post,
                'mag_y': mag_y, 'gate': gate, 'phase_y': phase_y,
            })
            z = z_next

        # Output norm
        mag_z = np.abs(z)
        rms_z = np.sqrt(np.mean(mag_z ** 2, axis=-1, keepdims=True) + 1e-6)
        out_s = np.abs(p['out_norm_scale'])
        z_normed = (z / (mag_z + 1e-8)) * (mag_z / rms_z) * out_s

        # Logits
        embed = p['embed']
        logits = (z_normed @ embed.conj().T).real  # [B, T, V]

        # Loss (next-token cross-entropy)
        logits_shift = logits[:, :-1]
        targets = token_ids[:, 1:]
        probs = _softmax(logits_shift)
        log_probs = np.log(probs + 1e-10)

        loss = 0.0
        for b in range(B):
            for t in range(T - 1):
                loss -= log_probs[b, t, targets[b, t]]
        loss /= (B * (T - 1))

        # ==================== BACKWARD ====================
        grads: Dict[str, np.ndarray] = {}

        # d(CE)/d(logits)
        dlogits = probs.copy()
        for b in range(B):
            for t in range(T - 1):
                dlogits[b, t, targets[b, t]] -= 1.0
        dlogits /= (B * (T - 1))

        dlogits_full = np.zeros((B, T, self.vocab_size))
        dlogits_full[:, :-1] = dlogits

        # Backprop through logits = Re(z_normed @ embed^H)
        dz_normed = dlogits_full @ embed  # Wirtinger: dRe/dz*
        grads['embed'] = np.zeros_like(embed)
        for vi in range(self.vocab_size):
            grads['embed'][vi] = np.sum(
                dlogits_full[:, :, vi:vi + 1] * z_normed, axis=(0, 1)
            )

        grads['out_norm_scale'] = np.zeros_like(p['out_norm_scale'])

        # Back through output norm (approximate pass-through)
        dz = dz_normed.copy()

        # Back through layers
        for li in reversed(range(self.n_layers)):
            pfx = f'L{li}'
            ld = layer_data[li]

            # Residual
            dy_act = self.residual_scale * dz

            # modReLU backward
            dy_post = dy_act * ld['gate']
            grads[f'{pfx}.modrelu_bias'] = (
                np.sum((dy_act * ld['phase_y'].conj()).real * ld['gate'], axis=(0, 1))
                + 0j
            )

            # Wo backward
            dy_pre = dy_post @ p[f'{pfx}.Wo'].conj()
            grads[f'{pfx}.Wo'] = np.einsum('btj,bti->ji', dy_post, ld['y_pre'].conj()) / B

            # Reshape to heads
            dy_heads = dy_pre.reshape(B, T, self.heads, self.d_head)

            q, k, v = ld['q'], ld['k'], ld['v']
            S_all, gamma = ld['S_all'], ld['gamma']
            scale = self.d_head ** -0.5

            dq_acc = np.zeros_like(q)
            dk_acc = np.zeros_like(k)
            dv_acc = np.zeros_like(v)
            dgamma_acc = np.zeros(self.heads)

            dS = np.zeros((B, self.heads, self.d_head, self.d_head), dtype=self.dtype)

            for t in reversed(range(T)):
                dy_t = dy_heads[:, t]
                S_t = S_all[:, t]
                q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]

                dS += dy_t[:, :, :, None] * q_t[:, :, None, :].conj()
                dq_acc[:, t] = np.einsum('bhji,bhj->bhi', S_t.conj(), dy_t)
                dv_acc[:, t] = np.einsum('bhij,bhj->bhi', dS, k_t)
                dk_acc[:, t] = np.einsum('bhij,bhi->bhj', dS, v_t).conj()

                if t > 0:
                    dgamma_acc += (dS * S_all[:, t - 1].conj()).real.sum(axis=(0, 2, 3))

                dS = dS * gamma[None, :, None, None]

            dq_acc *= scale

            dq_flat = dq_acc.reshape(B, T, self.dim)
            dk_flat = dk_acc.reshape(B, T, self.dim)
            dv_flat = dv_acc.reshape(B, T, self.dim)

            h = ld['h']
            grads[f'{pfx}.Wq'] = np.einsum('btj,bti->ji', dq_flat, h.conj()) / B
            grads[f'{pfx}.Wk'] = np.einsum('btj,bti->ji', dk_flat, h.conj()) / B
            grads[f'{pfx}.Wv'] = np.einsum('btj,bti->ji', dv_flat, h.conj()) / B

            sig = gamma
            grads[f'{pfx}.decay_bias'] = dgamma_acc * sig * (1 - sig) / B + 0j
            grads[f'{pfx}.norm_scale'] = np.zeros_like(p[f'{pfx}.norm_scale'])

            dh = (
                dq_flat @ p[f'{pfx}.Wq'].conj()
                + dk_flat @ p[f'{pfx}.Wk'].conj()
                + dv_flat @ p[f'{pfx}.Wv'].conj()
            )
            dz = dz + dh

        # Embedding gradient from input
        for b in range(B):
            for t in range(T):
                grads['embed'][token_ids[b, t]] += dz[b, t] / B

        return loss, grads

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> np.ndarray:
        """Autoregressive generation.

        Runs the full context through the model at each step (no KV cache).
        For short sequences this is fine; for longer ones, use sequential
        mode with cached state (future work).

        Args:
            prompt_ids: 1D integer array, the prompt token ids.
            max_tokens: Number of tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).

        Returns:
            1D integer array: prompt_ids concatenated with generated ids.
        """
        gen_ids = prompt_ids.copy()
        p = self.params

        for _ in range(max_tokens):
            # Use last chunk of context
            ctx = gen_ids[None, :]  # [1, T]
            logits = self.forward(ctx)  # [1, T, vocab]
            next_logits = logits[0, -1]  # [vocab]

            probs = _softmax(next_logits / temperature)
            next_id = np.random.choice(self.vocab_size, p=probs)
            gen_ids = np.append(gen_ids, next_id)

        return gen_ids

    def diagnostics(self) -> Dict[str, object]:
        """Compute phase coherence, state SVDs, and parameter norms.

        Returns a dict with:
          - 'embed_phase_coherence': float
          - 'embed_mag_mean': float
          - 'embed_mag_std': float
          - 'layer_gammas': list of arrays
          - 'param_norms': dict of param name -> float (Frobenius norm)
        """
        p = self.params
        embed = p['embed']

        # Embedding phase coherence
        pc = phase_coherence(embed)

        # Magnitude stats
        mags = np.abs(embed)

        # Per-layer gammas
        gammas = []
        for i in range(self.n_layers):
            db = p[f'L{i}.decay_bias']
            g = 1.0 / (1.0 + np.exp(-db.real))
            gammas.append(g)

        # Parameter norms
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


class ComplexProbe:
    """Probe real-valued model embeddings in complex space.

    Wraps a set of real-valued embeddings (e.g. from a pretrained LLM),
    projects them to complex space, and provides phase analysis tools.

    Args:
        real_dim:    Dimension of the real embeddings.
        complex_dim: Dimension of the complex projection (default real_dim // 2).
        method:      Projection method ('hilbert', 'paired', 'random_proj').

    Example:
        >>> # Get embeddings from a real-valued model
        >>> real_emb = np.random.randn(50000, 768)  # e.g. GPT-2 token embeddings
        >>> probe = ComplexProbe(real_dim=768)
        >>> z = probe.project(real_emb)
        >>> metrics = probe.analyze(z)
    """

    def __init__(
        self,
        real_dim: int,
        complex_dim: Optional[int] = None,
        method: str = 'hilbert',
    ):
        self.real_dim = real_dim
        self.complex_dim = complex_dim or real_dim // 2
        self.method = method

    def project(self, real_embeddings: np.ndarray) -> np.ndarray:
        """Project real embeddings to complex space.

        Args:
            real_embeddings: Real array, shape [..., real_dim].

        Returns:
            Complex array, shape [..., complex_dim].
        """
        return embed_to_complex(real_embeddings, method=self.method)

    def analyze(self, z: np.ndarray) -> Dict[str, float]:
        """Compute phase metrics on complex embeddings.

        Args:
            z: Complex array, shape [n_vectors, dim].

        Returns:
            Dict with:
              - 'phase_coherence': float in [0, 1]
              - 'mag_mean': mean magnitude
              - 'mag_std': std of magnitudes
              - 'phase_std': std of phases (uniform random ~ 1.81)
              - 'mean_interference': mean pairwise interference score
        """
        pc = phase_coherence(z)
        mags = np.abs(z)
        phases = np.angle(z)

        # Mean pairwise interference (sample a subset if large)
        n = z.shape[0]
        if n > 200:
            idx = np.random.choice(n, 200, replace=False)
            z_sample = z[idx]
        else:
            z_sample = z

        # Normalized embeddings for interference
        z_unit = z_sample / (np.sqrt(np.sum(np.abs(z_sample) ** 2, axis=-1, keepdims=True)) + 1e-10)
        gram = (z_unit @ z_unit.conj().T).real
        # Extract upper triangle (exclude diagonal)
        n_s = z_sample.shape[0]
        triu_idx = np.triu_indices(n_s, k=1)
        mean_interference = float(np.mean(gram[triu_idx]))

        return {
            'phase_coherence': pc,
            'mag_mean': float(mags.mean()),
            'mag_std': float(mags.std()),
            'phase_std': float(phases.std()),
            'mean_interference': mean_interference,
        }
