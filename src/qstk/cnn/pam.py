"""
Phase-Associative Memory (PAM) layer -- pure numpy, native complex128.

The PAM replaces attention with a complex matrix state that is updated
via rank-1 outer products and retrieved via complex matrix-vector multiply.
Phase interference naturally implements attention without softmax.

State update:  S_t = gamma * S_{t-1} + V_t (x) conj(K_t)
Retrieval:     Y_t = S_t @ Q_t

The decay gamma is learnable per-head via sigmoid(decay_bias).

Two compute modes:
  - 'sequential': O(T*d^2) per head, materializes state at each step.
    Required for inference and gives exact recurrent dynamics.
  - 'dual': O(T^2*d) per head, uses the attention-form expansion.
    More efficient for training when T < d.

Classes:
    PAMLayer -- one PAM layer with Q/K/V/O projections and multi-head state.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Literal

from .layers import ComplexLinear, ComplexNorm, mod_relu, complex_glorot, complex_randn


class PAMLayer:
    """Phase-Associative Memory layer.

    Multi-head complex matrix state with learnable decay. Each head
    maintains an independent d_head x d_head complex state matrix.

    Args:
        dim:        Model dimension (input/output).
        heads:      Number of attention heads.
        d_head:     Dimension per head.
        decay_init: Initial value for decay bias (before sigmoid).
                    Default -2.0 gives gamma ~ 0.12 (fast decay, safe start).
        dtype:      np.complex128 or np.complex64.

    Attributes:
        params: Dict of all learnable parameters (projections + decay + norm).

    Example:
        >>> pam = PAMLayer(dim=64, heads=4, d_head=16)
        >>> x = complex_randn(2, 32, 64)  # [batch, seq, dim]
        >>> y = pam.forward(x, mode='sequential')  # [2, 32, 64]
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        d_head: int,
        decay_init: float = -2.0,
        dtype: np.dtype = np.complex128,
    ):
        self.dim = dim
        self.heads = heads
        self.d_head = d_head
        self.inner_dim = heads * d_head
        self.dtype = dtype

        # Projections
        self.Wq = ComplexLinear(dim, self.inner_dim, dtype=dtype)
        self.Wk = ComplexLinear(dim, self.inner_dim, dtype=dtype)
        self.Wv = ComplexLinear(dim, self.inner_dim, dtype=dtype)
        self.Wo = ComplexLinear(self.inner_dim, dim, dtype=dtype)

        # Scale Wo down for stable residuals
        self.Wo.params['W'] *= 0.5

        # Learnable decay bias: sigmoid(decay_bias) -> gamma
        self.decay_bias = np.full(heads, decay_init, dtype=dtype)

        # modReLU bias (per output dim)
        self.modrelu_bias = np.full(dim, -0.05, dtype=dtype)

        # Pre-norm
        self.norm = ComplexNorm(dim, dtype=dtype)

    @property
    def params(self) -> Dict[str, np.ndarray]:
        """Collect all learnable parameters into a flat dict."""
        p: Dict[str, np.ndarray] = {}
        p['Wq'] = self.Wq.params['W']
        p['Wk'] = self.Wk.params['W']
        p['Wv'] = self.Wv.params['W']
        p['Wo'] = self.Wo.params['W']
        p['decay_bias'] = self.decay_bias
        p['modrelu_bias'] = self.modrelu_bias
        p['norm_scale'] = self.norm.params['scale']
        return p

    @params.setter
    def params(self, p: Dict[str, np.ndarray]) -> None:
        """Set parameters from a flat dict (inverse of getter)."""
        self.Wq.params['W'] = p['Wq']
        self.Wk.params['W'] = p['Wk']
        self.Wv.params['W'] = p['Wv']
        self.Wo.params['W'] = p['Wo']
        self.decay_bias = p['decay_bias']
        self.modrelu_bias = p['modrelu_bias']
        self.norm.params['scale'] = p['norm_scale']

    def _gamma(self) -> np.ndarray:
        """Compute decay factor: gamma = sigmoid(decay_bias.real)."""
        return 1.0 / (1.0 + np.exp(-self.decay_bias.real))  # [heads]

    def _project_qkv(
        self, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project input to Q, K, V and reshape to multi-head.

        Args:
            h: [..., B, T, dim] complex.

        Returns:
            q, k, v each of shape [B, T, heads, d_head] complex.
        """
        B, T, _ = h.shape
        q = self.Wq(h).reshape(B, T, self.heads, self.d_head)
        k = self.Wk(h).reshape(B, T, self.heads, self.d_head)
        v = self.Wv(h).reshape(B, T, self.heads, self.d_head)
        return q, k, v

    def forward(
        self,
        x: np.ndarray,
        mode: Literal['sequential', 'dual'] = 'sequential',
        state: Optional[np.ndarray] = None,
        residual_scale: float = 0.5,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Forward pass through PAM layer.

        Args:
            x:     Complex array, shape [B, T, dim].
            mode:  'sequential' (exact recurrence, O(T*d^2)) or
                   'dual' (attention form, O(T^2*d)).
            state: Optional initial state [B, heads, d_head, d_head],
                   used for continued generation in sequential mode.
            residual_scale: Scale factor for residual connection.

        Returns:
            Tuple of:
              - output: Complex array [B, T, dim].
              - final_state: Complex array [B, heads, d_head, d_head] or None.
        """
        B, T, D = x.shape

        # Pre-norm
        h = self.norm(x)

        # Project
        q, k, v = self._project_qkv(h)

        # QK normalization: unit magnitude, pure phase
        q = q / (np.abs(q) + 1e-8)
        k = k / (np.abs(k) + 1e-8)

        scale = self.d_head ** -0.5
        q = q * scale

        gamma = self._gamma()  # [heads]

        if mode == 'sequential':
            y_heads, final_state = self._sequential(q, k, v, gamma, B, T, state)
        elif mode == 'dual':
            y_heads, final_state = self._dual_form(q, k, v, gamma, B, T)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'sequential' or 'dual'.")

        # Merge heads and project out
        y_merged = y_heads.reshape(B, T, self.inner_dim)
        y_out = self.Wo(y_merged)

        # modReLU activation
        y_act = mod_relu(y_out, self.modrelu_bias)

        # Residual connection
        output = x + residual_scale * y_act

        return output, final_state

    def _sequential(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        gamma: np.ndarray,
        B: int,
        T: int,
        state: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sequential (recurrent) PAM computation.

        Iterates through time steps, maintaining the full state matrix.
        O(T * d^2) per head. Required for inference / generation.

        Args:
            q, k, v: [B, T, heads, d_head] complex.
            gamma:   [heads] real.
            B, T:    Batch size and sequence length.
            state:   Optional [B, heads, d_head, d_head] complex initial state.

        Returns:
            y_heads:    [B, T, heads, d_head] complex.
            final_state: [B, heads, d_head, d_head] complex.
        """
        H, d = self.heads, self.d_head

        if state is None:
            S = np.zeros((B, H, d, d), dtype=self.dtype)
        else:
            S = state.copy()

        y_heads = np.zeros((B, T, H, d), dtype=self.dtype)

        for t in range(T):
            # outer(v_t, conj(k_t)): [B, H, d, 1] * [B, H, 1, d] -> [B, H, d, d]
            outer = (
                v[:, t, :, :, None]
                * k[:, t, :, None, :].conj()
            )
            # State update: S = gamma * S + outer
            S = S * gamma[None, :, None, None] + outer
            # Retrieval: y_t = S @ q_t
            y_heads[:, t] = np.einsum('bhij,bhj->bhi', S, q[:, t])

        return y_heads, S

    def _dual_form(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        gamma: np.ndarray,
        B: int,
        T: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dual-form (attention-form) PAM computation.

        Expands the recurrence into a T x T attention-like matrix with
        geometric decay. O(T^2 * d) per head. More efficient for training
        when T is moderate relative to d.

        Args:
            q, k, v: [B, T, heads, d_head] complex.
            gamma:   [heads] real.
            B, T:    Batch size and sequence length.

        Returns:
            y_heads:    [B, T, heads, d_head] complex.
            final_state: [B, heads, d_head, d_head] complex (state at t=T).
        """
        H, d = self.heads, self.d_head

        # Build decay matrix D[t, i] = gamma^{t - i} for i <= t, else 0.
        # In log space for stability.
        log_gamma = np.log(gamma + 1e-12)  # [H]

        # Relative distances: dist[t, i] = t - i
        t_idx = np.arange(T)
        dist = t_idx[:, None] - t_idx[None, :]  # [T, T]
        causal_mask = (dist >= 0).astype(np.float64)  # lower triangular

        # D[h, t, i] = gamma_h^{dist} * mask
        # shape: [H, T, T]
        log_D = log_gamma[:, None, None] * dist[None, :, :]  # [H, T, T]
        D = np.exp(np.clip(log_D, -20.0, 0.0)) * causal_mask[None, :, :]  # [H, T, T]

        # Complex dot product: W[b, h, t, i] = q[b, t, h, :] . conj(k[b, i, h, :])
        # Transpose to [B, H, T, d] for matmul
        q_bht = q.transpose(0, 2, 1, 3)  # [B, H, T, d]
        k_bht = k.transpose(0, 2, 1, 3)  # [B, H, T, d]
        v_bht = v.transpose(0, 2, 1, 3)  # [B, H, T, d]

        # W = Q @ K^H  -> [B, H, T, T]
        W = q_bht @ k_bht.conj().transpose(0, 1, 3, 2)

        # Apply decay: A = W * D
        A = W * D[None, :, :, :]  # [B, H, T, T]

        # Output: Y = A @ V  -> [B, H, T, d]
        Y = A @ v_bht  # [B, H, T, d]

        # Transpose back to [B, T, H, d]
        y_heads = Y.transpose(0, 2, 1, 3)  # [B, T, H, d]

        # Compute final state S_T for potential continued generation
        # S_T = sum_{i=0}^{T-1} gamma^{T-1-i} * outer(v_i, conj(k_i))
        D_last = D[:, -1, :]  # [H, T] -- decay from each position to T-1
        # Weighted v and k
        wv = v_bht * D_last[None, :, :, None]  # [B, H, T, d]
        # S_T = sum_i wv_i @ conj(k_i).T  =  wv.T @ conj(k)
        final_state = wv.transpose(0, 1, 3, 2) @ k_bht.conj()  # [B, H, d, d]

        return y_heads, final_state

    def __call__(
        self,
        x: np.ndarray,
        mode: Literal['sequential', 'dual'] = 'sequential',
        state: Optional[np.ndarray] = None,
        residual_scale: float = 0.5,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Alias for forward(). See forward() docstring."""
        return self.forward(x, mode=mode, state=state, residual_scale=residual_scale)
