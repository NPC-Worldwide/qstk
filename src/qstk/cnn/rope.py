"""
Complex rotary position encoding -- pure numpy, native complex128.

In complex space, positional encoding is just multiplication by e^{i*pos*theta}.
No sin/cos decomposition needed. One complex multiply per position-dimension pair.

Functions:
    make_freqs    -- precompute the frequency table for RoPE
    complex_rope  -- apply rotary position encoding to complex embeddings
"""

import numpy as np
from typing import Optional


def make_freqs(
    dim: int,
    max_len: int,
    base: float = 10000.0,
    dtype: np.dtype = np.complex128,
) -> np.ndarray:
    """Precompute RoPE frequency table: e^{i * pos * theta_k}.

    theta_k = 1 / base^{k / dim}  for k = 0, 1, ..., dim-1.

    The result is a table of complex unit rotations indexed by
    (position, dimension).

    Args:
        dim:     Feature dimension (each dim gets its own frequency).
        max_len: Maximum sequence length.
        base:    Base for the geometric frequency schedule.
        dtype:   np.complex128 or np.complex64.

    Returns:
        Complex array of shape (max_len, dim), where entry [pos, k]
        is e^{i * pos * theta_k}.
    """
    k = np.arange(dim, dtype=np.float64)
    theta = 1.0 / (base ** (k / dim))  # [dim]
    positions = np.arange(max_len, dtype=np.float64)  # [max_len]
    angles = np.outer(positions, theta)  # [max_len, dim]
    freqs = np.exp(1j * angles)  # complex unit rotations
    if dtype == np.complex64:
        freqs = freqs.astype(np.complex64)
    return freqs


def complex_rope(
    x: np.ndarray,
    seq_len: int,
    base: float = 10000.0,
    freqs: Optional[np.ndarray] = None,
    offset: int = 0,
) -> np.ndarray:
    """Apply complex rotary position encoding.

    Each position-dimension pair is multiplied by e^{i * pos * theta_k}.
    This rotates the phase by an amount proportional to position,
    with different frequencies per dimension. The result is that
    relative position information is encoded in phase differences.

    In complex space this is a single element-wise multiply. Compare to
    the real-valued RoPE which needs sin/cos interleaving and two multiplies.

    Args:
        x:        Complex array, shape [..., seq_len, dim].
        seq_len:  Sequence length (must match x.shape[-2]).
        base:     Base for frequency schedule (default 10000).
        freqs:    Optional precomputed frequency table from make_freqs().
                  If None, computed on the fly.
        offset:   Position offset (for inference with cached state).

    Returns:
        Complex array, same shape as x, with RoPE applied.

    Example:
        >>> x = complex_randn(2, 32, 64)  # [batch, seq, dim]
        >>> x_pos = complex_rope(x, seq_len=32)
        >>> x_pos.shape  # (2, 32, 64)
    """
    dim = x.shape[-1]
    if freqs is None:
        freqs = make_freqs(dim, offset + seq_len, base=base, dtype=x.dtype)
    rope_slice = freqs[offset:offset + seq_len, :dim]  # [seq_len, dim]
    return x * rope_slice
