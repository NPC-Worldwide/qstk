"""
Complex-valued neural network building blocks -- pure numpy, native complex128.

Every array is a genuine numpy complex dtype. No real-pair [..., 2] tricks.
Phase is a first-class citizen: modReLU gates magnitude, CGU gates both.

Classes:
    ComplexLinear   -- complex-valued linear projection
    ComplexEmbed    -- complex embedding lookup table
    ComplexNorm     -- RMS normalization on magnitude, phase preserved

Functions:
    complex_randn   -- sample complex Gaussian
    complex_glorot  -- Glorot-scaled complex init
    mod_relu        -- phase-preserving activation (threshold magnitude)
    complex_gated_unit -- SwiGLU-style complex gating
"""

import numpy as np
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------

def complex_randn(
    *shape: int,
    scale: float = 0.02,
    dtype: np.dtype = np.complex128,
) -> np.ndarray:
    """Sample complex Gaussian: z = (x + iy) * scale, x,y ~ N(0,1).

    Args:
        *shape: Dimensions of the output array.
        scale: Standard deviation of each component (real and imag).
        dtype: np.complex128 or np.complex64.

    Returns:
        Complex array of the given shape and dtype.
    """
    real_dtype = np.float64 if dtype == np.complex128 else np.float32
    re = np.random.randn(*shape).astype(real_dtype)
    im = np.random.randn(*shape).astype(real_dtype)
    return (re + 1j * im).astype(dtype) * scale


def complex_glorot(
    fan_in: int,
    fan_out: int,
    dtype: np.dtype = np.complex128,
) -> np.ndarray:
    """Glorot (Xavier) initialization for complex weights.

    Scale = sqrt(2 / (fan_in + fan_out)), applied to both real and imag.

    Args:
        fan_in:  Number of input features.
        fan_out: Number of output features.
        dtype:   np.complex128 or np.complex64.

    Returns:
        Complex array of shape (fan_out, fan_in).
    """
    scale = np.sqrt(2.0 / (fan_in + fan_out))
    return complex_randn(fan_out, fan_in, scale=scale, dtype=dtype)


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def mod_relu(z: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Phase-preserving activation: threshold magnitude, keep phase.

    modReLU(z) = max(0, |z| + b) * z/|z|

    The bias is a learnable per-dimension scalar (real part used).
    When |z| + b <= 0 the output is exactly zero (killed).
    When |z| + b > 0 the phase is perfectly preserved.

    Args:
        z:    Complex array, shape [..., dim].
        bias: Complex array (real part used), shape [dim].

    Returns:
        Complex array, same shape as z.
    """
    mag = np.abs(z)  # [..., dim]
    activated = np.maximum(0.0, mag + bias.real)
    phase = z / (mag + 1e-8)
    return phase * activated


def complex_gated_unit(
    z: np.ndarray,
    W_gate: np.ndarray,
    W_val: np.ndarray,
) -> np.ndarray:
    """Complex Gated Unit (CGU): SwiGLU-style gating in complex space.

    gate = sigmoid(|W_gate @ z|)   -- magnitude controls how much passes
    val  = W_val @ z               -- complex-valued content
    out  = gate * val              -- gated output

    The gate operates on magnitudes (real-valued sigmoid), so it controls
    *intensity*. The value path preserves full complex structure (phase).

    Args:
        z:      Complex array, shape [..., in_dim].
        W_gate: Complex array, shape [hidden_dim, in_dim].
        W_val:  Complex array, shape [hidden_dim, in_dim].

    Returns:
        Complex array, shape [..., hidden_dim].
    """
    gate_pre = z @ W_gate.T  # [..., hidden_dim]
    gate = 1.0 / (1.0 + np.exp(-np.abs(gate_pre)))  # sigmoid on magnitude
    val = z @ W_val.T  # [..., hidden_dim]
    return gate * val


# ---------------------------------------------------------------------------
# ComplexLinear
# ---------------------------------------------------------------------------

class ComplexLinear:
    """Complex-valued linear layer: y = x @ W.T (no bias, all complex).

    The weight matrix W is a native complex array. Forward pass is a
    single complex matrix multiply -- numpy handles the algebra.

    Args:
        in_dim:  Input feature dimension.
        out_dim: Output feature dimension.
        dtype:   np.complex128 or np.complex64.

    Attributes:
        params: Dict with key 'W' -> complex array (out_dim, in_dim).

    Example:
        >>> layer = ComplexLinear(64, 32)
        >>> x = complex_randn(4, 10, 64)  # [batch, seq, dim]
        >>> y = layer(x)                   # [batch, seq, 32]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: np.dtype = np.complex128,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.params: Dict[str, np.ndarray] = {
            'W': complex_glorot(in_dim, out_dim, dtype=dtype),
        }

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward: y = x @ W.T.

        Args:
            x: Complex array, shape [..., in_dim].

        Returns:
            Complex array, shape [..., out_dim].
        """
        return x @ self.params['W'].T


# ---------------------------------------------------------------------------
# ComplexEmbed
# ---------------------------------------------------------------------------

class ComplexEmbed:
    """Complex-valued embedding table.

    Each token maps to a complex vector. Magnitude encodes salience,
    phase encodes semantic identity.

    Args:
        vocab_size: Number of tokens.
        dim:        Embedding dimension (complex).
        dtype:      np.complex128 or np.complex64.

    Attributes:
        params: Dict with key 'W' -> complex array (vocab_size, dim).

    Example:
        >>> embed = ComplexEmbed(256, 64)
        >>> ids = np.array([[0, 1, 2], [3, 4, 5]])
        >>> z = embed(ids)  # [2, 3, 64] complex128
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        dtype: np.dtype = np.complex128,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.params: Dict[str, np.ndarray] = {
            'W': complex_randn(vocab_size, dim, scale=0.1, dtype=dtype),
        }

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        """Lookup embeddings for token ids.

        Args:
            token_ids: Integer array, shape [...].

        Returns:
            Complex array, shape [..., dim].
        """
        return self.params['W'][token_ids]


# ---------------------------------------------------------------------------
# ComplexNorm
# ---------------------------------------------------------------------------

class ComplexNorm:
    """RMS normalization on magnitude, phase preserved, learnable scale.

    For complex z with magnitude |z| and phase phi:
        rms  = sqrt(mean(|z|^2))
        out  = (z / |z|) * (|z| / rms) * scale

    The phase is untouched. Magnitude is normalized to unit RMS, then
    scaled by a learnable per-dimension factor.

    Args:
        dim:   Feature dimension.
        eps:   Epsilon for numerical stability.
        dtype: np.complex128 or np.complex64.

    Attributes:
        params: Dict with key 'scale' -> complex array [dim] (real part used).

    Example:
        >>> norm = ComplexNorm(64)
        >>> z = complex_randn(4, 10, 64, scale=5.0)
        >>> z_normed = norm(z)
        >>> np.abs(z_normed).mean()  # close to scale value (~1.0)
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        dtype: np.dtype = np.complex128,
    ):
        self.dim = dim
        self.eps = eps
        self.params: Dict[str, np.ndarray] = {
            'scale': np.ones(dim, dtype=dtype),
        }

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Normalize magnitudes to unit RMS, preserve phase, apply scale.

        Args:
            z: Complex array, shape [..., dim].

        Returns:
            Complex array, same shape, normalized.
        """
        mag = np.abs(z)  # [..., dim]
        rms = np.sqrt(np.mean(mag ** 2, axis=-1, keepdims=True) + self.eps)
        phase = z / (mag + 1e-8)
        return phase * (mag / rms) * np.abs(self.params['scale'])
