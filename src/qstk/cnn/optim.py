"""
Complex Adam optimizer -- pure numpy.

Handles complex-valued parameters with Wirtinger-aware updates.
The first moment (m) is complex, tracking the exponential moving average
of complex gradients. The second moment (v) tracks |grad|^2 (real-valued),
which is the natural Wirtinger metric for complex optimization.

Classes:
    ComplexAdam -- Adam optimizer for dicts of complex numpy arrays.
"""

import numpy as np
from typing import Dict, Optional


class ComplexAdam:
    """Adam optimizer for complex-valued parameters.

    Following Wirtinger calculus conventions: the update direction is
    the conjugate of the Wirtinger derivative (natural gradient in the
    complex metric). The second moment uses |g|^2 = g_real^2 + g_imag^2,
    which is the proper norm for complex gradients.

    Args:
        params_dict: Dict mapping names to complex numpy arrays.
        lr:           Learning rate.
        beta1:        First moment decay (default 0.9).
        beta2:        Second moment decay (default 0.999).
        eps:          Epsilon for numerical stability.
        weight_decay: L2 regularization on complex magnitude.
        clip_percentile: Gradient clipping percentile (99 = clip top 1%).
                         Set to 0 or None to disable.

    Example:
        >>> params = {'W': complex_randn(32, 64), 'b': np.zeros(32, dtype=np.complex128)}
        >>> opt = ComplexAdam(params, lr=1e-3)
        >>> grads = {'W': ..., 'b': ...}
        >>> opt.step(grads)
    """

    def __init__(
        self,
        params_dict: Dict[str, np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        clip_percentile: float = 99.0,
    ):
        self.params = params_dict
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.clip_pct = clip_percentile
        self.t = 0

        # First moment (complex, same dtype as params)
        self.m: Dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in params_dict.items()
        }
        # Second moment (real, magnitude-squared of gradient)
        self.v: Dict[str, np.ndarray] = {
            k: np.zeros(v.shape, dtype=np.float64)
            for k, v in params_dict.items()
        }

    def step(self, grads: Dict[str, np.ndarray]) -> None:
        """Perform one Adam update step.

        Args:
            grads: Dict mapping parameter names to complex gradient arrays.
                   Names not in grads are skipped.
        """
        self.t += 1

        for name in self.params:
            if name not in grads:
                continue

            g = grads[name]

            # Weight decay (applied to complex params, not bias/scale)
            if self.wd > 0 and g.ndim >= 2:
                g = g + self.wd * self.params[name]

            # Gradient clipping by percentile
            if self.clip_pct and self.clip_pct < 100:
                g_mag = np.abs(g)
                max_mag = np.percentile(g_mag, self.clip_pct)
                if max_mag > 1.0:
                    g = g * (1.0 / max_mag)

            # First moment: EMA of complex gradient
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g

            # Second moment: EMA of |gradient|^2 (Wirtinger metric)
            self.v[name] = (
                self.beta2 * self.v[name]
                + (1 - self.beta2) * (g.real ** 2 + g.imag ** 2)
            )

            # Bias correction
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            # Update
            self.params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        """Reset optimizer state (moments and step counter).

        Useful when you want to restart optimization from scratch
        but keep the same parameter references.
        """
        self.t = 0
        for name in self.params:
            self.m[name] = np.zeros_like(self.m[name])
            self.v[name] = np.zeros_like(self.v[name])
