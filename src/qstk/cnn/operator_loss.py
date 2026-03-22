"""Differentiable operator-space style loss for training.

Provides a torch-native version of the operator extraction pipeline
so it can be used as an auxiliary training objective. The loss penalizes
deviations from a target style's operator distribution.

Usage:
    from operator_loss import OperatorStyleLoss

    loss_fn = OperatorStyleLoss.from_profile('style_profile.npz')

    # In training loop:
    embeddings = model.transformer.wte(input_ids)  # [B, T, d]
    style_loss = loss_fn(embeddings)
    total_loss = lm_loss + lambda_ * style_loss
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hilbert_projection_torch(x: torch.Tensor) -> torch.Tensor:
    """Project real embeddings R^d -> C^{d/2} via Hilbert transform.

    Torch-native implementation of probe.embed_to_complex('hilbert').

    Args:
        x: Real tensor, shape [..., d]. d must be even.

    Returns:
        Complex tensor, shape [..., d//2].
    """
    d = x.shape[-1]
    X = torch.fft.fft(x, dim=-1)

    h = torch.zeros(d, device=x.device, dtype=x.dtype)
    h[0] = 1.0
    if d % 2 == 0:
        h[d // 2] = 1.0
    h[1:(d + 1) // 2] = 2.0

    Z = X * h
    analytic = torch.fft.ifft(Z, dim=-1)
    return analytic[..., :d // 2]


def extract_operators_torch(complex_embs: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Extract transition operators between consecutive complex embeddings.

    Torch-native implementation of operators.extract_operators.

    Args:
        complex_embs: Complex tensor, shape [T, d].

    Returns:
        Complex tensor, shape [T-1, d]. Element-wise z_{t+1} / z_t.
    """
    z1 = complex_embs[:-1]
    z2 = complex_embs[1:]

    mag1 = torch.abs(z1)
    mag2 = torch.abs(z2)
    scale = mag2 / torch.clamp(mag1, min=eps)

    phase1 = torch.angle(z1)
    phase2 = torch.angle(z2)
    delta_phase = phase2 - phase1

    return scale * torch.exp(1j * delta_phase)


class OperatorStyleLoss(nn.Module):
    """Auxiliary loss that penalizes deviation from a target operator distribution.

    Computes three differentiable terms:
    1. Phase diversity loss: KL between generated phase distribution and target
    2. Rotation magnitude loss: MSE between mean rotation magnitudes
    3. Coherence loss: MSE between trajectory coherence statistics

    The total loss is a weighted sum of these terms.
    """

    def __init__(
        self,
        target_phase_diversity: float,
        target_magnitude_diversity: float,
        target_mean_rotation: float,
        target_phase_alignment: float,
        target_eigenvalues: torch.Tensor,
        w_phase: float = 1.0,
        w_rotation: float = 1.0,
        w_coherence: float = 0.5,
        w_spectrum: float = 0.5,
    ):
        super().__init__()
        self.target_phase_diversity = target_phase_diversity
        self.target_magnitude_diversity = target_magnitude_diversity
        self.target_mean_rotation = target_mean_rotation
        self.target_phase_alignment = target_phase_alignment
        self.register_buffer('target_eigenvalues', target_eigenvalues)

        self.w_phase = w_phase
        self.w_rotation = w_rotation
        self.w_coherence = w_coherence
        self.w_spectrum = w_spectrum

    @classmethod
    def from_profile(cls, profile_path: str, **kwargs) -> 'OperatorStyleLoss':
        """Create loss from a saved style profile."""
        path = Path(profile_path)
        arrays = np.load(path.with_suffix('.npz'))
        with open(path.with_suffix('.json')) as f:
            meta = json.load(f)

        eigenvalues = torch.tensor(arrays['eigenvalues'], dtype=torch.float32)

        return cls(
            target_phase_diversity=meta['diversity']['phase_diversity'],
            target_magnitude_diversity=meta['diversity']['magnitude_diversity'],
            target_mean_rotation=float(np.mean(np.abs(np.angle(arrays['operators'])))),
            target_phase_alignment=meta['coherence_stats']['phase_alignment'],
            target_eigenvalues=eigenvalues,
            **kwargs,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute operator-space style loss.

        Args:
            embeddings: Real-valued embeddings, shape [B, T, d] or [T, d].

        Returns:
            Scalar loss tensor.
        """
        if embeddings.dim() == 3:
            # Process each batch element and average
            losses = []
            for b in range(embeddings.shape[0]):
                losses.append(self._single_loss(embeddings[b]))
            return torch.stack(losses).mean()
        else:
            return self._single_loss(embeddings)

    def _single_loss(self, embs: torch.Tensor) -> torch.Tensor:
        """Compute loss for a single sequence [T, d]."""
        if embs.shape[0] < 4:
            return torch.tensor(0.0, device=embs.device, requires_grad=True)

        # Project to complex space
        complex_embs = hilbert_projection_torch(embs)
        operators = extract_operators_torch(complex_embs)

        n_ops = operators.shape[0]
        if n_ops < 2:
            return torch.tensor(0.0, device=embs.device, requires_grad=True)

        # --- Phase diversity loss ---
        phases = torch.angle(operators)  # [n_ops, d]
        mean_phase_per_op = phases.mean(dim=1)  # [n_ops]
        resultant = torch.abs(torch.mean(torch.exp(1j * mean_phase_per_op.to(torch.complex64))))
        phase_div = 1.0 - resultant
        phase_loss = (phase_div - self.target_phase_diversity) ** 2

        # --- Rotation magnitude loss ---
        mean_rotation = torch.mean(torch.abs(phases))
        rotation_loss = (mean_rotation - self.target_mean_rotation) ** 2

        # --- Coherence loss ---
        phase_diffs = phases[1:] - phases[:-1]
        alignment = torch.mean(torch.cos(phase_diffs))
        coherence_loss = (alignment - self.target_phase_alignment) ** 2

        # --- Spectrum loss ---
        # Compare eigenvalue distribution of operator PCA
        # Use the variance of phase vectors as a proxy (differentiable)
        phase_var = torch.var(phases, dim=0)  # [d]
        sorted_var, _ = torch.sort(phase_var, descending=True)
        n_comp = min(len(self.target_eigenvalues), len(sorted_var))
        target_ev = self.target_eigenvalues[:n_comp].to(embs.device)
        # Normalize both to sum to 1 for comparison
        gen_ev = sorted_var[:n_comp] / (sorted_var[:n_comp].sum() + 1e-10)
        tgt_ev = target_ev / (target_ev.sum() + 1e-10)
        spectrum_loss = F.mse_loss(gen_ev, tgt_ev)

        total = (
            self.w_phase * phase_loss
            + self.w_rotation * rotation_loss
            + self.w_coherence * coherence_loss
            + self.w_spectrum * spectrum_loss
        )

        return total
