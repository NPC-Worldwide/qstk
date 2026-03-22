#!/usr/bin/env python3
"""Operator-guided constrained decoding for style transfer.

Given a precomputed style profile (from style_profile.py), this sampler
wraps any HuggingFace causal LM and constrains token selection to stay
within the operator-space distribution of the target style.

At each generation step:
1. Predict the most likely next operator cluster from the transition matrix
2. Retrieve the token vocabulary reachable by that cluster
3. Mask the logits to zero out unreachable tokens
4. Sample from the constrained distribution

The constraint strength is tunable: at alpha=0 the sampler is unconstrained,
at alpha=1 it strictly restricts to cluster vocabulary.

Optionally integrates dynamic temperature from the Cognitive Temperature
framework: Poisson-distributed switching between exploration (high temp,
looser constraint) and convergence (low temp, tight constraint).

Usage:
    from operator_sampler import OperatorGuidedSampler

    sampler = OperatorGuidedSampler.from_profile('style_profile.npz')
    text = sampler.generate("The enclosure of the commons", max_new_tokens=200)
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from qstk.cnn.probe import embed_to_complex
from qstk.cnn.operators import transition_operator


class OperatorGuidedSampler:
    """Constrained decoding using operator-space style profiles."""

    def __init__(
        self,
        model,
        tokenizer,
        cluster_centroids: np.ndarray,
        cluster_vocabs: dict[int, list[int]],
        cluster_token_probs: dict[int, dict[int, float]],
        transition_matrix: np.ndarray,
        n_clusters: int,
        alpha: float = 0.7,
        temperature: float = 0.8,
        top_p: float = 0.95,
        dynamic_temp: bool = False,
        temp_low: float = 0.4,
        temp_high: float = 1.2,
        poisson_lambda: float = 15.0,
        device: str = 'auto',
    ):
        """
        Args:
            model: HuggingFace causal LM (e.g., GPT2LMHeadModel).
            tokenizer: Corresponding tokenizer.
            cluster_centroids: [k, 2*d] cos/sin phase centroids from KMeans.
            cluster_vocabs: {cluster_id: [token_ids]} reachable tokens per cluster.
            cluster_token_probs: {cluster_id: {token_id: prob}} within-cluster distribution.
            transition_matrix: [k, k] P(next_cluster | current_cluster).
            n_clusters: Number of operator clusters.
            alpha: Constraint strength in [0, 1]. 0 = unconstrained, 1 = strict.
            temperature: Base sampling temperature.
            top_p: Nucleus sampling threshold.
            dynamic_temp: Whether to use Poisson-switched temperature modulation.
            temp_low: Low temperature for convergent phases.
            temp_high: High temperature for exploratory phases.
            poisson_lambda: Mean tokens between temperature switches.
            device: 'cuda', 'cpu', or 'auto'.
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

        self.cluster_centroids = cluster_centroids
        self.cluster_vocabs = cluster_vocabs
        self.cluster_token_probs = cluster_token_probs
        self.transition_matrix = transition_matrix
        self.n_clusters = n_clusters

        self.alpha = alpha
        self.temperature = temperature
        self.top_p = top_p
        self.dynamic_temp = dynamic_temp
        self.temp_low = temp_low
        self.temp_high = temp_high
        self.poisson_lambda = poisson_lambda

        # Precompute cluster masks as boolean tensors for fast logit masking
        vocab_size = tokenizer.vocab_size
        self.cluster_masks = {}
        for c, tids in cluster_vocabs.items():
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            mask[tids] = True
            self.cluster_masks[c] = mask

    @classmethod
    def from_profile(
        cls,
        profile_path: str,
        model_name: str = 'gpt2',
        alpha: float = 0.7,
        temperature: float = 0.8,
        dynamic_temp: bool = False,
        device: str = 'auto',
    ) -> 'OperatorGuidedSampler':
        """Load a style profile and create a sampler."""
        from qstk.cnn.style_profile import load_profile
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        profile = load_profile(profile_path)

        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        return cls(
            model=model,
            tokenizer=tokenizer,
            cluster_centroids=profile['cluster_centroids'],
            cluster_vocabs=profile['cluster_vocabs'],
            cluster_token_probs=profile['cluster_token_probs'],
            transition_matrix=profile['transition_matrix'],
            n_clusters=profile['n_clusters'],
            alpha=alpha,
            temperature=temperature,
            dynamic_temp=dynamic_temp,
            device=device,
        )

    def _predict_next_cluster(self, current_cluster: int) -> int:
        """Predict the most likely next operator cluster."""
        probs = self.transition_matrix[current_cluster]
        return int(np.random.choice(self.n_clusters, p=probs))

    def _assign_cluster(self, operator: np.ndarray) -> int:
        """Assign an operator to its nearest cluster via phase signature."""
        phases = np.angle(operator)
        cos_sin = np.concatenate([np.cos(phases), np.sin(phases)])
        distances = np.linalg.norm(self.cluster_centroids - cos_sin, axis=1)
        return int(np.argmin(distances))

    def _get_current_temp(self, step: int, rng: np.random.RandomState) -> float:
        """Get temperature for current step, optionally with Poisson switching."""
        if not self.dynamic_temp:
            return self.temperature

        # Poisson-distributed regime switching
        if not hasattr(self, '_temp_state'):
            self._temp_state = {'regime': 'low', 'next_switch': rng.poisson(self.poisson_lambda)}

        if step >= self._temp_state['next_switch']:
            self._temp_state['regime'] = 'high' if self._temp_state['regime'] == 'low' else 'low'
            self._temp_state['next_switch'] = step + rng.poisson(self.poisson_lambda)

        return self.temp_low if self._temp_state['regime'] == 'low' else self.temp_high

    def _apply_constraint(self, logits: torch.Tensor, predicted_cluster: int) -> torch.Tensor:
        """Apply operator-guided constraint to logits.

        Blends unconstrained logits with cluster-masked logits according to alpha.
        """
        if self.alpha == 0:
            return logits

        mask = self.cluster_masks[predicted_cluster]

        if self.alpha >= 1.0:
            # Strict: zero out everything not in cluster vocabulary
            constrained = logits.clone()
            constrained[~mask] = float('-inf')
            return constrained

        # Soft: blend. Reduce probability of out-of-cluster tokens by alpha.
        penalty = torch.zeros_like(logits)
        penalty[~mask] = -self.alpha * 10.0  # log-space penalty
        return logits + penalty

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        seed: int = 42,
    ) -> str:
        """Generate text with operator-guided constrained decoding."""
        rng = np.random.RandomState(seed)
        if hasattr(self, '_temp_state'):
            del self._temp_state

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated = input_ids[0].tolist()

        # Get initial embedding for operator tracking
        with torch.no_grad():
            prev_emb = self.model.transformer.wte(input_ids[0, -1:]).cpu().numpy().astype(np.float64)
        prev_complex = embed_to_complex(prev_emb, method='hilbert')[0]

        current_cluster = 0  # start neutral

        for step in range(max_new_tokens):
            input_tensor = torch.tensor([generated], dtype=torch.long, device=self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs.logits[0, -1, :]

            # Temperature
            temp = self._get_current_temp(step, rng)
            logits = logits / temp

            # Predict next operator cluster and constrain
            predicted_cluster = self._predict_next_cluster(current_cluster)
            logits = self._apply_constraint(logits, predicted_cluster)

            # Nucleus sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            # Update operator tracking
            with torch.no_grad():
                curr_emb = self.model.transformer.wte(
                    torch.tensor([next_token], device=self.device)
                ).cpu().numpy().astype(np.float64)
            curr_complex = embed_to_complex(curr_emb, method='hilbert')[0]
            op = transition_operator(prev_complex, curr_complex)
            current_cluster = self._assign_cluster(op)
            prev_complex = curr_complex

            if next_token == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate text with operator-guided style constraint')
    parser.add_argument('--profile', type=str, required=True, help='Path to style profile (.npz)')
    parser.add_argument('--prompt', type=str, default='The enclosure of the commons')
    parser.add_argument('--max_tokens', type=int, default=200)
    parser.add_argument('--alpha', type=float, default=0.7, help='Constraint strength [0,1]')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--dynamic_temp', action='store_true')
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    sampler = OperatorGuidedSampler.from_profile(
        args.profile,
        model_name=args.model,
        alpha=args.alpha,
        temperature=args.temperature,
        dynamic_temp=args.dynamic_temp,
    )

    text = sampler.generate(args.prompt, max_new_tokens=args.max_tokens, seed=args.seed)
    print(text)
