"""
Complex-valued Knowledge Graph embeddings with phase-interference retrieval.

The central idea: entities live as complex vectors where magnitude encodes
salience and phase encodes semantic identity. Relations are pure rotations
on the unit circle -- composing relations is just multiplying phases.
Retrieval from memory is phase interference: aligned phases reinforce
(constructive), misaligned phases cancel (destructive). No softmax needed.

This module provides three layers of abstraction:

  1. ComplexKGEmbedding  -- learnable entity/relation embeddings with
                            RotatE-style and TransE-style scoring
  2. PAMGraphMemory      -- entire graph stored as a complex outer-product
                            state matrix; retrieval via matrix-vector multiply
  3. ComplexKG           -- combines both for training, encoding, and search

Plus a RealKGBaseline for apples-to-apples comparison, and standard
evaluation utilities (hits@k, MRR, mean rank).

All pure numpy, native complex128. No PyTorch.

Classes:
    ComplexKGEmbedding  -- complex entity/relation embeddings
    PAMGraphMemory      -- phase-associative graph memory
    ComplexKG           -- full complex-valued knowledge graph model
    RealKGBaseline      -- real-valued TransE baseline

Functions:
    hits_at_k           -- fraction of correct entities in top-k
    mean_rank           -- average rank of correct entity
    mrr                 -- mean reciprocal rank
    compare_to_baseline -- side-by-side evaluation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .layers import complex_randn


# ---------------------------------------------------------------------------
# ComplexKGEmbedding
# ---------------------------------------------------------------------------

class ComplexKGEmbedding:
    """Complex-valued entity and relation embeddings for knowledge graphs.

    Entities are general complex vectors: magnitude = salience, phase = identity.
    Relations are initialized on the unit circle (pure phase rotations), so
    composing relations h * r1 * r2 is just adding angles -- a rotation group.

    Two scoring functions:

      RotatE-style:  score(h, r, t) = Re(sum(h * r * conj(t)))
        Measures phase alignment after rotating h by r. If the rotated head
        aligns with the tail, phases match and the sum is large (constructive
        interference). Misaligned triples cancel (destructive).

      TransE-style:  score(h, r, t) = -||h + r - t||  in complex space
        Translation in the complex plane. The norm is the full complex norm.

    Args:
        n_entities:  Number of entities.
        n_relations: Number of relation types.
        dim:         Complex embedding dimension.
        dtype:       np.complex128 or np.complex64.

    Attributes:
        params: Dict with 'entities' and 'relations' arrays.
    """

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        dim: int,
        dtype: np.dtype = np.complex128,
    ):
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.dim = dim
        self.dtype = dtype

        # Entity embeddings: general complex vectors
        self.params: Dict[str, np.ndarray] = {
            'entities': complex_randn(n_entities, dim, scale=0.1, dtype=dtype),
            'relations': self._init_relations(n_relations, dim, dtype),
        }

    @staticmethod
    def _init_relations(
        n_relations: int,
        dim: int,
        dtype: np.dtype,
    ) -> np.ndarray:
        """Initialize relations as unit-circle rotations: e^{i*theta}.

        Each relation is a vector of independent phase rotations.
        Theta is drawn uniformly from [-pi, pi], so the initial relations
        cover the full circle. Magnitude is exactly 1 -- pure rotation.
        """
        theta = np.random.uniform(-np.pi, np.pi, size=(n_relations, dim))
        return np.exp(1j * theta).astype(dtype)

    def entity(self, idx: np.ndarray) -> np.ndarray:
        """Look up entity embeddings by index."""
        return self.params['entities'][idx]

    def relation(self, idx: np.ndarray) -> np.ndarray:
        """Look up relation embeddings by index."""
        return self.params['relations'][idx]

    def score_rotate(
        self,
        h: np.ndarray,
        r: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        """RotatE-style score: Re(sum(h * r * conj(t))).

        Rotate the head by the relation, then measure phase alignment
        with the tail via complex inner product. This is interference:
        aligned phases add constructively, misaligned ones cancel.

        Args:
            h: Complex array [..., dim] -- head embeddings.
            r: Complex array [..., dim] -- relation embeddings.
            t: Complex array [..., dim] -- tail embeddings.

        Returns:
            Real array [...] -- scores (higher = better match).
        """
        return np.real(np.sum(h * r * np.conj(t), axis=-1))

    def score_transe(
        self,
        h: np.ndarray,
        r: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        """TransE-style score: -||h + r - t|| in complex space.

        Translation model: the relation maps head to tail by addition.
        The norm is the full complex L2 norm (sqrt of sum of |z_i|^2)).

        Args:
            h, r, t: Complex arrays [..., dim].

        Returns:
            Real array [...] -- scores (higher = better, i.e. closer to 0).
        """
        diff = h + r - t
        return -np.sqrt(np.sum(np.abs(diff) ** 2, axis=-1) + 1e-12)

    def score(
        self,
        h_idx: np.ndarray,
        r_idx: np.ndarray,
        t_idx: np.ndarray,
        mode: str = 'rotate',
    ) -> np.ndarray:
        """Score triples by index.

        Args:
            h_idx, r_idx, t_idx: Integer arrays of entity/relation indices.
            mode: 'rotate' or 'transe'.

        Returns:
            Real array of scores.
        """
        h = self.entity(h_idx)
        r = self.relation(r_idx)
        t = self.entity(t_idx)
        if mode == 'rotate':
            return self.score_rotate(h, r, t)
        elif mode == 'transe':
            return self.score_transe(h, r, t)
        else:
            raise ValueError(f"Unknown scoring mode: {mode!r}")


# ---------------------------------------------------------------------------
# PAMGraphMemory
# ---------------------------------------------------------------------------

class PAMGraphMemory:
    """Phase-Associative Memory for knowledge graph storage and retrieval.

    The entire knowledge graph is encoded into a single complex matrix S.
    Each triple (h, r, t) is stored as a rank-1 outer product:

        S += (h * r) outer conj(t)

    This is the PAM write operation: the key is h*r (head rotated by
    relation) and the value is t (tail entity).

    Retrieval is a matrix-vector multiply:

        y = S @ (h * r)

    The result y is a superposition of all stored tail entities, weighted
    by their phase alignment with the query h*r. Tails whose encoding
    phases align with the query contribute constructively (large |y_i|);
    misaligned ones interfere destructively (small |y_i|). This is
    exactly the same mechanism as holographic reduced representations
    and Hopfield networks, but in native complex space.

    Multi-hop reasoning is trivial: to answer "what is the capital of
    the country where French is spoken?", compose the relation rotations
    r_speaks_inverse * r_capital_of and query once. Phase composition
    handles the chaining automatically.

    Args:
        dim:    Complex embedding dimension.
        n_heads: Number of independent memory heads (capacity multiplier).
        d_head: Dimension per head (default: dim // n_heads).
        gamma:  Decay factor for state updates (0 < gamma <= 1).
                Controls how much old memories fade when new ones arrive.
        dtype:  np.complex128 or np.complex64.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 1,
        d_head: Optional[int] = None,
        gamma: float = 1.0,
        dtype: np.dtype = np.complex128,
    ):
        self.dim = dim
        self.n_heads = n_heads
        self.d_head = d_head or dim
        self.gamma = gamma
        self.dtype = dtype

        # State matrices: one per head, each d_head x d_head
        self.S = np.zeros(
            (n_heads, self.d_head, self.d_head), dtype=dtype
        )
        self.n_stored = 0

    def _split_heads(self, z: np.ndarray) -> np.ndarray:
        """Split a [dim] vector into [n_heads, d_head] for multi-head storage.

        If dim != n_heads * d_head, we tile/truncate to fit.
        """
        total = self.n_heads * self.d_head
        if z.shape[-1] == total:
            return z.reshape(*z.shape[:-1], self.n_heads, self.d_head)
        elif z.shape[-1] < total:
            # Tile to fill
            reps = int(np.ceil(total / z.shape[-1]))
            z_tiled = np.tile(z, (*([1] * (z.ndim - 1)), reps))
            return z_tiled[..., :total].reshape(
                *z.shape[:-1], self.n_heads, self.d_head
            )
        else:
            return z[..., :total].reshape(
                *z.shape[:-1], self.n_heads, self.d_head
            )

    def add_triple(
        self,
        h: np.ndarray,
        r: np.ndarray,
        t: np.ndarray,
    ) -> None:
        """Store a triple (h, r, t) into the PAM state.

        The key is h*r (head rotated by relation), value is t (tail).
        Update rule: S = gamma * S + (h*r) outer conj(t)

        Args:
            h: Complex array [dim] -- head entity embedding.
            r: Complex array [dim] -- relation embedding.
            t: Complex array [dim] -- tail entity embedding.
        """
        key = h * r  # [dim]
        key_heads = self._split_heads(key)  # [n_heads, d_head]
        val_heads = self._split_heads(t)     # [n_heads, d_head]

        # Rank-1 outer product per head: [n_heads, d_head, 1] * [n_heads, 1, d_head]
        outer = val_heads[:, :, None] * np.conj(key_heads[:, None, :])

        self.S = self.gamma * self.S + outer
        self.n_stored += 1

    def query(
        self,
        h: np.ndarray,
        r: np.ndarray,
    ) -> np.ndarray:
        """Retrieve from memory: y = S @ (h * r).

        The result is a superposition of all stored tail entities,
        weighted by phase alignment with the query.

        Args:
            h: Complex array [dim] -- head entity embedding.
            r: Complex array [dim] -- relation embedding.

        Returns:
            Complex array [n_heads, d_head] -- retrieved superposition.
        """
        key = h * r
        key_heads = self._split_heads(key)  # [n_heads, d_head]

        # Matrix-vector multiply per head: S @ key
        # S: [n_heads, d_head, d_head], key: [n_heads, d_head]
        result = np.einsum('hij,hj->hi', self.S, key_heads)
        return result

    def multi_hop(
        self,
        h: np.ndarray,
        relations: List[np.ndarray],
    ) -> np.ndarray:
        """Multi-hop reasoning by composing relation rotations.

        To answer "Paris -> capital_of -> ? -> located_in -> ?",
        compose r_capital_of * r_located_in and query once. Phase
        composition handles the chaining: angles just add.

        Args:
            h: Complex array [dim] -- starting entity embedding.
            relations: List of complex arrays [dim] -- relation sequence.

        Returns:
            Complex array [n_heads, d_head] -- retrieved result.
        """
        composed_r = np.ones(h.shape[-1], dtype=self.dtype)
        for r in relations:
            composed_r = composed_r * r
        return self.query(h, composed_r)

    def reset(self) -> None:
        """Clear the memory state."""
        self.S[:] = 0
        self.n_stored = 0


# ---------------------------------------------------------------------------
# ComplexKG -- Full model
# ---------------------------------------------------------------------------

class ComplexKG:
    """Complex-valued Knowledge Graph: embeddings + PAM memory.

    Combines ComplexKGEmbedding for learnable entity/relation vectors
    with PAMGraphMemory for holographic graph storage and interference-
    based retrieval.

    Training uses contrastive margin loss: positive triples should score
    higher than negative (corrupted) triples by at least a margin.

    Args:
        n_entities:  Number of entities.
        n_relations: Number of relation types.
        dim:         Complex embedding dimension.
        n_heads:     Number of PAM memory heads.
        d_head:      Dimension per PAM head (default: dim).
        margin:      Contrastive margin.
        score_mode:  'rotate' or 'transe'.
        gamma:       PAM decay factor.
        dtype:       np.complex128 or np.complex64.
    """

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        dim: int,
        n_heads: int = 4,
        d_head: Optional[int] = None,
        margin: float = 1.0,
        score_mode: str = 'rotate',
        gamma: float = 1.0,
        dtype: np.dtype = np.complex128,
    ):
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.dim = dim
        self.margin = margin
        self.score_mode = score_mode
        self.dtype = dtype

        self.embeddings = ComplexKGEmbedding(
            n_entities, n_relations, dim, dtype=dtype
        )
        self.memory = PAMGraphMemory(
            dim, n_heads=n_heads,
            d_head=d_head or dim,
            gamma=gamma, dtype=dtype,
        )

    @property
    def params(self) -> Dict[str, np.ndarray]:
        """All learnable parameters."""
        return self.embeddings.params

    def _score_fn(self, h, r, t):
        """Dispatch to the configured scoring function."""
        if self.score_mode == 'rotate':
            return self.embeddings.score_rotate(h, r, t)
        else:
            return self.embeddings.score_transe(h, r, t)

    def train_step(
        self,
        triples: np.ndarray,
        neg_samples: int = 10,
        lr: float = 0.01,
    ) -> float:
        """One training step with contrastive margin loss.

        For each positive triple, generate neg_samples corrupted triples
        (replace tail with random entity). Loss = max(0, margin + neg - pos).

        Uses finite-difference gradient estimation on the embedding arrays.

        Args:
            triples:     Array of shape [N, 3] with (h_idx, r_idx, t_idx).
            neg_samples: Number of negative samples per positive.
            lr:          Learning rate for this step.

        Returns:
            Scalar loss value.
        """
        h_idx = triples[:, 0]
        r_idx = triples[:, 1]
        t_idx = triples[:, 2]

        h = self.embeddings.entity(h_idx)
        r = self.embeddings.relation(r_idx)
        t = self.embeddings.entity(t_idx)

        # Positive scores
        pos_scores = self._score_fn(h, r, t)  # [N]

        # Negative samples: corrupt tails
        N = len(triples)
        neg_t_idx = np.random.randint(0, self.n_entities, size=(N, neg_samples))
        neg_t = self.embeddings.entity(neg_t_idx)  # [N, neg_samples, dim]

        # Expand h, r for broadcasting: [N, 1, dim]
        h_exp = h[:, None, :]
        r_exp = r[:, None, :]

        neg_scores = self._score_fn(h_exp, r_exp, neg_t)  # [N, neg_samples]

        # Margin loss: max(0, margin + neg_score - pos_score)
        losses = np.maximum(0.0, self.margin + neg_scores - pos_scores[:, None])
        loss = losses.mean()

        # Gradient: d(loss)/d(entity) via analytical complex gradient
        # For RotatE: score = Re(sum(h * r * conj(t)))
        #   d_score/d_h = r * conj(t)  (Wirtinger derivative)
        #   d_score/d_t = conj(h * r)
        # For margin loss: gradient flows through positive and negative terms
        mask = (losses > 0).astype(np.float64)  # [N, neg_samples]
        n_active = mask.sum() + 1e-10

        # Entity gradients
        ent_grad = np.zeros_like(self.embeddings.params['entities'])
        rel_grad = np.zeros_like(self.embeddings.params['relations'])

        if self.score_mode == 'rotate':
            # Positive gradient (we want to increase pos_score)
            # d_pos/d_h = r * conj(t), d_pos/d_t = conj(h * r)
            pos_mask = (mask.sum(axis=1) > 0).astype(np.float64)  # [N]

            d_h_pos = r * np.conj(t)  # [N, dim]
            d_t_pos = np.conj(h * r)  # [N, dim]
            d_r_pos = h * np.conj(t)  # [N, dim]

            # Accumulate positive gradients (increase score = decrease loss)
            for i in range(N):
                if pos_mask[i] > 0:
                    ent_grad[h_idx[i]] += d_h_pos[i] * pos_mask[i] / n_active
                    ent_grad[t_idx[i]] += d_t_pos[i] * pos_mask[i] / n_active
                    rel_grad[r_idx[i]] += d_r_pos[i] * pos_mask[i] / n_active

            # Negative gradients (we want to decrease neg_score)
            d_h_neg = r_exp * np.conj(neg_t)  # [N, neg, dim]
            d_t_neg_base = np.conj(h_exp * r_exp)  # [N, 1, dim]
            d_t_neg = np.broadcast_to(d_t_neg_base, neg_t.shape).copy()  # [N, neg, dim]

            for i in range(N):
                for j in range(neg_samples):
                    if mask[i, j] > 0:
                        ent_grad[h_idx[i]] -= d_h_neg[i, j] / n_active
                        ent_grad[neg_t_idx[i, j]] -= d_t_neg[i, j] / n_active
                        rel_grad[r_idx[i]] -= (
                            h[i] * np.conj(neg_t[i, j])
                        ) / n_active
        else:
            # TransE: score = -||h + r - t||
            # d_score/d_h = -(h + r - t) / ||h + r - t||
            diff_pos = h + r - t  # [N, dim]
            norm_pos = np.sqrt(np.sum(np.abs(diff_pos) ** 2, axis=-1, keepdims=True) + 1e-12)
            d_pos = -diff_pos / norm_pos  # [N, dim]

            pos_mask = (mask.sum(axis=1) > 0).astype(np.float64)

            for i in range(N):
                if pos_mask[i] > 0:
                    # Positive: increase score = push h+r toward t
                    ent_grad[h_idx[i]] += d_pos[i] * pos_mask[i] / n_active
                    ent_grad[t_idx[i]] -= d_pos[i] * pos_mask[i] / n_active
                    rel_grad[r_idx[i]] += d_pos[i] * pos_mask[i] / n_active

            diff_neg = h_exp + r_exp - neg_t  # [N, neg, dim]
            norm_neg = np.sqrt(np.sum(np.abs(diff_neg) ** 2, axis=-1, keepdims=True) + 1e-12)
            d_neg = -diff_neg / norm_neg

            for i in range(N):
                for j in range(neg_samples):
                    if mask[i, j] > 0:
                        ent_grad[h_idx[i]] -= d_neg[i, j] / n_active
                        ent_grad[neg_t_idx[i, j]] += d_neg[i, j] / n_active
                        rel_grad[r_idx[i]] -= d_neg[i, j] / n_active

        # Apply gradients
        self.embeddings.params['entities'] += lr * ent_grad
        self.embeddings.params['relations'] += lr * rel_grad

        # Re-normalize relations to unit circle (project back)
        rel = self.embeddings.params['relations']
        self.embeddings.params['relations'] = rel / (np.abs(rel) + 1e-12)

        return loss

    def encode_graph(self, triples: np.ndarray) -> None:
        """Load all triples into PAM memory.

        Args:
            triples: Array [N, 3] of (h_idx, r_idx, t_idx).
        """
        self.memory.reset()
        for i in range(len(triples)):
            h = self.embeddings.entity(triples[i, 0])
            r = self.embeddings.relation(triples[i, 1])
            t = self.embeddings.entity(triples[i, 2])
            self.memory.add_triple(h, r, t)

    def search(
        self,
        entity_idx: int,
        relation_idx: int,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query PAM memory: given (entity, relation), find top-k tail entities.

        The retrieval is interference-based: the query h*r is projected
        through the state matrix, producing a superposition. We score
        each entity by its complex inner product with the retrieval result.

        Args:
            entity_idx:   Head entity index.
            relation_idx: Relation index.
            top_k:        Number of results to return.

        Returns:
            Tuple of (entity_indices, scores) arrays.
        """
        h = self.embeddings.entity(entity_idx)
        r = self.embeddings.relation(relation_idx)

        # Retrieve from PAM
        retrieved = self.memory.query(h, r)  # [n_heads, d_head]

        # Score all entities against the retrieved superposition
        all_ents = self.embeddings.params['entities']  # [n_entities, dim]
        ent_heads = self.memory._split_heads(all_ents)  # [n_entities, n_heads, d_head]

        # Interference score: Re(sum(retrieved * conj(entity)))
        # retrieved: [n_heads, d_head], ent_heads: [n_entities, n_heads, d_head]
        scores = np.real(
            np.sum(retrieved[None, :, :] * np.conj(ent_heads), axis=(-2, -1))
        )  # [n_entities]

        # Top-k
        top_idx = np.argsort(scores)[::-1][:top_k]
        return top_idx, scores[top_idx]

    def multi_hop_search(
        self,
        entity_idx: int,
        relation_path: List[int],
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-hop search: compose relation rotations and query once.

        Args:
            entity_idx:    Starting entity index.
            relation_path: List of relation indices to compose.
            top_k:         Number of results to return.

        Returns:
            Tuple of (entity_indices, scores) arrays.
        """
        h = self.embeddings.entity(entity_idx)
        relations = [self.embeddings.relation(r) for r in relation_path]

        retrieved = self.memory.multi_hop(h, relations)

        all_ents = self.embeddings.params['entities']
        ent_heads = self.memory._split_heads(all_ents)

        scores = np.real(
            np.sum(retrieved[None, :, :] * np.conj(ent_heads), axis=(-2, -1))
        )

        top_idx = np.argsort(scores)[::-1][:top_k]
        return top_idx, scores[top_idx]

    def phase_analysis(self) -> Dict[str, np.ndarray]:
        """Analyze the phase structure of entity and relation embeddings.

        Returns a dict with:
          - entity_phases: [n_entities, dim] angles in [-pi, pi]
          - entity_magnitudes: [n_entities, dim] absolute values
          - relation_phases: [n_relations, dim] angles
          - relation_magnitudes: [n_relations, dim] (should be ~1.0)
          - entity_phase_mean: [n_entities] mean phase per entity
          - entity_phase_std: [n_entities] phase spread per entity
          - relation_phase_mean: [n_relations] mean phase per relation
          - pam_state_magnitude: [n_heads] Frobenius norm of each state head
        """
        ents = self.embeddings.params['entities']
        rels = self.embeddings.params['relations']

        ent_phases = np.angle(ents)
        ent_mags = np.abs(ents)
        rel_phases = np.angle(rels)
        rel_mags = np.abs(rels)

        return {
            'entity_phases': ent_phases,
            'entity_magnitudes': ent_mags,
            'relation_phases': rel_phases,
            'relation_magnitudes': rel_mags,
            'entity_phase_mean': np.mean(ent_phases, axis=-1),
            'entity_phase_std': np.std(ent_phases, axis=-1),
            'relation_phase_mean': np.mean(rel_phases, axis=-1),
            'pam_state_magnitude': np.array([
                np.sqrt(np.sum(np.abs(self.memory.S[i]) ** 2))
                for i in range(self.memory.n_heads)
            ]),
        }

    def compare_embeddings(
        self,
        e1_idx: int,
        e2_idx: int,
    ) -> Dict[str, float]:
        """Compare two entity embeddings via interference.

        Returns:
            Dict with interference_score, phase_distance, magnitude_ratio.
        """
        e1 = self.embeddings.entity(e1_idx)
        e2 = self.embeddings.entity(e2_idx)

        # Interference score: Re(sum(e1 * conj(e2))) / (|e1| * |e2|)
        raw = np.real(np.sum(e1 * np.conj(e2)))
        norm = np.sqrt(np.sum(np.abs(e1) ** 2) * np.sum(np.abs(e2) ** 2)) + 1e-12
        interference = raw / norm

        # Phase distance: mean angular difference
        phase_diff = np.angle(e1) - np.angle(e2)
        # Wrap to [-pi, pi]
        phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
        phase_dist = np.mean(np.abs(phase_diff))

        # Magnitude ratio
        m1 = np.sqrt(np.sum(np.abs(e1) ** 2))
        m2 = np.sqrt(np.sum(np.abs(e2) ** 2))
        mag_ratio = float(m1 / (m2 + 1e-12))

        return {
            'interference_score': float(interference),
            'phase_distance': float(phase_dist),
            'magnitude_ratio': mag_ratio,
        }


# ---------------------------------------------------------------------------
# RealKGBaseline
# ---------------------------------------------------------------------------

class RealKGBaseline:
    """Real-valued TransE knowledge graph baseline for comparison.

    Same architecture and training loop as ComplexKG, but everything is
    float64 instead of complex128. This isolates the contribution of
    complex-valued phase structure.

    TransE scoring: score(h, r, t) = -||h + r - t||_2

    Args:
        n_entities:  Number of entities.
        n_relations: Number of relation types.
        dim:         Embedding dimension (real).
        margin:      Contrastive margin.
    """

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        dim: int,
        margin: float = 1.0,
    ):
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.dim = dim
        self.margin = margin

        self.params: Dict[str, np.ndarray] = {
            'entities': np.random.randn(n_entities, dim).astype(np.float64) * 0.1,
            'relations': np.random.randn(n_relations, dim).astype(np.float64) * 0.1,
        }
        # Normalize relations to unit length
        norms = np.linalg.norm(self.params['relations'], axis=-1, keepdims=True)
        self.params['relations'] /= (norms + 1e-12)

    def entity(self, idx):
        return self.params['entities'][idx]

    def relation(self, idx):
        return self.params['relations'][idx]

    def score(self, h, r, t):
        """TransE score: -||h + r - t||."""
        diff = h + r - t
        return -np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)

    def train_step(
        self,
        triples: np.ndarray,
        neg_samples: int = 10,
        lr: float = 0.01,
    ) -> float:
        """One training step with margin loss and analytical gradients."""
        h_idx = triples[:, 0]
        r_idx = triples[:, 1]
        t_idx = triples[:, 2]
        N = len(triples)

        h = self.entity(h_idx)
        r = self.relation(r_idx)
        t = self.entity(t_idx)

        pos_scores = self.score(h, r, t)

        neg_t_idx = np.random.randint(0, self.n_entities, size=(N, neg_samples))
        neg_t = self.entity(neg_t_idx)

        h_exp = h[:, None, :]
        r_exp = r[:, None, :]

        neg_scores = self.score(h_exp, r_exp, neg_t)

        losses = np.maximum(0.0, self.margin + neg_scores - pos_scores[:, None])
        loss = losses.mean()

        mask = (losses > 0).astype(np.float64)
        n_active = mask.sum() + 1e-10

        ent_grad = np.zeros_like(self.params['entities'])
        rel_grad = np.zeros_like(self.params['relations'])

        # TransE gradients
        diff_pos = h + r - t
        norm_pos = np.sqrt(np.sum(diff_pos ** 2, axis=-1, keepdims=True) + 1e-12)
        d_pos = -diff_pos / norm_pos

        pos_mask = (mask.sum(axis=1) > 0).astype(np.float64)

        for i in range(N):
            if pos_mask[i] > 0:
                ent_grad[h_idx[i]] += d_pos[i] * pos_mask[i] / n_active
                ent_grad[t_idx[i]] -= d_pos[i] * pos_mask[i] / n_active
                rel_grad[r_idx[i]] += d_pos[i] * pos_mask[i] / n_active

        diff_neg = h_exp + r_exp - neg_t
        norm_neg = np.sqrt(np.sum(diff_neg ** 2, axis=-1, keepdims=True) + 1e-12)
        d_neg = -diff_neg / norm_neg

        for i in range(N):
            for j in range(neg_samples):
                if mask[i, j] > 0:
                    ent_grad[h_idx[i]] -= d_neg[i, j] / n_active
                    ent_grad[neg_t_idx[i, j]] += d_neg[i, j] / n_active
                    rel_grad[r_idx[i]] -= d_neg[i, j] / n_active

        self.params['entities'] += lr * ent_grad
        self.params['relations'] += lr * rel_grad

        # Re-normalize relations
        norms = np.linalg.norm(self.params['relations'], axis=-1, keepdims=True)
        self.params['relations'] /= (norms + 1e-12)

        return loss

    def rank_entities(
        self,
        h_idx: int,
        r_idx: int,
    ) -> np.ndarray:
        """Score all entities as potential tails for (h, r, ?).

        Returns array of scores [n_entities].
        """
        h = self.entity(h_idx)
        r = self.relation(r_idx)
        all_t = self.params['entities']
        return self.score(h[None, :], r[None, :], all_t)


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def _rank_for_triple(
    model,
    h_idx: int,
    r_idx: int,
    t_idx: int,
) -> int:
    """Compute the rank of the correct tail entity among all entities.

    Works for both ComplexKG and RealKGBaseline.
    """
    if isinstance(model, ComplexKG):
        h = model.embeddings.entity(h_idx)
        r = model.embeddings.relation(r_idx)
        all_t = model.embeddings.params['entities']

        if model.score_mode == 'rotate':
            scores = model.embeddings.score_rotate(
                h[None, :], r[None, :], all_t
            )
        else:
            scores = model.embeddings.score_transe(
                h[None, :], r[None, :], all_t
            )
    elif isinstance(model, RealKGBaseline):
        scores = model.rank_entities(h_idx, r_idx)
    else:
        raise TypeError(f"Unknown model type: {type(model)}")

    # Rank: how many entities score higher than the correct one?
    correct_score = scores[t_idx]
    rank = int(np.sum(scores > correct_score)) + 1
    return rank


def hits_at_k(
    model,
    test_triples: np.ndarray,
    k: int = 10,
) -> float:
    """Fraction of test triples where the correct tail is in the top-k.

    Args:
        model:        ComplexKG or RealKGBaseline.
        test_triples: Array [N, 3] of (h_idx, r_idx, t_idx).
        k:            Cutoff for top-k.

    Returns:
        Hits@k as a float in [0, 1].
    """
    hits = 0
    for i in range(len(test_triples)):
        rank = _rank_for_triple(
            model,
            int(test_triples[i, 0]),
            int(test_triples[i, 1]),
            int(test_triples[i, 2]),
        )
        if rank <= k:
            hits += 1
    return hits / len(test_triples)


def mean_rank(
    model,
    test_triples: np.ndarray,
) -> float:
    """Average rank of the correct tail entity across test triples.

    Args:
        model:        ComplexKG or RealKGBaseline.
        test_triples: Array [N, 3] of (h_idx, r_idx, t_idx).

    Returns:
        Mean rank (lower is better; 1 is perfect).
    """
    ranks = []
    for i in range(len(test_triples)):
        rank = _rank_for_triple(
            model,
            int(test_triples[i, 0]),
            int(test_triples[i, 1]),
            int(test_triples[i, 2]),
        )
        ranks.append(rank)
    return float(np.mean(ranks))


def mrr(
    model,
    test_triples: np.ndarray,
) -> float:
    """Mean Reciprocal Rank across test triples.

    MRR = mean(1/rank). Higher is better; 1.0 is perfect.

    Args:
        model:        ComplexKG or RealKGBaseline.
        test_triples: Array [N, 3] of (h_idx, r_idx, t_idx).

    Returns:
        MRR as a float in (0, 1].
    """
    rr = []
    for i in range(len(test_triples)):
        rank = _rank_for_triple(
            model,
            int(test_triples[i, 0]),
            int(test_triples[i, 1]),
            int(test_triples[i, 2]),
        )
        rr.append(1.0 / rank)
    return float(np.mean(rr))


def compare_to_baseline(
    complex_model: ComplexKG,
    real_model: RealKGBaseline,
    test_triples: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Side-by-side evaluation of complex vs real models.

    Returns:
        Dict with 'complex' and 'real' sub-dicts, each containing
        hits_at_1, hits_at_3, hits_at_10, mrr, mean_rank.
    """
    results = {}
    for name, model in [('complex', complex_model), ('real', real_model)]:
        results[name] = {
            'hits_at_1': hits_at_k(model, test_triples, k=1),
            'hits_at_3': hits_at_k(model, test_triples, k=3),
            'hits_at_10': hits_at_k(model, test_triples, k=10),
            'mrr': mrr(model, test_triples),
            'mean_rank': mean_rank(model, test_triples),
        }
    return results
