"""Semantic trajectory analysis for LLM output streams.

Tracks semantic hops, attractors, winding numbers, and Berry phase
across generated text trajectories in embedding space.
"""

import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np


@dataclass
class SemanticHop:
    source_text: str
    target_text: str
    source_embedding: np.ndarray
    target_embedding: np.ndarray
    hop_distance: float
    source_language: str
    target_language: str
    bridge_strength: float
    position: int


@dataclass
class SemanticTrajectory:
    chunks: List[str]
    embeddings: np.ndarray
    positions: List[int]
    languages: List[str]
    hops: List[SemanticHop]
    attractors: List[Dict]
    orbital_period: Optional[float]
    berry_phase: Optional[float]
    winding_number: int


@dataclass
class WanderingConfig:
    model: str
    provider: str
    temperature: float
    prompt: str
    chunk_size: int = 3
    min_hop_distance: float = 0.3
    attractor_eps: float = 0.15
    attractor_min_samples: int = 2


def detect_language(text: str) -> str:
    """Detect the dominant script/language of a text string."""
    script_counts = defaultdict(int)
    for char in text:
        if char.isspace() or char in ".,;:!?\"'()-[]{}0123456789":
            continue
        try:
            name = unicodedata.name(char, "")
            for script, keyword in [
                ("Latin", "LATIN"), ("Cyrillic", "CYRILLIC"), ("Arabic", "ARABIC"),
                ("CJK", "CJK"), ("Hangul", "HANGUL"), ("Hebrew", "HEBREW"),
                ("Georgian", "GEORGIAN"), ("Thai", "THAI"),
                ("Devanagari", "DEVANAGARI"), ("Greek", "GREEK"),
            ]:
                if keyword in name:
                    script_counts[script] += 1
                    break
            else:
                if "HIRAGANA" in name or "KATAKANA" in name:
                    script_counts["Japanese"] += 1
                else:
                    script_counts["Other"] += 1
        except Exception:
            script_counts["Unknown"] += 1

    if not script_counts:
        return "Unknown"
    return max(script_counts, key=script_counts.get)


class SemanticWanderingAnalyzer:
    """Analyze semantic trajectories in LLM-generated text.

    Requires sentence-transformers, sklearn, and scipy as optional
    dependencies.
    """

    def __init__(self, embedder_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(embedder_name)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()

    def chunk_text(self, text: str, chunk_size: int = 3) -> List[Tuple[str, int]]:
        """Split text into word-level chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append((chunk, i))
        return chunks

    def compute_trajectory(
        self, text: str, prompt: str, chunk_size: int = 3,
    ) -> Tuple[List[str], np.ndarray, List[int], List[str]]:
        """Compute the embedding trajectory of chunked text.

        Returns (chunks, embeddings, positions, languages).
        """
        chunked = self.chunk_text(text, chunk_size)
        if not chunked:
            return [], np.array([]), [], []
        chunks = [c[0] for c in chunked]
        positions = [c[1] for c in chunked]
        embeddings = self.embedder.encode(chunks)
        languages = [detect_language(c) for c in chunks]
        return chunks, embeddings, positions, languages

    def detect_hops(
        self,
        chunks: List[str],
        embeddings: np.ndarray,
        positions: List[int],
        languages: List[str],
        min_distance: float = 0.3,
    ) -> List[SemanticHop]:
        """Detect semantic hops (large jumps or language switches) in the trajectory."""
        from scipy.spatial.distance import cosine

        hops = []
        for i in range(len(embeddings) - 1):
            dist = cosine(embeddings[i], embeddings[i + 1])
            is_lang_switch = languages[i] != languages[i + 1]
            is_semantic_jump = dist > min_distance

            if is_lang_switch or is_semantic_jump:
                bridge = self.compute_bridge_strength(
                    chunks[i], chunks[i + 1], embeddings[i], embeddings[i + 1]
                )
                hops.append(SemanticHop(
                    source_text=chunks[i],
                    target_text=chunks[i + 1],
                    source_embedding=embeddings[i],
                    target_embedding=embeddings[i + 1],
                    hop_distance=dist,
                    source_language=languages[i],
                    target_language=languages[i + 1],
                    bridge_strength=bridge,
                    position=positions[i + 1],
                ))
        return hops

    def compute_bridge_strength(
        self,
        source: str,
        target: str,
        source_emb: np.ndarray,
        target_emb: np.ndarray,
    ) -> float:
        """Compute semantic bridge strength between two chunks.

        Checks whether the midpoint embedding bridges the two.
        """
        combined = f"{source} {target}"
        combined_emb = self.embedder.encode(combined)
        from scipy.spatial.distance import cosine
        mid_emb = (source_emb + target_emb) / 2
        bridge_dist = cosine(combined_emb, mid_emb)
        return max(0.0, 1.0 - bridge_dist)

    def find_attractors(
        self, embeddings: np.ndarray, eps: float = 0.15, min_samples: int = 2,
    ) -> List[Dict]:
        """Find semantic attractors using DBSCAN clustering."""
        from sklearn.cluster import DBSCAN
        from sklearn.decomposition import PCA

        if len(embeddings) < min_samples:
            return []

        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = clustering.fit_predict(embeddings)

        attractors = []
        for label in set(labels):
            if label == -1:
                continue
            mask = labels == label
            cluster_embs = embeddings[mask]
            centroid = cluster_embs.mean(axis=0)
            visit_indices = np.where(mask)[0].tolist()
            attractors.append({
                "label": int(label),
                "centroid": centroid,
                "size": int(mask.sum()),
                "visit_indices": visit_indices,
                "radius": float(np.max([
                    1 - np.dot(e, centroid) / (np.linalg.norm(e) * np.linalg.norm(centroid))
                    for e in cluster_embs
                ])),
            })
        return attractors

    def compute_winding_number(self, embeddings: np.ndarray) -> int:
        """Compute the winding number around the trajectory centroid in 2D."""
        from sklearn.decomposition import PCA

        if len(embeddings) < 3:
            return 0
        pca = PCA(n_components=2)
        projected = pca.fit_transform(embeddings)
        centroid = projected.mean(axis=0)
        centered = projected - centroid

        angles = np.arctan2(centered[:, 1], centered[:, 0])
        diffs = np.diff(angles)
        diffs = np.where(diffs > np.pi, diffs - 2 * np.pi, diffs)
        diffs = np.where(diffs < -np.pi, diffs + 2 * np.pi, diffs)
        return int(np.round(np.sum(diffs) / (2 * np.pi)))

    def compute_berry_phase(self, embeddings: np.ndarray) -> Optional[float]:
        """Compute the Berry (geometric) phase of the trajectory."""
        if len(embeddings) < 3:
            return None
        total_phase = 0.0
        for i in range(len(embeddings) - 1):
            dot = np.dot(embeddings[i], embeddings[i + 1])
            norms = np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            if norms > 0:
                cos_angle = np.clip(dot / norms, -1, 1)
                total_phase += np.arccos(cos_angle)
        return float(total_phase)

    def analyze(self, text: str, config: WanderingConfig) -> SemanticTrajectory:
        """Full analysis pipeline: trajectory -> hops -> attractors -> topology.

        Parameters
        ----------
        text : str
            The generated text to analyze.
        config : WanderingConfig
            Analysis parameters.

        Returns
        -------
        SemanticTrajectory with all computed fields.
        """
        chunks, embeddings, positions, languages = self.compute_trajectory(
            text, config.prompt, config.chunk_size
        )
        if len(embeddings) == 0:
            return SemanticTrajectory(
                chunks=[], embeddings=np.array([]), positions=[], languages=[],
                hops=[], attractors=[], orbital_period=None,
                berry_phase=None, winding_number=0,
            )

        hops = self.detect_hops(chunks, embeddings, positions, languages, config.min_hop_distance)
        attractors = self.find_attractors(embeddings, config.attractor_eps, config.attractor_min_samples)
        winding = self.compute_winding_number(embeddings)
        berry = self.compute_berry_phase(embeddings)

        # Estimate orbital period from attractor revisits
        orbital_period = None
        if attractors:
            periods = []
            for att in attractors:
                visits = att["visit_indices"]
                if len(visits) >= 2:
                    diffs = [visits[i + 1] - visits[i] for i in range(len(visits) - 1)]
                    periods.extend(diffs)
            if periods:
                orbital_period = float(np.median(periods))

        return SemanticTrajectory(
            chunks=chunks,
            embeddings=embeddings,
            positions=positions,
            languages=languages,
            hops=hops,
            attractors=attractors,
            orbital_period=orbital_period,
            berry_phase=berry,
            winding_number=winding,
        )
