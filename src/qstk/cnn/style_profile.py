#!/usr/bin/env python3
"""Extract an operator-space style profile from a corpus of text.

Given a set of text files (e.g., Giacomo's papers), this script:
1. Tokenizes each text using GPT-2's tokenizer
2. Embeds tokens via GPT-2's wte layer
3. Projects embeddings to complex space (Hilbert transform)
4. Extracts transition operators between consecutive tokens
5. Clusters operators into a finite set of types
6. For each cluster, records which tokens are reachable
7. Computes aggregate statistics (phase diversity, rotation, coherence)
8. Saves the full profile as a .npz + .json for use in constrained decoding

Usage:
    python style_profile.py --corpus_dir /path/to/tex/files --output profile.npz
    python style_profile.py --hf_dataset /path/to/hf_dataset --split style_target --output profile.npz
"""

import argparse
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from qstk.cnn.probe import embed_to_complex
from qstk.cnn.operators import (
    extract_operators,
    operator_diversity,
    operator_spectrum,
    trajectory_coherence,
)

warnings.filterwarnings('ignore')


def load_corpus_texts(corpus_dir: str, extensions: tuple = ('.tex', '.md', '.txt')) -> list[str]:
    """Load all text files from a directory."""
    corpus_dir = Path(corpus_dir)
    texts = []
    for ext in extensions:
        for f in sorted(corpus_dir.rglob(f'*{ext}')):
            text = f.read_text(encoding='utf-8', errors='replace')
            if len(text.split()) > 50:
                texts.append(text)
    return texts


def load_hf_texts(dataset_path: str, split: str = 'style_target', text_col: str = 'text') -> list[str]:
    """Load texts from a HuggingFace dataset on disk."""
    from datasets import load_from_disk
    ds = load_from_disk(dataset_path)
    return [row[text_col] for row in ds[split] if len(row[text_col].split()) > 50]


def extract_profile(
    texts: list[str],
    model_name: str = 'gpt2',
    max_tokens_per_text: int = 512,
    max_clusters: int = 16,
    device: str = 'auto',
) -> dict:
    """Extract the full operator-space style profile from a list of texts.

    Returns a dict with:
        operators: np.ndarray [N_total, d_complex] — all operators concatenated
        cluster_labels: np.ndarray [N_total] — cluster assignment per operator
        cluster_centroids: np.ndarray [k, 2*d_complex] — cos/sin phase centroids
        cluster_vocabs: dict[int, list[int]] — token ids reachable per cluster
        cluster_token_probs: dict[int, dict[int, float]] — per-cluster token probabilities
        diversity: dict — aggregate operator diversity stats
        spectrum: dict — PCA decomposition of operator basis
        coherence_stats: dict — mean trajectory coherence across all texts
        n_clusters: int
        full_entropy: float — entropy of the full token distribution
        cluster_entropies: dict[int, float] — per-cluster token entropy
        transition_matrix: np.ndarray [k, k] — P(cluster_t | cluster_{t-1})
        metadata: dict — model name, corpus size, etc.
    """
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size

    all_operators = []
    all_next_tokens = []
    all_coherences = []
    total_tokens_processed = 0

    print(f"Processing {len(texts)} texts...")
    for i, text in enumerate(texts):
        token_ids = tokenizer.encode(text)[:max_tokens_per_text]
        if len(token_ids) < 10:
            continue

        ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            embs = model.transformer.wte(ids_tensor).cpu().numpy().astype(np.float64)

        complex_embs = embed_to_complex(embs, method='hilbert')
        ops = extract_operators(complex_embs)

        all_operators.append(ops)
        for t in range(len(ops)):
            if t + 1 < len(token_ids):
                all_next_tokens.append(token_ids[t + 1])

        coh = trajectory_coherence(complex_embs)
        all_coherences.append(coh)
        total_tokens_processed += len(token_ids)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(texts)} texts, {total_tokens_processed} tokens")

    concat_ops = np.concatenate(all_operators, axis=0)
    next_tokens = np.array(all_next_tokens[:len(concat_ops)])
    print(f"Total operators: {len(concat_ops)}, dim: {concat_ops.shape[1]}")

    # Cluster operators by phase signature
    print("Clustering operators...")
    phases = np.angle(concat_ops)
    cos_sin = np.concatenate([np.cos(phases), np.sin(phases)], axis=1)

    best_k, best_score = 2, -1
    sample_size = min(5000, len(cos_sin))
    for k in range(2, max_clusters + 1):
        km = KMeans(n_clusters=k, n_init=5, random_state=42, max_iter=100)
        labels = km.fit_predict(cos_sin)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(cos_sin, labels, sample_size=sample_size)
        if score > best_score:
            best_score = score
            best_k = k

    print(f"Optimal clusters: k={best_k} (silhouette={best_score:.3f})")
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    cluster_labels = km.fit_predict(cos_sin)

    # Per-cluster vocabulary and token probabilities
    cluster_vocabs = {}
    cluster_token_probs = {}
    cluster_entropies = {}

    full_counts = Counter(next_tokens.tolist())
    full_probs = np.array(list(full_counts.values()), dtype=float)
    full_probs /= full_probs.sum()
    full_entropy = -float(np.sum(full_probs * np.log2(full_probs + 1e-30)))

    for c in range(best_k):
        mask = cluster_labels == c
        cluster_toks = next_tokens[mask]
        counts = Counter(cluster_toks.tolist())
        total = sum(counts.values())
        probs = {tid: cnt / total for tid, cnt in counts.items()}
        cluster_vocabs[c] = sorted(counts.keys())
        cluster_token_probs[c] = probs

        p = np.array(list(counts.values()), dtype=float) / total
        cluster_entropies[c] = -float(np.sum(p * np.log2(p + 1e-30)))

    # Transition matrix P(cluster_t | cluster_{t-1})
    transition_matrix = np.zeros((best_k, best_k), dtype=float)
    offset = 0
    for ops in all_operators:
        n = len(ops)
        labels_seq = cluster_labels[offset:offset + n]
        offset += n
        for t in range(1, len(labels_seq)):
            transition_matrix[labels_seq[t-1], labels_seq[t]] += 1

    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = transition_matrix / np.maximum(row_sums, 1)

    # Aggregate statistics
    diversity = operator_diversity(concat_ops)
    spectrum = operator_spectrum(concat_ops)

    mean_coherence = {}
    for key in all_coherences[0]:
        mean_coherence[key] = float(np.mean([c[key] for c in all_coherences]))

    return {
        'operators': concat_ops,
        'cluster_labels': cluster_labels,
        'cluster_centroids': km.cluster_centers_,
        'cluster_vocabs': cluster_vocabs,
        'cluster_token_probs': cluster_token_probs,
        'n_clusters': best_k,
        'diversity': diversity,
        'spectrum': {
            'eigenvalues': spectrum['eigenvalues'],
            'eigenvectors': spectrum['eigenvectors'],
            'explained_variance_ratio': spectrum['explained_variance_ratio'],
        },
        'coherence_stats': mean_coherence,
        'full_entropy': full_entropy,
        'cluster_entropies': cluster_entropies,
        'transition_matrix': transition_matrix,
        'metadata': {
            'model_name': model_name,
            'n_texts': len(texts),
            'total_tokens': total_tokens_processed,
            'total_operators': len(concat_ops),
            'silhouette_score': best_score,
            'vocab_size': vocab_size,
        },
    }


def save_profile(profile: dict, output_path: str):
    """Save profile to .npz (arrays) + .json (metadata/vocabs)."""
    output_path = Path(output_path)

    # Save arrays
    np.savez_compressed(
        output_path.with_suffix('.npz'),
        operators=profile['operators'],
        cluster_labels=profile['cluster_labels'],
        cluster_centroids=profile['cluster_centroids'],
        transition_matrix=profile['transition_matrix'],
        eigenvalues=profile['spectrum']['eigenvalues'],
        eigenvectors=profile['spectrum']['eigenvectors'],
        explained_variance_ratio=profile['spectrum']['explained_variance_ratio'],
    )

    # Save JSON-serializable parts
    json_data = {
        'cluster_vocabs': {str(k): v for k, v in profile['cluster_vocabs'].items()},
        'cluster_token_probs': {
            str(k): {str(tid): p for tid, p in v.items()}
            for k, v in profile['cluster_token_probs'].items()
        },
        'n_clusters': profile['n_clusters'],
        'diversity': profile['diversity'],
        'coherence_stats': profile['coherence_stats'],
        'full_entropy': profile['full_entropy'],
        'cluster_entropies': {str(k): v for k, v in profile['cluster_entropies'].items()},
        'metadata': profile['metadata'],
    }
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"Saved: {output_path.with_suffix('.npz')}")
    print(f"Saved: {output_path.with_suffix('.json')}")


def load_profile(path: str) -> dict:
    """Load a saved profile from .npz + .json."""
    path = Path(path)
    arrays = np.load(path.with_suffix('.npz'))
    with open(path.with_suffix('.json')) as f:
        meta = json.load(f)

    # Reconstruct cluster_vocabs with int keys
    cluster_vocabs = {int(k): v for k, v in meta['cluster_vocabs'].items()}
    cluster_token_probs = {
        int(k): {int(tid): p for tid, p in v.items()}
        for k, v in meta['cluster_token_probs'].items()
    }

    return {
        'operators': arrays['operators'],
        'cluster_labels': arrays['cluster_labels'],
        'cluster_centroids': arrays['cluster_centroids'],
        'transition_matrix': arrays['transition_matrix'],
        'spectrum': {
            'eigenvalues': arrays['eigenvalues'],
            'eigenvectors': arrays['eigenvectors'],
            'explained_variance_ratio': arrays['explained_variance_ratio'],
        },
        'cluster_vocabs': cluster_vocabs,
        'cluster_token_probs': cluster_token_probs,
        'n_clusters': meta['n_clusters'],
        'diversity': meta['diversity'],
        'coherence_stats': meta['coherence_stats'],
        'full_entropy': meta['full_entropy'],
        'cluster_entropies': {int(k): v for k, v in meta['cluster_entropies'].items()},
        'metadata': meta['metadata'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract operator-space style profile')
    parser.add_argument('--corpus_dir', type=str, help='Directory of .tex/.md/.txt files')
    parser.add_argument('--hf_dataset', type=str, help='Path to HF dataset on disk')
    parser.add_argument('--split', type=str, default='style_target')
    parser.add_argument('--output', type=str, default='style_profile')
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--max_clusters', type=int, default=16)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.corpus_dir:
        texts = load_corpus_texts(args.corpus_dir)
    elif args.hf_dataset:
        texts = load_hf_texts(args.hf_dataset, args.split)
    else:
        parser.error("Provide --corpus_dir or --hf_dataset")

    print(f"Loaded {len(texts)} texts")
    profile = extract_profile(
        texts,
        model_name=args.model,
        max_tokens_per_text=args.max_tokens,
        max_clusters=args.max_clusters,
        device=args.device,
    )

    save_profile(profile, args.output)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Style Profile Summary")
    print(f"{'='*60}")
    print(f"Texts: {profile['metadata']['n_texts']}")
    print(f"Total tokens: {profile['metadata']['total_tokens']}")
    print(f"Total operators: {profile['metadata']['total_operators']}")
    print(f"Operator clusters: {profile['n_clusters']}")
    print(f"Phase diversity: {profile['diversity']['phase_diversity']:.4f}")
    print(f"Cluster entropy: {profile['diversity']['cluster_entropy']:.4f}")
    print(f"Full token entropy: {profile['full_entropy']:.2f} bits")
    print(f"Mean coherence: {profile['coherence_stats']}")
    wce = sum(
        (np.bincount(profile['cluster_labels'], minlength=profile['n_clusters'])[c] / len(profile['cluster_labels']))
        * profile['cluster_entropies'][c]
        for c in range(profile['n_clusters'])
    )
    print(f"Within-cluster entropy: {wce:.2f} bits")
    print(f"Entropy reduction: {profile['full_entropy'] - wce:.2f} bits ({(1 - wce/profile['full_entropy'])*100:.1f}%)")
