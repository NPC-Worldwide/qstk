"""Quantum-enhanced topic clustering.

Documents from three topics are encoded as normalized TF-IDF amplitude
vectors over a curated topic vocabulary.  A quantum k-means loop estimates
document-centroid distances with swap-test circuits on the AerSimulator,
assigns each document to the closest centroid classically, and updates the
centroids classically.  The quantum circuits replace the inner-product step
of k-means with a real measurement-based distance estimate.
"""

import math
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from qstk.qc.circuits import swap_test_circuit


DOCUMENTS = [
    ("the stock market rose after strong earnings as investors bought shares", "finance"),
    ("investors bought bonds as interest rates fell sharply in the market", "finance"),
    ("the quarterback threw a touchdown in the final minute of the game", "sports"),
    ("the team clinched the championship with a clean sweep of the series", "sports"),
    ("gene editing could cure inherited diseases according to the study", "biology"),
    ("the new study maps proteins inside a single cell with high precision", "biology"),
]

VOCABULARY = [
    "stock", "market", "investors",
    "quarterback", "touchdown", "team", "championship",
    "gene", "cell", "proteins",
]

N_CLUSTERS = 3
N_ITERATIONS = 8
N_SHOTS = 4096
SEED = 42


def tokenize(text):
    return text.lower().split()


def term_counts(text, vocab):
    tokens = tokenize(text)
    return {term: tokens.count(term) for term in vocab}


def compute_idf(documents, vocab):
    n = len(documents)
    idf = {}
    for term in vocab:
        df = sum(1 for text, _ in documents if term in tokenize(text))
        idf[term] = math.log((n + 1) / (df + 1)) + 1.0
    return idf


def tfidf_vector(counts, idf, vocab):
    total = sum(counts.values())
    vec = np.array([counts.get(term, 0) / total * idf.get(term, 0.0) for term in vocab], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def pad_to_power_of_two(vec):
    dim = 2 ** math.ceil(math.log2(max(len(vec), 1)))
    out = np.zeros(dim, dtype=complex)
    out[:len(vec)] = vec
    norm = np.linalg.norm(out)
    if norm == 0:
        out[0] = 1.0
        return out
    return out / norm


def squared_distance_from_overlap(overlap_squared):
    return max(0.0, 2.0 - 2.0 * overlap_squared)


def estimate_distance(state_a, state_b, n_shots, seed):
    simulator = AerSimulator(seed_simulator=seed)
    qc = swap_test_circuit(state_a, state_b)
    transpiled = transpile(qc, simulator)
    counts = simulator.run(transpiled, shots=n_shots).result().get_counts(transpiled)
    total = sum(counts.values())
    p0 = counts.get("0", 0) / total if total > 0 else 0.5
    overlap_squared = 2.0 * p0 - 1.0
    return squared_distance_from_overlap(overlap_squared)


def initialize_centroids(states, k, seed, init_indices=None):
    if init_indices is not None:
        return [states[i].copy() for i in init_indices]
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(states), size=k, replace=False)
    return [states[i].copy() for i in indices]


def average_state(states):
    if not states:
        return None
    avg = np.zeros_like(states[0], dtype=complex)
    for s in states:
        avg += s
    avg = avg / np.linalg.norm(avg)
    return avg


def quantum_kmeans(states, k, n_iterations, n_shots, seed, init_indices=None):
    centroids = initialize_centroids(states, k, seed, init_indices)
    assignments = [0] * len(states)
    for iteration in range(n_iterations):
        new_assignments = []
        for i, state in enumerate(states):
            distances = [estimate_distance(state, centroid, n_shots, seed + iteration * 100 + i * 10 + c)
                         for c, centroid in enumerate(centroids)]
            new_assignments.append(int(np.argmin(distances)))
        if new_assignments == assignments and iteration > 0:
            assignments = new_assignments
            break
        assignments = new_assignments
        new_centroids = []
        for c in range(k):
            cluster = [states[i] for i, a in enumerate(assignments) if a == c]
            new_centroids.append(average_state(cluster) if cluster else centroids[c])
        centroids = new_centroids
    return assignments, centroids


def classical_kmeans(vectors, k, n_iterations, seed, init_indices=None):
    centroids = [vectors[i].copy() for i in init_indices] if init_indices else initialize_centroids(vectors, k, seed)
    assignments = [0] * len(vectors)
    for iteration in range(n_iterations):
        new_assignments = []
        for vec in vectors:
            distances = [np.linalg.norm(vec - c) for c in centroids]
            new_assignments.append(int(np.argmin(distances)))
        if new_assignments == assignments and iteration > 0:
            assignments = new_assignments
            break
        assignments = new_assignments
        new_centroids = []
        for c in range(k):
            cluster = [vectors[i] for i, a in enumerate(assignments) if a == c]
            new_centroids.append(np.mean(cluster, axis=0) if cluster else centroids[c])
            norm = np.linalg.norm(new_centroids[-1])
            if norm > 0:
                new_centroids[-1] = new_centroids[-1] / norm
        centroids = new_centroids
    return assignments


def main():
    idf = compute_idf(DOCUMENTS, VOCABULARY)
    vectors = [tfidf_vector(term_counts(text, VOCABULARY), idf, VOCABULARY) for text, _ in DOCUMENTS]
    states = [pad_to_power_of_two(v) for v in vectors]

    true_labels = [label for _, label in DOCUMENTS]

    print("Quantum-enhanced topic clustering")
    print(f"Documents: {len(DOCUMENTS)}  Clusters: {N_CLUSTERS}  Vocabulary: {VOCABULARY}")
    print(f"State dimension: {len(states[0])} ({int(math.log2(len(states[0])))} qubits per state)")
    print()

    q_assignments, _ = quantum_kmeans(states, N_CLUSTERS, N_ITERATIONS, N_SHOTS, SEED)
    c_assignments = classical_kmeans(vectors, N_CLUSTERS, N_ITERATIONS, SEED)

    print("Doc  True topic           Quantum cluster  Classical cluster")
    for i, (text, true) in enumerate(DOCUMENTS):
        print(f"{i:<4} {true:<20} {q_assignments[i]:<16} {c_assignments[i]}")

    q_matches = sum(1 for i in range(len(DOCUMENTS)) for j in range(i + 1, len(DOCUMENTS))
                    if (true_labels[i] == true_labels[j]) == (q_assignments[i] == q_assignments[j]))
    c_matches = sum(1 for i in range(len(DOCUMENTS)) for j in range(i + 1, len(DOCUMENTS))
                    if (true_labels[i] == true_labels[j]) == (c_assignments[i] == c_assignments[j]))
    pairs = len(DOCUMENTS) * (len(DOCUMENTS) - 1) // 2
    print()
    print(f"Quantum cluster agreement with true topics: {q_matches}/{pairs} ({q_matches / pairs:.2%})")
    print(f"Classical cluster agreement with true topics: {c_matches}/{pairs} ({c_matches / pairs:.2%})")

    print()
    print("The quantum k-means loop uses AerSimulator swap-test circuits to")
    print("estimate document-centroid distances; everything else is classical.")


if __name__ == "__main__":
    main()
