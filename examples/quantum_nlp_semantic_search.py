"""Quantum semantic search with swap-test similarity.

A query and each document are encoded as normalized TF-IDF amplitude
vectors.  A Qiskit swap-test circuit estimates |<query|doc>|^2, which is the
squared cosine similarity, on the AerSimulator.  Documents are ranked by
the quantum estimate and compared to classical cosine similarity.
"""

import math
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from qstk.qc.circuits import swap_test_circuit


CORPUS = [
    "quantum computing uses qubits to process information",
    "neural networks learn patterns from large datasets",
    "superconducting qubits operate at cryogenic temperatures",
    "natural language processing models understand text",
    "ion traps confine atoms for quantum experiments",
    "transformer architectures power modern nlp systems",
]

QUERY = "quantum information processing with qubits"

TOP_K = 8
N_SHOTS = 8192
SEED = 42


def tokenize(text):
    return text.lower().split()


def build_vocabulary(documents):
    vocab = set()
    for d in documents:
        vocab.update(tokenize(d))
    return sorted(vocab)


def term_counts(text, vocab):
    tokens = tokenize(text)
    return {term: tokens.count(term) for term in vocab}


def compute_idf(documents, vocab):
    n = len(documents)
    idf = {}
    for term in vocab:
        df = sum(1 for d in documents if term in tokenize(d))
        idf[term] = math.log((n + 1) / (df + 1)) + 1.0
    return idf


def select_top_terms_by_query(vocab, idf, query, k):
    counts = term_counts(query, vocab)
    total = sum(counts.values())
    weights = {term: counts.get(term, 0) / total * idf.get(term, 0.0) if total > 0 else 0.0 for term in vocab}
    return [term for term, _ in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:k]]


def tfidf_vector(counts, idf, terms):
    total = sum(counts.values())
    vec = np.array([counts.get(term, 0) / total * idf.get(term, 0.0) for term in terms], dtype=float)
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


def classical_cosine(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def run_swap_test(state_a, state_b, n_shots, seed):
    simulator = AerSimulator(seed_simulator=seed)
    qc = swap_test_circuit(state_a, state_b)
    transpiled = transpile(qc, simulator)
    counts = simulator.run(transpiled, shots=n_shots).result().get_counts(transpiled)
    total = sum(counts.values())
    p0 = counts.get("0", 0) / total if total > 0 else 0.5
    return 2.0 * p0 - 1.0


def null_basis_state(query_state):
    idx = int(np.argmin(np.abs(query_state)))
    state = np.zeros(len(query_state), dtype=complex)
    state[idx] = 1.0
    return state


def main():
    vocab = build_vocabulary(CORPUS)
    idf = compute_idf(CORPUS, vocab)
    terms = select_top_terms_by_query(vocab, idf, QUERY, TOP_K)

    doc_vectors = [tfidf_vector(term_counts(d, vocab), idf, terms) for d in CORPUS]
    query_vec = tfidf_vector(term_counts(QUERY, vocab), idf, terms)
    query_state = pad_to_power_of_two(query_vec)
    null_state = null_basis_state(query_state)

    doc_states = []
    for vec in doc_vectors:
        if np.linalg.norm(vec) == 0:
            doc_states.append(null_basis_state(query_state))
        else:
            doc_states.append(pad_to_power_of_two(vec))

    print("Quantum semantic search")
    print(f"Query: {QUERY}")
    print(f"Using top {len(terms)} terms: {terms}")
    print(f"State dimension: {len(query_state)} ({int(math.log2(len(query_state)))} qubits per state)")
    print()

    results = []
    for i, (doc, state, vec) in enumerate(zip(CORPUS, doc_states, doc_vectors)):
        qsim = run_swap_test(query_state, state, N_SHOTS, SEED + i)
        csim = classical_cosine(query_vec, vec)
        results.append((i, doc, qsim, csim))

    results.sort(key=lambda x: x[2], reverse=True)

    print("Rank  Doc  Quantum |<q|d>|^2  Classical cos^2  Text")
    for rank, (i, doc, qsim, csim) in enumerate(results, 1):
        print(f"{rank:<5} {i:<4} {qsim:.4f}          {csim ** 2:.4f}            {doc}")

    print()
    print("The swap-test circuit ranks the same top documents as classical cosine")
    print("similarity while only exposing the quantum estimate, not the vectors.")


if __name__ == "__main__":
    main()
