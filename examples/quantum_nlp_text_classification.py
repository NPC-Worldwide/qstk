"""Quantum kernel text classification.

Documents are encoded as normalized TF-IDF amplitude vectors over a curated
sentiment vocabulary.  A quantum kernel matrix is built by running Qiskit
Aer swap-test circuits for every pair of documents, estimating the squared
overlap |<d_i|d_j>|^2.  A classical SVM then uses that quantum kernel to
classify sentiment.  This is a hybrid quantum-classical NLP classifier:
the quantum circuits compute the similarity structure, the classical model
learns the decision boundary.
"""

import math
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from qstk.qc.circuits import swap_test_circuit


DOCUMENTS = [
    ("the movie was absolutely excellent and thrilling", 1),
    ("a wonderful performance by the amazing cast", 1),
    ("one of the greatest films ever brilliant and excellent", 1),
    ("amazing direction excellent story truly wonderful", 1),
    ("terrible acting and a boring predictable plot", 0),
    ("worst movie ever completely awful dull waste of time", 0),
    ("awful acting boring story disappointing in every way", 0),
    ("terrible dull and awful no redeeming qualities", 0),
]

VOCABULARY = [
    "excellent", "amazing", "brilliant", "wonderful",
    "terrible", "awful", "boring", "dull",
]

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


def quantum_kernel_matrix(states, n_shots, seed):
    n = len(states)
    K = np.eye(n, dtype=float)
    simulator = AerSimulator(seed_simulator=seed)
    for i in range(n):
        for j in range(i + 1, n):
            qc = swap_test_circuit(states[i], states[j])
            transpiled = transpile(qc, simulator)
            counts = simulator.run(transpiled, shots=n_shots).result().get_counts(transpiled)
            total = sum(counts.values())
            p0 = counts.get("0", 0) / total if total > 0 else 0.5
            k = 2.0 * p0 - 1.0
            K[i, j] = k
            K[j, i] = k
    return K


def kernel_nearest_neighbor(K, labels, test_idx, train_idx):
    best = None
    best_sim = -1.0
    for i in train_idx:
        if K[test_idx, i] > best_sim:
            best_sim = K[test_idx, i]
            best = i
    return labels[best]


def main():
    idf = compute_idf(DOCUMENTS, VOCABULARY)
    labels = np.array([label for _, label in DOCUMENTS])
    states = [pad_to_power_of_two(tfidf_vector(term_counts(text, VOCABULARY), idf, VOCABULARY))
              for text, _ in DOCUMENTS]

    print("Quantum kernel text classification")
    print(f"Documents: {len(DOCUMENTS)}  Features: {len(VOCABULARY)}  Qubits per state: {int(math.log2(len(states[0])))}")
    print(f"Vocabulary: {VOCABULARY}")
    print("Building kernel matrix from AerSimulator swap-test circuits...")

    K = quantum_kernel_matrix(states, N_SHOTS, SEED)

    print("Kernel matrix:")
    print(np.round(K, 3))
    print()

    knn_predictions = []
    for test_idx in range(len(DOCUMENTS)):
        train_idx = [i for i in range(len(DOCUMENTS)) if i != test_idx]
        knn_predictions.append(kernel_nearest_neighbor(K, labels, test_idx, train_idx))
    knn_accuracy = sum(1 for p, t in zip(knn_predictions, labels) if p == t) / len(labels)
    print(f"Quantum kernel 1-NN leave-one-out accuracy: {knn_accuracy:.2%}")

    try:
        from sklearn.svm import SVC
        svm_predictions = []
        for test_idx in range(len(DOCUMENTS)):
            train_idx = [i for i in range(len(DOCUMENTS)) if i != test_idx]
            svm = SVC(kernel="precomputed")
            svm.fit(K[np.ix_(train_idx, train_idx)], labels[train_idx])
            svm_predictions.append(svm.predict(K[np.ix_([test_idx], train_idx)])[0])
        svm_accuracy = sum(1 for p, t in zip(svm_predictions, labels) if p == t) / len(labels)
        print(f"Quantum kernel SVM leave-one-out accuracy: {svm_accuracy:.2%}")
    except ImportError:
        print("sklearn not installed; SVM result skipped")

    print()
    print("The classifier is hybrid: quantum state preparation + swap-test")
    print("similarity circuits produce the kernel, and a classical SVM learns")
    print("the sentiment boundary from that kernel.")


if __name__ == "__main__":
    main()
