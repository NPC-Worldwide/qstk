"""Basic CharPAM example -- train a character-level language model.

Demonstrates:
  - Building a CharPAM model (pure numpy, no torch)
  - Training with ComplexAdam optimizer
  - Generating text autoregressively

Usage:
    python -m qstk.cnn.examples.basic_pam
"""

import numpy as np
from qstk.cnn import CharPAM, ComplexAdam


def char_to_ids(text: str) -> np.ndarray:
    """Convert ASCII text to integer ids (0-127)."""
    return np.array([ord(c) % 128 for c in text], dtype=np.int64)


def ids_to_text(ids: np.ndarray) -> str:
    """Convert integer ids back to ASCII text."""
    return "".join(chr(int(i) % 128) for i in ids)


def main():
    # -- Config --
    vocab_size = 128  # ASCII
    dim = 32
    heads = 4
    d_head = 8
    n_layers = 2
    lr = 3e-3
    n_steps = 200
    seq_len = 64

    # -- Training data --
    corpus = (
        "the cat sat on the mat. the dog sat on the log. "
        "a bird flew over the hill. the sun set behind the trees. "
        "water flows down the river to the sea. "
    ) * 20  # repeat to make a small corpus

    print(f"Corpus length: {len(corpus)} characters")

    # -- Build model --
    model = CharPAM(
        vocab_size=vocab_size,
        dim=dim,
        heads=heads,
        d_head=d_head,
        n_layers=n_layers,
    )
    print(f"CharPAM: {model._n_params:,} complex parameters")

    # -- Optimizer --
    opt = ComplexAdam(model.params, lr=lr)

    # -- Training loop --
    corpus_ids = char_to_ids(corpus)

    for step in range(n_steps):
        # Random chunk from corpus
        start = np.random.randint(0, len(corpus_ids) - seq_len)
        chunk = corpus_ids[start : start + seq_len][None, :]  # [1, seq_len]

        loss, grads = model.forward_backward(chunk)
        opt.step(grads)

        # Sync params back into model (optimizer updates the dict in place,
        # but model components hold their own references)
        model.params = opt.params

        if step % 20 == 0:
            diag = model.diagnostics()
            print(
                f"step {step:4d} | loss {loss:.3f} | "
                f"phase_coh {diag['embed_phase_coherence']:.4f}"
            )

    # -- Generate --
    prompt = "the cat "
    prompt_ids = char_to_ids(prompt)
    gen_ids = model.generate(prompt_ids, max_tokens=50, temperature=0.8)
    generated = ids_to_text(gen_ids)
    print(f"\nPrompt:    {prompt!r}")
    print(f"Generated: {generated!r}")


if __name__ == "__main__":
    main()
