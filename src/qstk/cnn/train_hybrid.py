#!/usr/bin/env python3
"""
train_hybrid.py -- Train CharPAM (autograd) on Shakespeare, compare to pure numpy.

Uses PyTorch autograd for exact Wirtinger gradients through the full
complex-valued forward pass, while keeping numpy as the storage format.

The hypothesis: exact autograd gives cleaner training signal than the
approximate analytical gradients in CharPAM.forward_backward(), especially
through complex normalization layers and modReLU phase paths.

Usage:
    python train_hybrid.py

Outputs:
    hybrid_results.png -- loss curves, phase coherence, state SVD
"""

import sys
import os
import time
import numpy as np

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from qstk.cnn.autograd import CharPAM as CharPAMAutograd
from qstk.cnn.model import CharPAM

# Line-buffered output
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Shakespeare corpus
# ---------------------------------------------------------------------------
SHAKESPEARE = """
HAMLET: To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them. To die, to sleep --
No more -- and by a sleep to say we end
The heartache, and the thousand natural shocks
That flesh is heir to. 'Tis a consummation
Devoutly to be wished. To die, to sleep --
To sleep -- perchance to dream: ay, there's the rub,
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office, and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscovered country, from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment,
With this regard their currents turn awry
And lose the name of action.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious:
If it were so, it was a grievous fault,
And grievously hath Caesar answered it.
Here, under leave of Brutus and the rest --
For Brutus is an honourable man;
So are they all, all honourable men --
Come I to speak in Caesar's funeral.
He was my friend, faithful and just to me:
But Brutus says he was ambitious;
And Brutus is an honourable man.

All the world's a stage,
And all the men and women merely players:
They have their exits and their entrances;
And one man in his time plays many parts,
His acts being seven ages. At first the infant,
Mewling and puking in the nurse's arms.

Now is the winter of our discontent
Made glorious summer by this son of York;
And all the clouds that loured upon our house
In the deep bosom of the ocean buried.

Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date.
""".strip()


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.token2idx = {c: i for i, c in enumerate(chars)}
        self.idx2token = {i: c for c, i in self.token2idx.items()}
        self.vocab_size = len(chars)
        self.corpus_ids = np.array(
            [self.token2idx[c] for c in text], dtype=np.int64
        )
        self.kind = 'char'

    def encode(self, text):
        return np.array(
            [self.token2idx.get(c, 0) for c in text], dtype=np.int64
        )

    def decode(self, ids):
        return ''.join(self.idx2token.get(int(i), '?') for i in ids)


# ---------------------------------------------------------------------------
# Complex Adam optimizer (operates on numpy arrays)
# ---------------------------------------------------------------------------
class ComplexAdam:
    """Adam optimizer for complex128 parameters stored as numpy arrays."""

    def __init__(self, param_dict, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0, grad_clip=1.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.grad_clip = grad_clip
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in param_dict.items()}
        self.v = {k: np.zeros_like(v, dtype=np.float64)
                  for k, v in param_dict.items()}

    def step(self, params, grads):
        self.t += 1
        for name in params:
            if name not in grads:
                continue
            g = grads[name].copy()

            # Weight decay on projection matrices
            if self.wd > 0 and 'W' in name:
                g = g + self.wd * params[name]

            # Gradient clipping by magnitude
            g_mag = np.abs(g)
            max_mag = g_mag.max() if g_mag.size > 0 else 0.0
            if max_mag > self.grad_clip:
                g = g * (self.grad_clip / max_mag)

            # Adam update
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = (self.beta2 * self.v[name]
                            + (1 - self.beta2) * (g.real**2 + g.imag**2))

            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Data batching
# ---------------------------------------------------------------------------
def get_batch(corpus_ids, batch_size, seq_len):
    max_start = len(corpus_ids) - seq_len - 1
    starts = np.random.randint(0, max(1, max_start), size=batch_size)
    return np.array([corpus_ids[s:s + seq_len] for s in starts])


# ---------------------------------------------------------------------------
# Training loop for one model
# ---------------------------------------------------------------------------
def train_model(model, tokenizer, label, n_epochs=600, lr=5e-4,
                batch_size=8, seq_len=48, use_hybrid=True):
    """Train either CharPAMAutograd or CharPAM, return metrics."""

    opt = ComplexAdam(
        model.params, lr=lr, beta1=0.9, beta2=0.999,
        weight_decay=1e-4, grad_clip=1.0,
    )

    losses = []
    coherences = []
    gammas_history = []
    svd_snapshots = []  # (epoch, singular values of S from layer 0)
    times = []

    corpus = tokenizer.corpus_ids

    print(f"\n{'='*60}")
    print(f"  Training: {label}")
    print(f"  Params: {sum(p.size for p in model.params.values()):,} complex")
    print(f"  Epochs: {n_epochs}, LR: {lr}, Batch: {batch_size}, Seq: {seq_len}")
    print(f"{'='*60}\n")

    t0 = time.time()

    for epoch in range(n_epochs):
        batch = get_batch(corpus, batch_size, seq_len)

        # Forward + backward
        loss, grads = model.forward_backward(batch)

        # Cosine LR schedule with warmup
        warmup = 30
        if epoch < warmup:
            lr_t = lr * (epoch + 1) / warmup
        else:
            progress = (epoch - warmup) / max(1, n_epochs - warmup)
            lr_t = lr * 0.5 * (1 + np.cos(np.pi * progress))
        opt.lr = lr_t

        # Update
        opt.step(model.params, grads)
        if hasattr(model, '_params'):
            # CharPAMAutograd stores in _params
            pass  # params property already references _params

        losses.append(loss)
        elapsed = time.time() - t0
        times.append(elapsed)

        # Diagnostics
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            diag = model.diagnostics()
            coherences.append((epoch, diag['embed_phase_coherence']))

            gamma_vals = diag['layer_gammas']
            gammas_history.append((epoch, gamma_vals))

            # SVD of PAM state from a probe forward pass
            probe = get_batch(corpus, 1, seq_len)
            svs = _get_state_svd(model, probe)
            svd_snapshots.append((epoch, svs))

            print(
                f"  [{label}] epoch {epoch:4d}  "
                f"loss={loss:.4f}  "
                f"coherence={diag['embed_phase_coherence']:.4f}  "
                f"gamma_L0={gamma_vals[0].mean():.4f}  "
                f"lr={lr_t:.6f}  "
                f"time={elapsed:.1f}s"
            )

    total_time = time.time() - t0
    print(f"\n  [{label}] Done. Final loss: {losses[-1]:.4f}  "
          f"Time: {total_time:.1f}s\n")

    return {
        'losses': losses,
        'coherences': coherences,
        'gammas': gammas_history,
        'svd_snapshots': svd_snapshots,
        'times': times,
        'total_time': total_time,
    }


def _get_state_svd(model, token_ids):
    """Run a forward pass and extract singular values of PAM state (layer 0)."""
    p = model.params if isinstance(model.params, dict) else model.params
    B, T = token_ids.shape
    dim = model.dim
    heads = model.heads
    d_head = model.d_head

    z = p['embed'][token_ids]

    # Just run layer 0
    pfx = 'L0'
    mag = np.abs(z)
    rms = np.sqrt(np.mean(mag ** 2, axis=-1, keepdims=True) + 1e-6)
    ns = np.abs(p[f'{pfx}.norm_scale'])
    h = (z / (mag + 1e-8)) * (mag / rms) * ns

    Wq, Wk, Wv = p[f'{pfx}.Wq'], p[f'{pfx}.Wk'], p[f'{pfx}.Wv']
    q = (h @ Wq.T).reshape(B, T, heads, d_head)
    k = (h @ Wk.T).reshape(B, T, heads, d_head)
    v = (h @ Wv.T).reshape(B, T, heads, d_head)

    q = q / (np.abs(q) + 1e-8) * (d_head ** -0.5)
    k = k / (np.abs(k) + 1e-8)

    gamma = 1.0 / (1.0 + np.exp(-p[f'{pfx}.decay_bias'].real))

    S = np.zeros((B, heads, d_head, d_head), dtype=np.complex128)
    for t in range(T):
        outer = v[:, t, :, :, None] * k[:, t, :, None, :].conj()
        S = S * gamma[None, :, None, None] + outer

    # SVD of final state, head 0
    S_head0 = S[0, 0]  # [d_head, d_head]
    svs = np.linalg.svd(S_head0, compute_uv=False)
    return svs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plots(hybrid_metrics, numpy_metrics, save_path):
    """Generate comparison plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'CharPAMAutograd (torch autograd) vs CharPAM (analytical gradients)',
        fontsize=14, fontweight='bold',
    )

    # --- Loss curves ---
    ax = axes[0, 0]
    h_losses = hybrid_metrics['losses']
    n_losses = numpy_metrics['losses']

    # Smoothed loss (exponential moving average)
    def ema(data, alpha=0.05):
        out = []
        s = data[0]
        for x in data:
            s = alpha * x + (1 - alpha) * s
            out.append(s)
        return out

    ax.plot(ema(h_losses), label='Hybrid (torch autograd)', color='#2196F3',
            linewidth=2)
    ax.plot(ema(n_losses), label='NumPy (analytical grad)', color='#FF5722',
            linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Training Loss (smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark final losses
    ax.axhline(y=h_losses[-1], color='#2196F3', alpha=0.3, linestyle=':')
    ax.axhline(y=n_losses[-1], color='#FF5722', alpha=0.3, linestyle=':')
    ax.text(len(h_losses) * 0.7, h_losses[-1] + 0.05,
            f'final={h_losses[-1]:.3f}', color='#2196F3', fontsize=9)
    ax.text(len(n_losses) * 0.7, n_losses[-1] + 0.15,
            f'final={n_losses[-1]:.3f}', color='#FF5722', fontsize=9)

    # --- Phase coherence ---
    ax = axes[0, 1]
    if hybrid_metrics['coherences']:
        h_ep, h_pc = zip(*hybrid_metrics['coherences'])
        ax.plot(h_ep, h_pc, 'o-', label='Hybrid', color='#2196F3',
                markersize=5)
    if numpy_metrics['coherences']:
        n_ep, n_pc = zip(*numpy_metrics['coherences'])
        ax.plot(n_ep, n_pc, 's--', label='NumPy', color='#FF5722',
                markersize=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Phase Coherence')
    ax.set_title('Embedding Phase Coherence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- State SVD (final snapshot) ---
    ax = axes[1, 0]
    if hybrid_metrics['svd_snapshots']:
        _, h_svs = hybrid_metrics['svd_snapshots'][-1]
        ax.bar(np.arange(len(h_svs)) - 0.15, h_svs, width=0.3,
               label='Hybrid', color='#2196F3', alpha=0.8)
    if numpy_metrics['svd_snapshots']:
        _, n_svs = numpy_metrics['svd_snapshots'][-1]
        ax.bar(np.arange(len(n_svs)) + 0.15, n_svs, width=0.3,
               label='NumPy', color='#FF5722', alpha=0.8)
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Magnitude')
    ax.set_title('PAM State SVD (Layer 0, Head 0, Final)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Gamma evolution ---
    ax = axes[1, 1]
    if hybrid_metrics['gammas']:
        h_ep_g = [e for e, _ in hybrid_metrics['gammas']]
        h_g0 = [g[0].mean() for _, g in hybrid_metrics['gammas']]
        ax.plot(h_ep_g, h_g0, 'o-', label='Hybrid L0', color='#2196F3',
                markersize=5)
        if len(hybrid_metrics['gammas'][0][1]) > 1:
            h_g1 = [g[1].mean() for _, g in hybrid_metrics['gammas']]
            ax.plot(h_ep_g, h_g1, '^-', label='Hybrid L1', color='#03A9F4',
                    markersize=5)
    if numpy_metrics['gammas']:
        n_ep_g = [e for e, _ in numpy_metrics['gammas']]
        n_g0 = [g[0].mean() for _, g in numpy_metrics['gammas']]
        ax.plot(n_ep_g, n_g0, 's--', label='NumPy L0', color='#FF5722',
                markersize=5)
        if len(numpy_metrics['gammas'][0][1]) > 1:
            n_g1 = [g[1].mean() for _, g in numpy_metrics['gammas']]
            ax.plot(n_ep_g, n_g1, 'v--', label='NumPy L1', color='#FF9800',
                    markersize=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gamma (decay factor)')
    ax.set_title('Learned Decay Rates')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(42)

    # Tokenize
    tokenizer = CharTokenizer(SHAKESPEARE)
    V = tokenizer.vocab_size
    print(f"Vocabulary: {V} characters, Corpus: {len(tokenizer.corpus_ids)} tokens")

    # Hyperparameters (same for both)
    dim = 32
    heads = 4
    d_head = 8
    n_layers = 2
    n_epochs = 600
    lr = 5e-4
    batch_size = 8
    seq_len = 48

    # ---- Train CharPAMAutograd (torch autograd) ----
    np.random.seed(42)
    hybrid_model = CharPAMAutograd(
        vocab_size=V, dim=dim, heads=heads, d_head=d_head,
        n_layers=n_layers, decay_init=-2.0,
    )
    hybrid_metrics = train_model(
        hybrid_model, tokenizer, label='Hybrid',
        n_epochs=n_epochs, lr=lr, batch_size=batch_size, seq_len=seq_len,
        use_hybrid=True,
    )

    # ---- Train CharPAM (analytical gradients) ----
    np.random.seed(42)
    numpy_model = CharPAM(
        vocab_size=V, dim=dim, heads=heads, d_head=d_head,
        n_layers=n_layers, decay_init=-2.0,
    )
    numpy_metrics = train_model(
        numpy_model, tokenizer, label='NumPy',
        n_epochs=n_epochs, lr=lr, batch_size=batch_size, seq_len=seq_len,
        use_hybrid=False,
    )

    # ---- Generate samples ----
    print("\n" + "=" * 60)
    print("  Generation Samples")
    print("=" * 60)

    prompt = "HAMLET: "
    prompt_ids = tokenizer.encode(prompt)

    print(f"\n--- Hybrid model ---")
    np.random.seed(7)
    gen_hybrid = hybrid_model.generate(prompt_ids, max_tokens=120, temperature=0.7)
    print(tokenizer.decode(gen_hybrid))

    print(f"\n--- NumPy model ---")
    np.random.seed(7)
    gen_numpy = numpy_model.generate(prompt_ids, max_tokens=120, temperature=0.7)
    print(tokenizer.decode(gen_numpy))

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    h_final = hybrid_metrics['losses'][-1]
    n_final = numpy_metrics['losses'][-1]
    h_min = min(hybrid_metrics['losses'])
    n_min = min(numpy_metrics['losses'])
    print(f"  Hybrid -- final loss: {h_final:.4f}, best: {h_min:.4f}, "
          f"time: {hybrid_metrics['total_time']:.1f}s")
    print(f"  NumPy  -- final loss: {n_final:.4f}, best: {n_min:.4f}, "
          f"time: {numpy_metrics['total_time']:.1f}s")

    if h_final < n_final:
        pct = (n_final - h_final) / n_final * 100
        print(f"\n  Hybrid wins by {pct:.1f}% lower final loss.")
    else:
        pct = (h_final - n_final) / h_final * 100
        print(f"\n  NumPy wins by {pct:.1f}% lower final loss.")

    # ---- Plot ----
    save_path = os.path.join(os.path.dirname(__file__), 'hybrid_results.png')
    make_plots(hybrid_metrics, numpy_metrics, save_path)


if __name__ == '__main__':
    main()
