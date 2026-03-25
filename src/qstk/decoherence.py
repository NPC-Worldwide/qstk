"""Decoherence metrics for LLM output analysis.

Measures script diversity, entropy, code emergence, coherence breakdown,
and semantic drift as functions of sampling parameters.
"""

import os
import json
import time
import hashlib
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np


@dataclass
class StreamSample:
    model: str
    provider: str
    temperature: float
    top_p: Optional[float]
    prompt: str
    output: str
    token_count: int
    generation_time: float
    interrupted: bool
    interrupt_position: Optional[int]


@dataclass
class DecoherenceMetrics:
    script_diversity: int
    script_distribution: Dict[str, int]
    char_entropy: float
    word_entropy: float
    avg_word_length: float
    code_fragment_density: float
    whitespace_ratio: float
    punctuation_ratio: float
    numeric_ratio: float
    longest_coherent_run: int
    language_tags: List[str]
    embedding_drift: Optional[float] = None


@dataclass
class DecoherenceExperimentConfig:
    name: str
    models: List[Dict[str, str]]
    temperatures: List[float]
    top_ps: List[Optional[float]]
    prompts: List[str]
    n_samples_per_config: int = 3
    max_tokens: int = 200
    interrupt_likelihood: float = 0.01
    output_dir: str = "./decoherence_experiments"


def get_unicode_script(char: str) -> str:
    """Classify a character by its Unicode script."""
    try:
        name = unicodedata.name(char, "UNKNOWN")
        for script, keyword in [
            ("Latin", "LATIN"), ("Cyrillic", "CYRILLIC"), ("Arabic", "ARABIC"),
            ("Hangul", "HANGUL"), ("Hebrew", "HEBREW"), ("Georgian", "GEORGIAN"),
            ("Thai", "THAI"), ("Devanagari", "DEVANAGARI"), ("Greek", "GREEK"),
        ]:
            if keyword in name:
                return script
        if "CJK" in name or "CHINESE" in name:
            return "CJK"
        if "HIRAGANA" in name or "KATAKANA" in name:
            return "Japanese"
        if char.isspace():
            return "Whitespace"
        if char.isdigit():
            return "Numeric"
        if char in ".,;:!?\"'()-[]{}":
            return "Punctuation"
        return "Other"
    except Exception:
        return "Unknown"


def compute_entropy(counts: Counter) -> float:
    """Compute Shannon entropy from a Counter of symbol counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)


def detect_code_fragments(text: str) -> float:
    """Estimate code fragment density in text (hits per word)."""
    code_patterns = [
        "def ", "class ", "import ", "from ", "return ",
        "function ", "const ", "let ", "var ", "=>",
        "public ", "private ", "void ", "int ", "string ",
        "if (", "for (", "while (", "else {",
        ".()", ".get(", ".set(", "->", "::",
        "{{", "}}", "[];", "();",
    ]
    text_lower = text.lower()
    hits = sum(1 for p in code_patterns if p.lower() in text_lower)
    return hits / max(len(text.split()), 1)


def find_longest_coherent_run(text: str) -> int:
    """Find the longest run of words sharing the same dominant script."""
    words = text.split()
    if not words:
        return 0

    max_run = 0
    current_run = 0
    prev_script = None

    for word in words:
        if not word:
            continue
        scripts = set(
            get_unicode_script(c) for c in word if c not in " \t\n"
        )
        scripts -= {"Whitespace", "Punctuation", "Numeric"}

        dominant = (
            max(scripts, key=lambda s: sum(1 for c in word if get_unicode_script(c) == s), default="Other")
            if scripts
            else "Other"
        )

        if dominant == prev_script and dominant != "Other":
            current_run += 1
        else:
            max_run = max(max_run, current_run)
            current_run = 1
            prev_script = dominant

    return max(max_run, current_run)


def compute_decoherence_metrics(
    text: str,
    prompt: str = "",
    embedder: Optional[Any] = None,
) -> DecoherenceMetrics:
    """Compute a full set of decoherence metrics for a text sample.

    Parameters
    ----------
    text : str
        The LLM-generated output text.
    prompt : str
        The original prompt (for embedding drift calculation).
    embedder : optional
        A SentenceTransformer instance (or compatible) for embedding drift.

    Returns
    -------
    DecoherenceMetrics dataclass.
    """
    script_counts = Counter()
    for char in text:
        script_counts[get_unicode_script(char)] += 1

    meaningful_scripts = {
        k: v for k, v in script_counts.items()
        if k not in ["Whitespace", "Punctuation", "Numeric"]
    }

    char_entropy = compute_entropy(Counter(text))

    words = text.split()
    word_entropy = compute_entropy(Counter(words))
    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0

    code_density = detect_code_fragments(text)

    total_chars = max(len(text), 1)
    whitespace_ratio = sum(1 for c in text if c.isspace()) / total_chars
    punct_ratio = script_counts.get("Punctuation", 0) / total_chars
    numeric_ratio = script_counts.get("Numeric", 0) / total_chars

    longest_run = find_longest_coherent_run(text)

    lang_tags = [s for s in meaningful_scripts if meaningful_scripts[s] > 5]

    embedding_drift = None
    if embedder and prompt:
        prompt_emb = embedder.encode(prompt)
        text_emb = embedder.encode(text[:500])
        embedding_drift = float(
            1 - np.dot(prompt_emb, text_emb)
            / (np.linalg.norm(prompt_emb) * np.linalg.norm(text_emb))
        )

    return DecoherenceMetrics(
        script_diversity=len(meaningful_scripts),
        script_distribution=dict(meaningful_scripts),
        char_entropy=char_entropy,
        word_entropy=word_entropy,
        avg_word_length=avg_word_len,
        code_fragment_density=code_density,
        whitespace_ratio=whitespace_ratio,
        punctuation_ratio=punct_ratio,
        numeric_ratio=numeric_ratio,
        longest_coherent_run=longest_run,
        language_tags=lang_tags,
        embedding_drift=embedding_drift,
    )


@dataclass
class DecoherenceResult:
    config_hash: str
    sample: StreamSample
    metrics: DecoherenceMetrics
    timestamp: float


class DecoherenceExperiment:
    """Run decoherence sweeps across models and temperature settings."""

    def __init__(self, config: DecoherenceExperimentConfig):
        self.config = config
        self.results: List[DecoherenceResult] = []
        self.embedder = None

        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pass

        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _config_hash(self, model, provider, temp, top_p, prompt):
        key = f"{model}|{provider}|{temp}|{top_p}|{prompt[:50]}"
        return hashlib.md5(key.encode()).hexdigest()[:8]

    def run(
        self,
        generation_fn=None,
        verbose: bool = True,
    ) -> List[DecoherenceResult]:
        """Run the experiment.

        Parameters
        ----------
        generation_fn : callable, optional
            A function(prompt, model, provider, temperature, top_p, max_tokens,
            interrupt_likelihood) -> StreamSample. If None, uses npcpy's
            get_litellm_response.
        verbose : bool
            Print progress.
        """
        if generation_fn is None:
            from npcpy.gen.response import get_litellm_response
            generation_fn = self._default_generation

        total_runs = (
            len(self.config.models)
            * len(self.config.temperatures)
            * len(self.config.top_ps)
            * len(self.config.prompts)
            * self.config.n_samples_per_config
        )

        run_idx = 0
        for model_cfg in self.config.models:
            model = model_cfg["model"]
            provider = model_cfg["provider"]
            for temp in self.config.temperatures:
                for top_p in self.config.top_ps:
                    for prompt in self.config.prompts:
                        cfg_hash = self._config_hash(model, provider, temp, top_p, prompt)
                        for _ in range(self.config.n_samples_per_config):
                            run_idx += 1
                            if verbose:
                                print(
                                    f"[{run_idx}/{total_runs}] "
                                    f"{provider}/{model} T={temp} top_p={top_p}"
                                )
                            try:
                                sample = generation_fn(
                                    prompt, model, provider, temp, top_p,
                                    self.config.max_tokens,
                                    self.config.interrupt_likelihood,
                                )
                                metrics = compute_decoherence_metrics(
                                    sample.output, prompt, self.embedder
                                )
                                result = DecoherenceResult(
                                    config_hash=cfg_hash,
                                    sample=sample,
                                    metrics=metrics,
                                    timestamp=time.time(),
                                )
                                self.results.append(result)
                                if verbose:
                                    print(
                                        f"    scripts={metrics.script_diversity} "
                                        f"entropy={metrics.char_entropy:.2f} "
                                        f"coherent_run={metrics.longest_coherent_run}"
                                    )
                            except Exception as e:
                                print(f"    ERROR: {e}")

        return self.results

    @staticmethod
    def _default_generation(prompt, model, provider, temperature, top_p,
                            max_tokens, interrupt_likelihood):
        from npcpy.gen.response import get_litellm_response

        kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        if top_p is not None:
            kwargs["top_p"] = top_p

        messages = [
            {"role": "system",
             "content": "Continue generating without attempting to answer. "
                        "Simply produce text without consideration for "
                        "practicality or coherence."},
            {"role": "user", "content": prompt},
        ]

        start_time = time.time()
        response = get_litellm_response(
            prompt=prompt, model=model, provider=provider,
            messages=messages, stream=True, **kwargs,
        )

        output = ""
        interrupted = False
        interrupt_pos = None
        token_count = 0

        for chunk in response["response"]:
            if provider == "ollama":
                content = chunk.get("message", {}).get("content", "")
            else:
                content = "".join(
                    choice.delta.content
                    for choice in chunk.choices
                    if choice.delta.content is not None
                )
            if content:
                output += content
                token_count += 1
                if interrupt_likelihood > 0 and np.random.random() < interrupt_likelihood:
                    interrupted = True
                    interrupt_pos = len(output)
                    break

        return StreamSample(
            model=model, provider=provider, temperature=temperature,
            top_p=top_p, prompt=prompt, output=output,
            token_count=token_count, generation_time=time.time() - start_time,
            interrupted=interrupted, interrupt_position=interrupt_pos,
        )

    def save_results(self, filename: str = "results.json"):
        data = [
            {
                "config_hash": r.config_hash,
                "timestamp": r.timestamp,
                "sample": asdict(r.sample),
                "metrics": asdict(r.metrics),
            }
            for r in self.results
        ]
        path = self.output_path / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Saved {len(data)} results to {path}")

    def load_results(self, filename: str = "results.json"):
        path = self.output_path / filename
        with open(path) as f:
            data = json.load(f)
        self.results = []
        for entry in data:
            sample = StreamSample(**entry["sample"])
            metrics = DecoherenceMetrics(**entry["metrics"])
            self.results.append(
                DecoherenceResult(
                    config_hash=entry["config_hash"],
                    sample=sample,
                    metrics=metrics,
                    timestamp=entry["timestamp"],
                )
            )
        print(f"Loaded {len(self.results)} results")

    def to_dataframe(self):
        import pandas as pd
        rows = []
        for r in self.results:
            rows.append({
                "model": r.sample.model,
                "provider": r.sample.provider,
                "temperature": r.sample.temperature,
                "top_p": r.sample.top_p,
                "token_count": r.sample.token_count,
                "gen_time": r.sample.generation_time,
                "script_diversity": r.metrics.script_diversity,
                "char_entropy": r.metrics.char_entropy,
                "word_entropy": r.metrics.word_entropy,
                "code_density": r.metrics.code_fragment_density,
                "coherent_run": r.metrics.longest_coherent_run,
                "embedding_drift": r.metrics.embedding_drift,
                "n_languages": len(r.metrics.language_tags),
            })
        return pd.DataFrame(rows)


def find_decoherence_threshold(
    experiment: DecoherenceExperiment,
    metric: str = "script_diversity",
    threshold_multiplier: float = 2.0,
) -> Dict[str, Optional[float]]:
    """Find the temperature threshold where decoherence begins for each model.

    Returns a dict of model -> threshold temperature (or None if not found).
    """
    df = experiment.to_dataframe()
    thresholds = {}

    for model in df["model"].unique():
        model_df = df[df["model"] == model].sort_values("temperature")
        low_temp = model_df[model_df["temperature"] <= 0.7]
        if low_temp.empty:
            low_temp = model_df.head(3)
        baseline = low_temp[metric].mean()
        baseline_std = low_temp[metric].std()
        threshold_val = baseline + threshold_multiplier * max(baseline_std, 0.1)
        above = model_df[model_df[metric] > threshold_val]
        thresholds[model] = above["temperature"].min() if not above.empty else None

    return thresholds
