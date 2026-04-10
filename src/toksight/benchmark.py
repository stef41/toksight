"""Tokenizer benchmark suite for toksight."""

from __future__ import annotations

import random
import string
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    iterations: int = 1000
    warmup: int = 100
    text_lengths: List[int] = field(default_factory=lambda: [100, 500, 1000, 5000])


@dataclass
class BenchmarkResult:
    """Result of a single benchmark measurement."""

    tokenizer_name: str
    text_length: int
    encode_time_ms: float
    decode_time_ms: float
    tokens_per_second: float
    memory_chars_per_token: float


def generate_benchmark_texts(lengths: List[int]) -> List[str]:
    """Generate varied test texts of specified lengths.

    Produces texts mixing words, punctuation, numbers, and whitespace
    to simulate realistic tokenizer workloads.
    """
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "transformer", "tokenizer", "benchmark",
        "neural", "network", "attention", "embedding", "gradient",
        "optimization", "parameter", "inference", "training", "dataset",
        "vocabulary", "encoder", "decoder", "sequence", "prediction",
        "Hello,", "world!", "How", "are", "you?", "I'm", "fine.",
        "2024", "42", "3.14", "(test)", "[data]", "{config}",
        "multi-line\ntext", "tab\there", "special: @#$%",
    ]
    rng = random.Random(42)
    texts: List[str] = []
    for length in lengths:
        parts: List[str] = []
        current = 0
        while current < length:
            word = rng.choice(words)
            parts.append(word)
            current += len(word) + 1  # +1 for space
        text = " ".join(parts)[:length]
        # Pad if needed
        if len(text) < length:
            text += " " * (length - len(text))
        texts.append(text)
    return texts


class TokenizerBenchmark:
    """Benchmark suite for tokenizer encode/decode performance."""

    def __init__(self, config: Optional[BenchmarkConfig] = None) -> None:
        self.config = config or BenchmarkConfig()

    def benchmark_encode(
        self,
        tokenize_fn: Callable[[str], List[int]],
        texts: List[str],
        name: str = "custom",
    ) -> List[BenchmarkResult]:
        """Benchmark encoding speed across texts."""
        results: List[BenchmarkResult] = []
        for text in texts:
            # Warmup
            for _ in range(self.config.warmup):
                tokenize_fn(text)

            # Timed run
            start = time.perf_counter()
            tokens: List[int] = []
            for _ in range(self.config.iterations):
                tokens = tokenize_fn(text)
            elapsed = time.perf_counter() - start

            encode_ms = (elapsed / self.config.iterations) * 1000
            total_tokens = len(tokens) * self.config.iterations
            tps = total_tokens / elapsed if elapsed > 0 else 0.0
            cpt = len(text) / len(tokens) if tokens else 0.0

            results.append(BenchmarkResult(
                tokenizer_name=name,
                text_length=len(text),
                encode_time_ms=encode_ms,
                decode_time_ms=0.0,
                tokens_per_second=tps,
                memory_chars_per_token=cpt,
            ))
        return results

    def benchmark_decode(
        self,
        decode_fn: Callable[[List[int]], str],
        token_lists: List[List[int]],
        name: str = "custom",
    ) -> List[BenchmarkResult]:
        """Benchmark decoding speed across token lists."""
        results: List[BenchmarkResult] = []
        for tokens in token_lists:
            # Warmup
            for _ in range(self.config.warmup):
                decode_fn(tokens)

            # Timed run
            start = time.perf_counter()
            decoded = ""
            for _ in range(self.config.iterations):
                decoded = decode_fn(tokens)
            elapsed = time.perf_counter() - start

            decode_ms = (elapsed / self.config.iterations) * 1000
            tps = len(tokens) * self.config.iterations / elapsed if elapsed > 0 else 0.0
            cpt = len(decoded) / len(tokens) if tokens else 0.0

            results.append(BenchmarkResult(
                tokenizer_name=name,
                text_length=len(decoded),
                encode_time_ms=0.0,
                decode_time_ms=decode_ms,
                tokens_per_second=tps,
                memory_chars_per_token=cpt,
            ))
        return results

    def benchmark_roundtrip(
        self,
        tokenize_fn: Callable[[str], List[int]],
        decode_fn: Callable[[List[int]], str],
        texts: List[str],
        name: str = "custom",
    ) -> List[BenchmarkResult]:
        """Benchmark full encode→decode roundtrip."""
        results: List[BenchmarkResult] = []
        for text in texts:
            # Warmup
            for _ in range(self.config.warmup):
                decode_fn(tokenize_fn(text))

            # Timed run
            start_enc = time.perf_counter()
            tokens: List[int] = []
            for _ in range(self.config.iterations):
                tokens = tokenize_fn(text)
            mid = time.perf_counter()
            for _ in range(self.config.iterations):
                decode_fn(tokens)
            end = time.perf_counter()

            encode_ms = ((mid - start_enc) / self.config.iterations) * 1000
            decode_ms = ((end - mid) / self.config.iterations) * 1000
            total_elapsed = end - start_enc
            tps = len(tokens) * self.config.iterations / total_elapsed if total_elapsed > 0 else 0.0
            cpt = len(text) / len(tokens) if tokens else 0.0

            results.append(BenchmarkResult(
                tokenizer_name=name,
                text_length=len(text),
                encode_time_ms=encode_ms,
                decode_time_ms=decode_ms,
                tokens_per_second=tps,
                memory_chars_per_token=cpt,
            ))
        return results

    def compare(
        self,
        results_a: List[BenchmarkResult],
        results_b: List[BenchmarkResult],
    ) -> Dict[str, Any]:
        """Compare two sets of benchmark results.

        Returns a dict with per-text-length comparisons and an overall summary.
        """
        comparisons: List[Dict[str, Any]] = []
        for ra, rb in zip(results_a, results_b):
            enc_speedup = ra.encode_time_ms / rb.encode_time_ms if rb.encode_time_ms > 0 else 0.0
            dec_speedup = ra.decode_time_ms / rb.decode_time_ms if rb.decode_time_ms > 0 else 0.0
            tps_ratio = rb.tokens_per_second / ra.tokens_per_second if ra.tokens_per_second > 0 else 0.0
            comparisons.append({
                "text_length": ra.text_length,
                "tokenizer_a": ra.tokenizer_name,
                "tokenizer_b": rb.tokenizer_name,
                "encode_speedup": enc_speedup,
                "decode_speedup": dec_speedup,
                "tokens_per_second_ratio": tps_ratio,
                "chars_per_token_a": ra.memory_chars_per_token,
                "chars_per_token_b": rb.memory_chars_per_token,
            })

        avg_enc = sum(c["encode_speedup"] for c in comparisons) / len(comparisons) if comparisons else 0.0
        avg_dec = sum(c["decode_speedup"] for c in comparisons) / len(comparisons) if comparisons else 0.0
        avg_tps = sum(c["tokens_per_second_ratio"] for c in comparisons) / len(comparisons) if comparisons else 0.0

        name_a = results_a[0].tokenizer_name if results_a else "a"
        name_b = results_b[0].tokenizer_name if results_b else "b"

        return {
            "tokenizer_a": name_a,
            "tokenizer_b": name_b,
            "comparisons": comparisons,
            "summary": {
                "avg_encode_speedup": avg_enc,
                "avg_decode_speedup": avg_dec,
                "avg_tokens_per_second_ratio": avg_tps,
                "faster_encoder": name_b if avg_enc > 1.0 else name_a,
                "faster_decoder": name_b if avg_dec > 1.0 else name_a,
            },
        }


def format_benchmark_report(results: List[BenchmarkResult]) -> str:
    """Format benchmark results as a readable table."""
    if not results:
        return "No benchmark results."

    header = (
        f"{'Tokenizer':<20} {'TextLen':>8} {'Enc(ms)':>10} "
        f"{'Dec(ms)':>10} {'Tok/s':>12} {'Chars/Tok':>10}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        lines.append(
            f"{r.tokenizer_name:<20} {r.text_length:>8} "
            f"{r.encode_time_ms:>10.3f} {r.decode_time_ms:>10.3f} "
            f"{r.tokens_per_second:>12.0f} {r.memory_chars_per_token:>10.2f}"
        )
    return "\n".join(lines)
