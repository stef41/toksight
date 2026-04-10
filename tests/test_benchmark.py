"""Tests for the tokenizer benchmark suite."""

import pytest

from toksight.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    TokenizerBenchmark,
    format_benchmark_report,
    generate_benchmark_texts,
)


# --- helpers ----------------------------------------------------------------

def _simple_encode(text: str) -> list[int]:
    """Encode each character as its ordinal."""
    return [ord(c) for c in text]


def _simple_decode(tokens: list[int]) -> str:
    """Decode ordinals back to characters."""
    return "".join(chr(t) for t in tokens)


def _word_encode(text: str) -> list[int]:
    """Encode by splitting on spaces."""
    return [hash(w) % 50000 for w in text.split()]


def _word_decode(tokens: list[int]) -> str:
    return " ".join(f"t{t}" for t in tokens)


# --- BenchmarkConfig -------------------------------------------------------

class TestBenchmarkConfig:
    def test_defaults(self):
        cfg = BenchmarkConfig()
        assert cfg.iterations == 1000
        assert cfg.warmup == 100
        assert cfg.text_lengths == [100, 500, 1000, 5000]

    def test_custom(self):
        cfg = BenchmarkConfig(iterations=10, warmup=2, text_lengths=[50])
        assert cfg.iterations == 10
        assert cfg.warmup == 2
        assert cfg.text_lengths == [50]


# --- BenchmarkResult -------------------------------------------------------

class TestBenchmarkResult:
    def test_fields(self):
        r = BenchmarkResult(
            tokenizer_name="test",
            text_length=100,
            encode_time_ms=1.5,
            decode_time_ms=0.8,
            tokens_per_second=1000.0,
            memory_chars_per_token=3.5,
        )
        assert r.tokenizer_name == "test"
        assert r.text_length == 100
        assert r.encode_time_ms == 1.5
        assert r.decode_time_ms == 0.8


# --- generate_benchmark_texts -----------------------------------------------

class TestGenerateBenchmarkTexts:
    def test_correct_lengths(self):
        lengths = [50, 200, 500]
        texts = generate_benchmark_texts(lengths)
        assert len(texts) == 3
        for text, expected in zip(texts, lengths):
            assert len(text) == expected

    def test_deterministic(self):
        a = generate_benchmark_texts([100, 200])
        b = generate_benchmark_texts([100, 200])
        assert a == b

    def test_single_length(self):
        texts = generate_benchmark_texts([10])
        assert len(texts) == 1
        assert len(texts[0]) == 10

    def test_large_length(self):
        texts = generate_benchmark_texts([10000])
        assert len(texts[0]) == 10000


# --- TokenizerBenchmark.benchmark_encode ------------------------------------

class TestBenchmarkEncode:
    def test_returns_results(self):
        cfg = BenchmarkConfig(iterations=5, warmup=1, text_lengths=[20])
        bench = TokenizerBenchmark(cfg)
        texts = generate_benchmark_texts(cfg.text_lengths)
        results = bench.benchmark_encode(_simple_encode, texts, name="char")
        assert len(results) == 1
        assert results[0].tokenizer_name == "char"
        assert results[0].encode_time_ms > 0
        assert results[0].tokens_per_second > 0

    def test_multiple_texts(self):
        cfg = BenchmarkConfig(iterations=3, warmup=1, text_lengths=[10, 50])
        bench = TokenizerBenchmark(cfg)
        texts = generate_benchmark_texts(cfg.text_lengths)
        results = bench.benchmark_encode(_simple_encode, texts)
        assert len(results) == 2

    def test_chars_per_token(self):
        cfg = BenchmarkConfig(iterations=3, warmup=1)
        bench = TokenizerBenchmark(cfg)
        results = bench.benchmark_encode(_simple_encode, ["hello"])
        # char-based: 1 char per token
        assert results[0].memory_chars_per_token == 1.0


# --- TokenizerBenchmark.benchmark_decode ------------------------------------

class TestBenchmarkDecode:
    def test_returns_results(self):
        cfg = BenchmarkConfig(iterations=5, warmup=1)
        bench = TokenizerBenchmark(cfg)
        token_lists = [[72, 101, 108, 108, 111]]
        results = bench.benchmark_decode(_simple_decode, token_lists, name="char")
        assert len(results) == 1
        assert results[0].decode_time_ms > 0

    def test_multiple_lists(self):
        cfg = BenchmarkConfig(iterations=3, warmup=1)
        bench = TokenizerBenchmark(cfg)
        token_lists = [[65, 66], [67, 68, 69, 70]]
        results = bench.benchmark_decode(_simple_decode, token_lists)
        assert len(results) == 2


# --- TokenizerBenchmark.benchmark_roundtrip ---------------------------------

class TestBenchmarkRoundtrip:
    def test_roundtrip(self):
        cfg = BenchmarkConfig(iterations=5, warmup=1)
        bench = TokenizerBenchmark(cfg)
        results = bench.benchmark_roundtrip(
            _simple_encode, _simple_decode, ["hello world"], name="char",
        )
        assert len(results) == 1
        r = results[0]
        assert r.encode_time_ms > 0
        assert r.decode_time_ms > 0
        assert r.tokens_per_second > 0


# --- TokenizerBenchmark.compare --------------------------------------------

class TestCompare:
    def test_compare_same(self):
        cfg = BenchmarkConfig(iterations=5, warmup=1, text_lengths=[20])
        bench = TokenizerBenchmark(cfg)
        texts = generate_benchmark_texts(cfg.text_lengths)
        ra = bench.benchmark_encode(_simple_encode, texts, name="a")
        rb = bench.benchmark_encode(_simple_encode, texts, name="b")
        cmp = bench.compare(ra, rb)
        assert "comparisons" in cmp
        assert "summary" in cmp
        assert cmp["tokenizer_a"] == "a"
        assert cmp["tokenizer_b"] == "b"
        assert len(cmp["comparisons"]) == 1

    def test_compare_different_tokenizers(self):
        cfg = BenchmarkConfig(iterations=5, warmup=1, text_lengths=[50])
        bench = TokenizerBenchmark(cfg)
        texts = generate_benchmark_texts(cfg.text_lengths)
        ra = bench.benchmark_encode(_simple_encode, texts, name="char")
        rb = bench.benchmark_encode(_word_encode, texts, name="word")
        cmp = bench.compare(ra, rb)
        assert cmp["summary"]["avg_encode_speedup"] > 0
        # Word tokenizer produces fewer tokens → different chars_per_token
        c = cmp["comparisons"][0]
        assert c["chars_per_token_a"] != c["chars_per_token_b"]


# --- format_benchmark_report ------------------------------------------------

class TestFormatReport:
    def test_empty(self):
        assert format_benchmark_report([]) == "No benchmark results."

    def test_has_header(self):
        r = BenchmarkResult("t", 100, 1.0, 0.5, 5000.0, 3.0)
        report = format_benchmark_report([r])
        assert "Tokenizer" in report
        assert "Enc(ms)" in report
        assert "t" in report

    def test_multiple_rows(self):
        results = [
            BenchmarkResult("a", 100, 1.0, 0.5, 5000.0, 3.0),
            BenchmarkResult("b", 200, 2.0, 1.0, 4000.0, 2.5),
        ]
        report = format_benchmark_report(results)
        lines = report.strip().split("\n")
        # header + separator + 2 data rows
        assert len(lines) == 4


# --- default config ---------------------------------------------------------

class TestDefaultConfig:
    def test_default_config_used(self):
        bench = TokenizerBenchmark()
        assert bench.config.iterations == 1000
        assert bench.config.warmup == 100
