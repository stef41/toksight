"""Tests for toksight.training_coverage module."""

from __future__ import annotations

import pytest

from toksight.loader import TokenizerWrapper
from toksight._types import TokenizerBackend
from toksight.training_coverage import (
    CoverageEstimate,
    DomainCoverageResult,
    estimate_coverage,
    domain_coverage,
    format_coverage_report,
)


def _make_tokenizer(vocab: dict[str, int] | None = None) -> TokenizerWrapper:
    """Build a trivial word-level tokenizer for testing."""
    if vocab is None:
        vocab = {
            "hello": 0,
            "world": 1,
            "foo": 2,
            "bar": 3,
            "baz": 4,
            "the": 5,
            "quick": 6,
            "brown": 7,
            "fox": 8,
            "jumps": 9,
        }
    id_to_tok = {v: k for k, v in vocab.items()}

    def encode(text: str) -> list[int]:
        ids = []
        for word in text.lower().split():
            if word in vocab:
                ids.append(vocab[word])
        return ids

    def decode(ids: list[int]) -> str:
        return " ".join(id_to_tok.get(i, "<unk>") for i in ids)

    return TokenizerWrapper(
        name="test-tok",
        backend=TokenizerBackend.CUSTOM,
        encode_fn=encode,
        decode_fn=decode,
        vocab=vocab,
    )


class TestEstimateCoverage:
    def test_basic_coverage(self):
        tok = _make_tokenizer()
        est = estimate_coverage(tok, ["hello world", "foo bar"])
        assert est.vocab_total == 10
        assert est.vocab_used == 4
        assert 0.3 < est.coverage_ratio < 0.5

    def test_full_coverage(self):
        tok = _make_tokenizer()
        text = "hello world foo bar baz the quick brown fox jumps"
        est = estimate_coverage(tok, [text])
        assert est.vocab_used == 10
        assert est.coverage_ratio == 1.0
        assert est.unused_tokens == []

    def test_empty_corpus(self):
        tok = _make_tokenizer()
        est = estimate_coverage(tok, [])
        assert est.vocab_used == 0
        assert est.coverage_ratio == 0.0
        assert len(est.unused_tokens) == 10

    def test_unused_tokens_listed(self):
        tok = _make_tokenizer()
        est = estimate_coverage(tok, ["hello world"])
        assert est.vocab_used == 2
        assert "foo" in est.unused_tokens
        assert "hello" not in est.unused_tokens

    def test_most_frequent(self):
        tok = _make_tokenizer()
        est = estimate_coverage(tok, ["hello hello hello world"])
        freq = dict(est.most_frequent)
        assert freq["hello"] == 3
        assert freq["world"] == 1

    def test_rare_tokens(self):
        tok = _make_tokenizer()
        est = estimate_coverage(tok, ["hello hello hello world"])
        rare = dict(est.rare_tokens)
        assert rare["world"] == 1

    def test_top_k_parameter(self):
        tok = _make_tokenizer()
        text = "hello world foo bar baz the quick brown fox jumps"
        est = estimate_coverage(tok, [text], top_k=3)
        assert len(est.most_frequent) <= 3
        assert len(est.rare_tokens) <= 3

    def test_coverage_estimate_dataclass(self):
        est = CoverageEstimate(
            vocab_used=5,
            vocab_total=10,
            coverage_ratio=0.5,
            unused_tokens=["a"],
            rare_tokens=[("b", 1)],
            most_frequent=[("c", 100)],
        )
        assert est.vocab_used == 5
        assert est.coverage_ratio == 0.5


class TestDomainCoverage:
    def test_basic_domain_comparison(self):
        tok = _make_tokenizer()
        domain = ["foo bar foo"]
        reference = ["hello world hello"]
        result = domain_coverage(tok, domain, reference)
        assert isinstance(result, DomainCoverageResult)
        assert result.domain_vocab_used == 2
        assert result.reference_vocab_used == 2

    def test_shared_tokens(self):
        tok = _make_tokenizer()
        domain = ["hello foo"]
        reference = ["hello bar"]
        result = domain_coverage(tok, domain, reference)
        assert result.shared_tokens == 1  # "hello"

    def test_domain_only_tokens(self):
        tok = _make_tokenizer()
        domain = ["foo bar"]
        reference = ["hello world"]
        result = domain_coverage(tok, domain, reference)
        assert "foo" in result.domain_only_tokens
        assert "hello" in result.reference_only_tokens

    def test_overrepresented(self):
        tok = _make_tokenizer()
        # "hello" appears much more in domain than reference
        domain = ["hello hello hello hello world"]
        reference = ["hello world world world world"]
        result = domain_coverage(tok, domain, reference)
        over_tokens = [t for t, _ in result.overrepresented]
        under_tokens = [t for t, _ in result.underrepresented]
        assert "hello" in over_tokens
        assert "world" in under_tokens

    def test_empty_domain(self):
        tok = _make_tokenizer()
        result = domain_coverage(tok, [], ["hello world"])
        assert result.domain_vocab_used == 0


class TestFormatReport:
    def test_format_contains_header(self):
        est = CoverageEstimate(
            vocab_used=5,
            vocab_total=10,
            coverage_ratio=0.5,
            unused_tokens=["x"],
            rare_tokens=[("y", 1)],
            most_frequent=[("z", 100)],
        )
        report = format_coverage_report(est)
        assert "Vocabulary Coverage Report" in report
        assert "5" in report
        assert "10" in report

    def test_format_shows_tokens(self):
        est = CoverageEstimate(
            vocab_used=2,
            vocab_total=10,
            coverage_ratio=0.2,
            unused_tokens=[],
            rare_tokens=[("rare_tok", 1)],
            most_frequent=[("freq_tok", 99)],
        )
        report = format_coverage_report(est)
        assert "rare_tok" in report
        assert "freq_tok" in report
        assert "Most Frequent" in report
        assert "Rare Tokens" in report
