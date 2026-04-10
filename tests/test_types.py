"""Tests for _types module."""

import pytest
from toksight._types import (
    AuditFinding,
    AuditResult,
    CompareResult,
    CompressionStats,
    CostEstimate,
    CoverageResult,
    TokenizerBackend,
    TokenizerInfo,
    TokenSpan,
    ToksightError,
    VocabStats,
)


class TestTokenizerBackend:
    def test_values(self):
        assert TokenizerBackend.TIKTOKEN.value == "tiktoken"
        assert TokenizerBackend.HUGGINGFACE.value == "huggingface"
        assert TokenizerBackend.SENTENCEPIECE.value == "sentencepiece"
        assert TokenizerBackend.CUSTOM.value == "custom"

    def test_is_string(self):
        assert isinstance(TokenizerBackend.TIKTOKEN, str)


class TestTokenizerInfo:
    def test_basic(self):
        info = TokenizerInfo(name="test", backend=TokenizerBackend.CUSTOM, vocab_size=1000)
        assert info.name == "test"
        assert info.vocab_size == 1000
        assert info.special_tokens == []
        assert info.metadata == {}


class TestTokenSpan:
    def test_basic(self):
        span = TokenSpan(token_id=1, text="hello", byte_length=5, char_offset=0, char_length=5)
        assert span.token_id == 1
        assert span.text == "hello"
        assert span.byte_length == 5

    def test_bytes_per_char(self):
        span = TokenSpan(token_id=1, text="hello", byte_length=5, char_offset=0, char_length=5)
        assert span.bytes_per_char == 1.0

    def test_bytes_per_char_unicode(self):
        span = TokenSpan(token_id=1, text="日", byte_length=3, char_offset=0, char_length=1)
        assert span.bytes_per_char == 3.0

    def test_bytes_per_char_zero_length(self):
        span = TokenSpan(token_id=1, text="", byte_length=0, char_offset=0, char_length=0)
        assert span.bytes_per_char == 0.0


class TestCompressionStats:
    def test_compression_ratio(self):
        stats = CompressionStats(
            total_chars=100, total_bytes=100, total_tokens=25,
            bytes_per_token=4.0, chars_per_token=4.0,
        )
        assert stats.compression_ratio == 4.0

    def test_compression_ratio_zero(self):
        stats = CompressionStats(
            total_chars=0, total_bytes=0, total_tokens=0,
            bytes_per_token=0.0, chars_per_token=0.0,
        )
        assert stats.compression_ratio == 0.0


class TestCoverageResult:
    def test_basic(self):
        result = CoverageResult(
            total_codepoints_tested=100,
            codepoints_covered=95,
            coverage_ratio=0.95,
        )
        assert result.uncovered_count == 5

    def test_full_coverage(self):
        result = CoverageResult(
            total_codepoints_tested=50,
            codepoints_covered=50,
            coverage_ratio=1.0,
        )
        assert result.uncovered_count == 0


class TestCompareResult:
    def test_basic(self):
        result = CompareResult(
            tokenizer_a="a", tokenizer_b="b",
            vocab_overlap=500, vocab_only_a=200, vocab_only_b=300,
            jaccard_similarity=0.5,
        )
        assert result.overlap_ratio_a == pytest.approx(500 / 700)
        assert result.overlap_ratio_b == pytest.approx(500 / 800)

    def test_zero_vocab(self):
        result = CompareResult(
            tokenizer_a="a", tokenizer_b="b",
            vocab_overlap=0, vocab_only_a=0, vocab_only_b=0,
            jaccard_similarity=0.0,
        )
        assert result.overlap_ratio_a == 0.0
        assert result.overlap_ratio_b == 0.0


class TestAuditResult:
    def test_empty(self):
        result = AuditResult(tokenizer_name="test")
        assert result.n_critical == 0
        assert result.n_warnings == 0
        assert result.by_category == {}

    def test_counts(self):
        findings = [
            AuditFinding("glitch", "critical", 1, "a", "test"),
            AuditFinding("degenerate", "warning", 2, "b", "test"),
            AuditFinding("glitch", "warning", 3, "c", "test"),
        ]
        result = AuditResult(tokenizer_name="test", findings=findings)
        assert result.n_critical == 1
        assert result.n_warnings == 2
        assert len(result.by_category["glitch"]) == 2
        assert len(result.by_category["degenerate"]) == 1


class TestCostEstimate:
    def test_basic(self):
        est = CostEstimate(corpus_chars=1000)
        assert est.corpus_chars == 1000
        assert est.estimates == {}


class TestVocabStats:
    def test_basic(self):
        stats = VocabStats(
            vocab_size=50000, n_special_tokens=5, n_byte_tokens=256,
            n_single_char=500, n_multiword=100, avg_token_length=4.5,
            max_token_length=64,
        )
        assert stats.vocab_size == 50000


class TestToksightError:
    def test_is_exception(self):
        assert issubclass(ToksightError, Exception)
        err = ToksightError("test")
        assert str(err) == "test"
