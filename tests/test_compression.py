"""Tests for compression module."""

import pytest
from toksight.compression import (
    compare_compression,
    compression_by_language,
    compute_compression,
    fertility_analysis,
)


class TestComputeCompression:
    def test_basic(self, mock_tokenizer):
        texts = ["hello world", "the test is a test"]
        stats = compute_compression(mock_tokenizer, texts)
        assert stats.total_chars > 0
        assert stats.total_bytes > 0
        assert stats.total_tokens > 0
        assert stats.bytes_per_token > 0
        assert stats.chars_per_token > 0

    def test_empty_corpus(self, mock_tokenizer):
        stats = compute_compression(mock_tokenizer, [])
        assert stats.total_tokens == 0
        assert stats.bytes_per_token == 0.0

    def test_empty_strings(self, mock_tokenizer):
        stats = compute_compression(mock_tokenizer, ["", ""])
        assert stats.total_tokens == 0

    def test_single_text(self, mock_tokenizer):
        stats = compute_compression(mock_tokenizer, ["hello"])
        assert stats.total_tokens > 0
        assert stats.total_chars == 5

    def test_fertility(self, mock_tokenizer):
        texts = ["hello world is a test"]
        stats = compute_compression(mock_tokenizer, texts)
        assert stats.fertility > 0  # tokens per word

    def test_unicode_text(self, mock_tokenizer):
        texts = ["日本語"]
        stats = compute_compression(mock_tokenizer, texts)
        assert stats.total_bytes > stats.total_chars  # UTF-8 multibyte

    def test_compression_ratio(self, mock_tokenizer):
        texts = ["hello world test"]
        stats = compute_compression(mock_tokenizer, texts)
        assert stats.compression_ratio > 0


class TestCompressionByLanguage:
    def test_basic(self, mock_tokenizer):
        texts_by_lang = {
            "english": ["hello world", "the test"],
            "short": ["a", "b"],
        }
        results = compression_by_language(mock_tokenizer, texts_by_lang)
        assert "english" in results
        assert "short" in results
        assert results["english"].total_tokens > 0

    def test_empty_lang(self, mock_tokenizer):
        results = compression_by_language(mock_tokenizer, {"empty": []})
        assert results["empty"].total_tokens == 0


class TestCompareCompression:
    def test_basic(self, mock_tokenizer, mock_tokenizer_b):
        texts = ["hello world"]
        results = compare_compression([mock_tokenizer, mock_tokenizer_b], texts)
        assert "mock" in results
        assert "mock_b" in results
        assert results["mock"].total_tokens > 0


class TestFertilityAnalysis:
    def test_basic(self, mock_tokenizer):
        texts = ["hello world is a test", "the test is hello"]
        result = fertility_analysis(mock_tokenizer, texts)
        assert result["count"] == 2.0
        assert result["mean"] > 0
        assert result["min"] > 0
        assert result["max"] >= result["min"]

    def test_empty_corpus(self, mock_tokenizer):
        result = fertility_analysis(mock_tokenizer, [])
        assert result["count"] == 0.0

    def test_whitespace_only(self, mock_tokenizer):
        result = fertility_analysis(mock_tokenizer, ["   ", "  "])
        assert result["count"] == 0.0  # empty after strip
