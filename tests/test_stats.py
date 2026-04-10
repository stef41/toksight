"""Tests for stats module."""

import pytest
from toksight.stats import token_length_histogram, top_tokens_by_length, vocab_stats


class TestVocabStats:
    def test_basic(self, mock_tokenizer):
        stats = vocab_stats(mock_tokenizer)
        assert stats.vocab_size > 0
        assert stats.n_special_tokens >= 0
        assert stats.avg_token_length > 0
        assert stats.max_token_length > 0

    def test_length_distribution(self, mock_tokenizer):
        stats = vocab_stats(mock_tokenizer)
        assert len(stats.length_distribution) > 0
        # All counts should be positive
        assert all(c > 0 for c in stats.length_distribution.values())

    def test_script_coverage(self, mock_tokenizer):
        stats = vocab_stats(mock_tokenizer)
        assert len(stats.script_coverage) > 0

    def test_single_char_count(self, mock_tokenizer):
        stats = vocab_stats(mock_tokenizer)
        assert stats.n_single_char > 0  # Has ASCII chars

    def test_byte_tokens(self, mock_tokenizer):
        stats = vocab_stats(mock_tokenizer)
        # Many 1-byte tokens in mock vocab
        assert stats.n_byte_tokens > 0


class TestTopTokensByLength:
    def test_basic(self, mock_tokenizer):
        result = top_tokens_by_length(mock_tokenizer, n=5)
        assert len(result) <= 5
        assert len(result) > 0
        # Should be sorted by byte_length descending
        for i in range(len(result) - 1):
            assert result[i]["byte_length"] >= result[i + 1]["byte_length"]

    def test_structure(self, mock_tokenizer):
        result = top_tokens_by_length(mock_tokenizer, n=1)
        entry = result[0]
        assert "token_id" in entry
        assert "text" in entry
        assert "byte_length" in entry
        assert "char_length" in entry


class TestTokenLengthHistogram:
    def test_basic(self, mock_tokenizer):
        hist = token_length_histogram(mock_tokenizer)
        assert len(hist) > 0
        assert all(isinstance(k, int) for k in hist.keys())
        assert all(isinstance(v, int) for v in hist.values())
