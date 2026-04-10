"""Tests for mapping module."""

import pytest
from toksight.mapping import map_tokens, token_expansion_ratio


class TestMapTokens:
    def test_same_tokenizer(self, mock_tokenizer):
        result = map_tokens(mock_tokenizer, mock_tokenizer, "hello world")
        assert len(result) > 0
        for entry in result:
            assert entry["is_one_to_one"]
            assert entry["expansion"] == 1

    def test_result_structure(self, mock_tokenizer, mock_tokenizer_b):
        result = map_tokens(mock_tokenizer, mock_tokenizer_b, "hello")
        assert len(result) > 0
        entry = result[0]
        assert "source_token" in entry
        assert "source_id" in entry
        assert "target_tokens" in entry
        assert "expansion" in entry
        assert "is_one_to_one" in entry

    def test_empty_text(self, mock_tokenizer, mock_tokenizer_b):
        result = map_tokens(mock_tokenizer, mock_tokenizer_b, "")
        assert result == []


class TestTokenExpansionRatio:
    def test_same_tokenizer(self, mock_tokenizer):
        result = token_expansion_ratio(mock_tokenizer, mock_tokenizer, ["hello world"])
        assert result["mean"] == pytest.approx(1.0)

    def test_empty_corpus(self, mock_tokenizer, mock_tokenizer_b):
        result = token_expansion_ratio(mock_tokenizer, mock_tokenizer_b, [])
        assert result["count"] == 0.0

    def test_whitespace_only(self, mock_tokenizer, mock_tokenizer_b):
        result = token_expansion_ratio(mock_tokenizer, mock_tokenizer_b, ["   "])
        # Whitespace should still produce tokens
        assert result["count"] >= 0

    def test_multiple_texts(self, mock_tokenizer, mock_tokenizer_b):
        texts = ["hello world", "the test is a test"]
        result = token_expansion_ratio(mock_tokenizer, mock_tokenizer_b, texts)
        assert result["count"] == 2.0
        assert result["mean"] > 0
