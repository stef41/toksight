"""Tests for compare module."""

import pytest
from toksight.compare import (
    boundary_alignment,
    compare_on_corpus,
    compare_vocabularies,
    fragmentation_map,
)


class TestCompareVocabularies:
    def test_same_tokenizer(self, mock_tokenizer):
        result = compare_vocabularies(mock_tokenizer, mock_tokenizer)
        assert result.vocab_overlap == mock_tokenizer.vocab_size
        assert result.vocab_only_a == 0
        assert result.vocab_only_b == 0
        assert result.jaccard_similarity == 1.0

    def test_different_tokenizers(self, mock_tokenizer, mock_tokenizer_b):
        result = compare_vocabularies(mock_tokenizer, mock_tokenizer_b)
        assert result.vocab_overlap > 0
        assert result.vocab_only_b > 0  # mock_b has extra tokens
        assert 0 < result.jaccard_similarity < 1.0

    def test_names(self, mock_tokenizer, mock_tokenizer_b):
        result = compare_vocabularies(mock_tokenizer, mock_tokenizer_b)
        assert result.tokenizer_a == "mock"
        assert result.tokenizer_b == "mock_b"


class TestCompareOnCorpus:
    def test_basic(self, mock_tokenizer, mock_tokenizer_b):
        texts = ["hello world", "the test"]
        result = compare_on_corpus(mock_tokenizer, mock_tokenizer_b, texts)
        assert result.compression_a is not None
        assert result.compression_b is not None
        assert result.compression_a.total_tokens > 0

    def test_boundary_agreement(self, mock_tokenizer, mock_tokenizer_b):
        texts = ["hello"]
        result = compare_on_corpus(mock_tokenizer, mock_tokenizer_b, texts)
        assert 0.0 <= result.boundary_agreement <= 1.0

    def test_empty_corpus(self, mock_tokenizer, mock_tokenizer_b):
        result = compare_on_corpus(mock_tokenizer, mock_tokenizer_b, [])
        assert result.boundary_agreement == 0.0

    def test_same_tokenizer(self, mock_tokenizer):
        texts = ["hello world"]
        result = compare_on_corpus(mock_tokenizer, mock_tokenizer, texts)
        assert result.boundary_agreement == 1.0


class TestBoundaryAlignment:
    def test_same_tokenizer(self, mock_tokenizer):
        result = boundary_alignment(mock_tokenizer, mock_tokenizer, "hello world")
        assert result["agreement"] == 1.0
        assert len(result["only_a"]) == 0
        assert len(result["only_b"]) == 0

    def test_tokens_included(self, mock_tokenizer):
        result = boundary_alignment(mock_tokenizer, mock_tokenizer, "hello")
        assert "tokens_a" in result
        assert "tokens_b" in result
        assert len(result["tokens_a"]) > 0

    def test_text_length(self, mock_tokenizer):
        result = boundary_alignment(mock_tokenizer, mock_tokenizer, "abc")
        assert result["text_length"] == 3


class TestFragmentationMap:
    def test_basic(self, mock_tokenizer, mock_tokenizer_b):
        entries = fragmentation_map(mock_tokenizer, mock_tokenizer_b, "hello")
        assert len(entries) > 0
        assert "token_a" in entries[0]
        assert "token_b" in entries[0]
        assert "aligned" in entries[0]

    def test_same_tokenizer(self, mock_tokenizer):
        entries = fragmentation_map(mock_tokenizer, mock_tokenizer, "hello")
        assert all(e["aligned"] for e in entries)

    def test_empty_text(self, mock_tokenizer, mock_tokenizer_b):
        entries = fragmentation_map(mock_tokenizer, mock_tokenizer_b, "")
        assert entries == []
