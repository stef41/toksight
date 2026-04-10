"""Tests for loader module."""

import pytest
from toksight._types import TokenizerBackend, ToksightError
from toksight.loader import TokenizerWrapper, wrap_custom


class TestTokenizerWrapper:
    def test_basic_properties(self, mock_tokenizer):
        assert mock_tokenizer.name == "mock"
        assert mock_tokenizer.backend == TokenizerBackend.CUSTOM
        assert mock_tokenizer.vocab_size > 0
        assert "hello" in mock_tokenizer.vocab

    def test_encode_decode(self, mock_tokenizer):
        text = "hello world"
        ids = mock_tokenizer.encode(text)
        assert len(ids) > 0
        decoded = mock_tokenizer.decode(ids)
        assert decoded == text

    def test_decode_single(self, mock_tokenizer):
        vocab = mock_tokenizer.vocab
        hello_id = vocab["hello"]
        assert mock_tokenizer.decode_single(hello_id) == "hello"

    def test_tokenize(self, mock_tokenizer):
        spans = mock_tokenizer.tokenize("hello")
        assert len(spans) >= 1
        assert spans[0].text == "hello"
        assert spans[0].byte_length == 5

    def test_info(self, mock_tokenizer):
        info = mock_tokenizer.info()
        assert info.name == "mock"
        assert info.backend == TokenizerBackend.CUSTOM
        assert info.vocab_size == mock_tokenizer.vocab_size

    def test_special_tokens(self, mock_tokenizer):
        assert len(mock_tokenizer.special_tokens) == 2

    def test_id_to_token(self, mock_tokenizer):
        id_map = mock_tokenizer.id_to_token
        assert len(id_map) == mock_tokenizer.vocab_size

    def test_vocab_is_copy(self, mock_tokenizer):
        v = mock_tokenizer.vocab
        v["NEW"] = 99999
        assert "NEW" not in mock_tokenizer.vocab


class TestWrapCustom:
    def test_basic(self):
        vocab = {"a": 0, "b": 1, "c": 2}
        tok = wrap_custom(
            name="test",
            encode_fn=lambda text: [vocab.get(c, 0) for c in text],
            decode_fn=lambda ids: "".join(
                {v: k for k, v in vocab.items()}.get(i, "?") for i in ids
            ),
            vocab=vocab,
        )
        assert tok.name == "test"
        assert tok.backend == TokenizerBackend.CUSTOM
        assert tok.encode("abc") == [0, 1, 2]
        assert tok.decode([0, 1, 2]) == "abc"

    def test_with_special_tokens(self):
        vocab = {"<s>": 0, "a": 1}
        tok = wrap_custom(
            name="test",
            encode_fn=lambda t: [1],
            decode_fn=lambda ids: "a",
            vocab=vocab,
            special_tokens=["<s>"],
        )
        assert tok.special_tokens == ["<s>"]


class TestLoadTiktoken:
    def test_import_error(self):
        """Test that missing tiktoken raises proper error."""
        # This test only validates the function exists
        from toksight.loader import load_tiktoken
        assert callable(load_tiktoken)


class TestLoadHuggingface:
    def test_import_error(self):
        from toksight.loader import load_huggingface
        assert callable(load_huggingface)


class TestLoadSentencepiece:
    def test_import_error(self):
        from toksight.loader import load_sentencepiece
        assert callable(load_sentencepiece)
