"""Shared test fixtures — mock tokenizer for testing without external deps."""

import pytest
from toksight.loader import TokenizerWrapper, wrap_custom


def _simple_vocab():
    """A small character+word-level vocab for testing."""
    vocab = {}
    tid = 0

    # Single byte/char tokens
    for i in range(256):
        ch = chr(i) if 32 <= i < 127 else f"<byte_{i:02x}>"
        vocab[ch] = tid
        tid += 1

    # Word tokens
    words = [
        "hello", "world", "the", "is", "a", "an",
        "test", "ing", "ed", "ly", " the", " is", " a",
        "日本", "語", "한국", "어",
        "  ", "\t", "\n",
        "aaaa",  # repeated char token
    ]
    for w in words:
        if w not in vocab:
            vocab[w] = tid
            tid += 1

    return vocab


def _make_mock_tokenizer(name="mock") -> TokenizerWrapper:
    """Create a simple character-level mock tokenizer."""
    vocab = _simple_vocab()
    reverse = {v: k for k, v in vocab.items()}

    # Prefer longer tokens (word > char)
    sorted_tokens = sorted(vocab.keys(), key=len, reverse=True)

    def encode(text: str) -> list:
        ids = []
        i = 0
        while i < len(text):
            matched = False
            for token in sorted_tokens:
                if text[i:i+len(token)] == token:
                    ids.append(vocab[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                # Fallback: single char
                ch = text[i]
                if ch in vocab:
                    ids.append(vocab[ch])
                else:
                    ids.append(vocab.get("<byte_3f>", 0))  # '?'
                i += 1
        return ids

    def decode(ids: list) -> str:
        parts = []
        for tid in ids:
            if tid in reverse:
                parts.append(reverse[tid])
            else:
                parts.append("?")
        return "".join(parts)

    return wrap_custom(
        name=name,
        encode_fn=encode,
        decode_fn=decode,
        vocab=vocab,
        special_tokens=["<byte_00>", "<byte_01>"],
    )


@pytest.fixture
def mock_tokenizer():
    return _make_mock_tokenizer()


@pytest.fixture
def mock_tokenizer_b():
    """Second tokenizer with slightly different vocab for comparison tests."""
    tok = _make_mock_tokenizer(name="mock_b")
    # Add some unique tokens
    extra = {"python": len(tok._vocab), "rust": len(tok._vocab) + 1}
    tok._vocab.update(extra)
    tok._id_to_token.update({v: k for k, v in extra.items()})
    return tok
