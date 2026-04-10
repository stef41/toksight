"""Tests for audit module."""

import pytest
from toksight.audit import audit, find_degenerate_tokens, find_glitch_tokens
from toksight.loader import wrap_custom


class TestAudit:
    def test_basic(self, mock_tokenizer):
        result = audit(mock_tokenizer)
        assert result.tokenizer_name == "mock"
        assert isinstance(result.findings, list)

    def test_max_tokens(self, mock_tokenizer):
        result = audit(mock_tokenizer, max_tokens=10)
        # Limited audit should still work
        assert result.tokenizer_name == "mock"

    def test_finds_degenerate(self):
        """Tokenizer with whitespace-only tokens should report degenerate."""
        vocab = {"a": 0, "b": 1, "   ": 2, "\t": 3}
        tok = wrap_custom(
            name="degen_test",
            encode_fn=lambda t: [0],
            decode_fn=lambda ids: {0: "a", 1: "b", 2: "   ", 3: "\t"}.get(ids[0], "?"),
            vocab=vocab,
        )
        result = audit(tok)
        degenerate = [f for f in result.findings if f.category == "degenerate"]
        assert len(degenerate) >= 1

    def test_finds_control_chars(self):
        """Tokenizer with control character tokens."""
        vocab = {"a": 0, "\x01": 1, "\x02": 2}
        tok = wrap_custom(
            name="ctrl_test",
            encode_fn=lambda t: [0],
            decode_fn=lambda ids: {0: "a", 1: "\x01", 2: "\x02"}.get(ids[0], "?"),
            vocab=vocab,
        )
        result = audit(tok)
        ctrl = [f for f in result.findings if f.category == "control_char"]
        assert len(ctrl) >= 1

    def test_finds_repeated_char(self):
        """Tokenizer with repeated character tokens."""
        vocab = {"a": 0, "aaaa": 1, "bbbb": 2}
        tok = wrap_custom(
            name="repeat_test",
            encode_fn=lambda t: [0],
            decode_fn=lambda ids: {0: "a", 1: "aaaa", 2: "bbbb"}.get(ids[0], "?"),
            vocab=vocab,
        )
        result = audit(tok)
        repeated = [f for f in result.findings if f.category == "repeated_char"]
        assert len(repeated) >= 1

    def test_finds_duplicate_surface(self):
        """Multiple IDs decoding to same string."""
        vocab = {"a": 0, "A": 1}  # Different keys in vocab
        # But decode gives same output for different IDs
        tok = wrap_custom(
            name="dup_test",
            encode_fn=lambda t: [0],
            decode_fn=lambda ids: "a",  # Always returns "a"
            vocab=vocab,
        )
        result = audit(tok)
        dups = [f for f in result.findings if f.category == "duplicate_surface"]
        assert len(dups) >= 1

    def test_categories(self, mock_tokenizer):
        result = audit(mock_tokenizer)
        categories = result.by_category
        # Should have dict of category -> list of findings
        assert isinstance(categories, dict)


class TestFindGlitchTokens:
    def test_returns_list(self, mock_tokenizer):
        result = find_glitch_tokens(mock_tokenizer, max_tokens=50)
        assert isinstance(result, list)


class TestFindDegenerateTokens:
    def test_returns_list(self, mock_tokenizer):
        result = find_degenerate_tokens(mock_tokenizer, max_tokens=50)
        assert isinstance(result, list)
