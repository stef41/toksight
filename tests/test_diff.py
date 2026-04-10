"""Tests for the visual token-diff module."""

import pytest

from toksight.diff import (
    TokenDiffResult,
    format_diff_html,
    format_diff_inline,
    format_diff_text,
    token_diff,
)
from toksight.loader import wrap_custom


# --- helpers: two tokenizers that genuinely split differently ----------------

def _greedy_tokenizer(name, token_list):
    """Build a greedy longest-match tokenizer from *token_list*."""
    vocab = {t: i for i, t in enumerate(token_list)}
    reverse = {i: t for t, i in vocab.items()}
    sorted_tokens = sorted(token_list, key=len, reverse=True)

    def encode(text):
        ids, i = [], 0
        while i < len(text):
            for tok in sorted_tokens:
                if text[i : i + len(tok)] == tok:
                    ids.append(vocab[tok])
                    i += len(tok)
                    break
            else:
                i += 1  # skip unknown
        return ids

    def decode(ids):
        return "".join(reverse.get(tid, "?") for tid in ids)

    return wrap_custom(name=name, encode_fn=encode, decode_fn=decode, vocab=vocab)


@pytest.fixture
def tok_word():
    """Treats 'hello' and 'world' as single tokens."""
    return _greedy_tokenizer("word", ["hello", " ", "world"])


@pytest.fixture
def tok_subword():
    """Splits into sub-word pieces."""
    return _greedy_tokenizer("subword", ["he", "llo", " ", "wor", "ld"])


# --- token_diff -------------------------------------------------------------

class TestTokenDiff:
    def test_same_tokenizer(self, mock_tokenizer):
        result = token_diff("hello world", mock_tokenizer, mock_tokenizer)
        assert result.diff_count == 0
        assert result.common_count > 0
        assert result.tokens_a == result.tokens_b
        assert result.boundaries_a == result.boundaries_b

    def test_result_fields(self, mock_tokenizer):
        result = token_diff("hello", mock_tokenizer, mock_tokenizer)
        assert result.text == "hello"
        assert isinstance(result.tokens_a, list)
        assert isinstance(result.boundaries_a, list)
        assert isinstance(result.common_count, int)
        assert isinstance(result.diff_count, int)

    def test_boundaries_cumulative(self, mock_tokenizer):
        result = token_diff("hello world", mock_tokenizer, mock_tokenizer)
        for i in range(1, len(result.boundaries_a)):
            assert result.boundaries_a[i] > result.boundaries_a[i - 1]
        assert result.boundaries_a[-1] == len("hello world")

    def test_empty_text(self, mock_tokenizer):
        result = token_diff("", mock_tokenizer, mock_tokenizer)
        assert result.tokens_a == []
        assert result.tokens_b == []
        assert result.boundaries_a == []
        assert result.boundaries_b == []
        assert result.common_count == 0
        assert result.diff_count == 0

    def test_single_char(self, mock_tokenizer):
        result = token_diff("a", mock_tokenizer, mock_tokenizer)
        assert len(result.tokens_a) == 1
        assert result.boundaries_a == [1]
        assert result.common_count == 1
        assert result.diff_count == 0

    def test_different_tokenizers(self, tok_word, tok_subword):
        # word:    ["hello", " ", "world"]  → boundaries [5, 6, 11]
        # subword: ["he","llo"," ","wor","ld"] → boundaries [2, 5, 6, 9, 11]
        result = token_diff("hello world", tok_word, tok_subword)
        assert result.tokens_a == ["hello", " ", "world"]
        assert result.tokens_b == ["he", "llo", " ", "wor", "ld"]
        assert result.boundaries_a == [5, 6, 11]
        assert result.boundaries_b == [2, 5, 6, 9, 11]
        # common: {5, 6, 11} = 3   diff: {2, 9} = 2
        assert result.common_count == 3
        assert result.diff_count == 2

    def test_common_plus_diff_equals_union(self, tok_word, tok_subword):
        result = token_diff("hello world", tok_word, tok_subword)
        union = set(result.boundaries_a) | set(result.boundaries_b)
        assert result.common_count + result.diff_count == len(union)


# --- format_diff_text -------------------------------------------------------

class TestFormatDiffText:
    def test_contains_tokens(self, tok_word, tok_subword):
        result = token_diff("hello world", tok_word, tok_subword)
        text = format_diff_text(result)
        assert "[hello]" in text
        assert "[he]" in text
        assert "[llo]" in text

    def test_contains_counts(self, tok_word, tok_subword):
        result = token_diff("hello world", tok_word, tok_subword)
        text = format_diff_text(result)
        assert "Common: 3" in text
        assert "Different: 2" in text

    def test_alignment_markers(self, tok_word, tok_subword):
        result = token_diff("hello world", tok_word, tok_subword)
        text = format_diff_text(result)
        assert "Alignment:" in text
        # 2 and 9 are subword-only → marked B
        assert "2:B" in text
        assert "9:B" in text
        # 5, 6, 11 are shared → marked =
        assert "5:=" in text


# --- format_diff_html -------------------------------------------------------

class TestFormatDiffHtml:
    def test_contains_spans(self, mock_tokenizer):
        result = token_diff("hello", mock_tokenizer, mock_tokenizer)
        html = format_diff_html(result)
        assert "<span" in html
        assert "background:" in html

    def test_escapes_special_chars(self, mock_tokenizer):
        result = token_diff("<b>", mock_tokenizer, mock_tokenizer)
        html = format_diff_html(result)
        assert "&lt;" in html
        assert "&gt;" in html
        assert "<b>" not in html  # must be escaped

    def test_green_for_shared(self, tok_word, tok_subword):
        result = token_diff("hello world", tok_word, tok_subword)
        html = format_diff_html(result)
        # "hello" in tok_word ends at 5 which is shared → green
        assert "#c8e6c9" in html
        # "he" in tok_subword ends at 2 which is unique → red
        assert "#ffcdd2" in html


# --- format_diff_inline -----------------------------------------------------

class TestFormatDiffInline:
    def test_boundaries(self, tok_word, tok_subword):
        result = token_diff("hello world", tok_word, tok_subword)
        text = format_diff_inline(result)
        assert "A: |hello| |world|" in text
        assert "B: |he|llo| |wor|ld|" in text

    def test_both_labels(self, mock_tokenizer):
        result = token_diff("ab", mock_tokenizer, mock_tokenizer)
        text = format_diff_inline(result)
        assert text.startswith("A: ")
        assert "\nB: " in text

    def test_empty_tokens(self, mock_tokenizer):
        result = token_diff("", mock_tokenizer, mock_tokenizer)
        text = format_diff_inline(result)
        assert "A: ||" in text
        assert "B: ||" in text
