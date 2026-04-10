"""Vocabulary statistics — size, length distribution, script coverage, special tokens."""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Optional

from toksight._types import VocabStats
from toksight.loader import TokenizerWrapper


def vocab_stats(tokenizer: TokenizerWrapper) -> VocabStats:
    """Compute comprehensive vocabulary statistics."""
    vocab = tokenizer.vocab
    special = set(tokenizer.special_tokens)

    n_special = 0
    n_byte = 0
    n_single_char = 0
    n_multiword = 0
    lengths: List[int] = []
    length_dist: Dict[int, int] = {}
    script_counts: Dict[str, int] = {}

    for token_text, token_id in vocab.items():
        if token_text in special:
            n_special += 1
            continue

        byte_len = len(token_text.encode("utf-8", errors="replace"))
        lengths.append(byte_len)

        # Length distribution
        bucket = byte_len
        length_dist[bucket] = length_dist.get(bucket, 0) + 1

        # Single byte tokens
        if byte_len == 1:
            n_byte += 1

        # Single character tokens
        if len(token_text) == 1:
            n_single_char += 1

        # Multi-word tokens (contain space)
        if " " in token_text or "\t" in token_text:
            n_multiword += 1

        # Script detection (first non-space character)
        for ch in token_text:
            if ch.isspace():
                continue
            try:
                name = unicodedata.name(ch, "")
                script = name.split()[0] if name else "UNKNOWN"
            except (ValueError, IndexError):
                script = "UNKNOWN"
            script_counts[script] = script_counts.get(script, 0) + 1
            break

    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    max_len = max(lengths) if lengths else 0

    return VocabStats(
        vocab_size=len(vocab),
        n_special_tokens=n_special,
        n_byte_tokens=n_byte,
        n_single_char=n_single_char,
        n_multiword=n_multiword,
        avg_token_length=round(avg_len, 2),
        max_token_length=max_len,
        length_distribution=dict(sorted(length_dist.items())),
        script_coverage=dict(sorted(script_counts.items(), key=lambda x: -x[1])),
    )


def top_tokens_by_length(
    tokenizer: TokenizerWrapper,
    n: int = 20,
) -> List[Dict[str, object]]:
    """Get the longest tokens by byte length."""
    entries = []
    for token_text, token_id in tokenizer.vocab.items():
        if token_text in set(tokenizer.special_tokens):
            continue
        byte_len = len(token_text.encode("utf-8", errors="replace"))
        entries.append({
            "token_id": token_id,
            "text": token_text,
            "byte_length": byte_len,
            "char_length": len(token_text),
        })

    entries.sort(key=lambda x: x["byte_length"], reverse=True)
    return entries[:n]


def token_length_histogram(
    tokenizer: TokenizerWrapper,
) -> Dict[int, int]:
    """Get histogram of token byte lengths."""
    stats = vocab_stats(tokenizer)
    return stats.length_distribution
