"""Compression ratio and fertility analysis."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from toksight._types import CompressionStats
from toksight.loader import TokenizerWrapper


def compute_compression(
    tokenizer: TokenizerWrapper,
    texts: Sequence[str],
) -> CompressionStats:
    """Compute compression statistics over a corpus of texts."""
    total_chars = 0
    total_bytes = 0
    total_tokens = 0
    total_words = 0

    for text in texts:
        if not text:
            continue
        ids = tokenizer.encode(text)
        total_chars += len(text)
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(ids)
        total_words += len(text.split())

    if total_tokens == 0:
        return CompressionStats(
            total_chars=total_chars,
            total_bytes=total_bytes,
            total_tokens=0,
            bytes_per_token=0.0,
            chars_per_token=0.0,
            tokens_per_word=0.0,
            fertility=0.0,
        )

    bpt = total_bytes / total_tokens
    cpt = total_chars / total_tokens
    tpw = total_tokens / total_words if total_words else 0.0

    return CompressionStats(
        total_chars=total_chars,
        total_bytes=total_bytes,
        total_tokens=total_tokens,
        bytes_per_token=bpt,
        chars_per_token=cpt,
        tokens_per_word=tpw,
        fertility=tpw,
    )


def compression_by_language(
    tokenizer: TokenizerWrapper,
    texts_by_lang: Dict[str, Sequence[str]],
) -> Dict[str, CompressionStats]:
    """Compute compression for each language/category in a dict."""
    return {lang: compute_compression(tokenizer, texts) for lang, texts in texts_by_lang.items()}


def compare_compression(
    tokenizers: Sequence[TokenizerWrapper],
    texts: Sequence[str],
) -> Dict[str, CompressionStats]:
    """Compare compression across multiple tokenizers."""
    return {tok.name: compute_compression(tok, texts) for tok in tokenizers}


def fertility_analysis(
    tokenizer: TokenizerWrapper,
    texts: Sequence[str],
) -> Dict[str, float]:
    """Detailed fertility (tokens-per-word) analysis.

    Returns a dict with per-text fertility and overall stats.
    """
    fertilities: List[float] = []
    for text in texts:
        if not text.strip():
            continue
        words = text.split()
        if not words:
            continue
        tokens = tokenizer.encode(text)
        fertilities.append(len(tokens) / len(words))

    if not fertilities:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 0.0}

    n = len(fertilities)
    mean = sum(fertilities) / n
    variance = sum((f - mean) ** 2 for f in fertilities) / n if n > 1 else 0.0

    return {
        "mean": mean,
        "min": min(fertilities),
        "max": max(fertilities),
        "std": variance**0.5,
        "count": float(n),
    }
