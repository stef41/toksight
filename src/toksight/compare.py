"""Cross-tokenizer comparison — vocabulary overlap, boundary alignment, fragmentation."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from toksight._types import CompareResult, CompressionStats
from toksight.compression import compute_compression
from toksight.loader import TokenizerWrapper


def compare_vocabularies(
    tok_a: TokenizerWrapper,
    tok_b: TokenizerWrapper,
) -> CompareResult:
    """Compare vocabulary overlap between two tokenizers."""
    vocab_a = set(tok_a.vocab.keys())
    vocab_b = set(tok_b.vocab.keys())

    overlap = vocab_a & vocab_b
    only_a = vocab_a - vocab_b
    only_b = vocab_b - vocab_a

    union = vocab_a | vocab_b
    jaccard = len(overlap) / len(union) if union else 0.0

    return CompareResult(
        tokenizer_a=tok_a.name,
        tokenizer_b=tok_b.name,
        vocab_overlap=len(overlap),
        vocab_only_a=len(only_a),
        vocab_only_b=len(only_b),
        jaccard_similarity=round(jaccard, 4),
    )


def compare_on_corpus(
    tok_a: TokenizerWrapper,
    tok_b: TokenizerWrapper,
    texts: Sequence[str],
) -> CompareResult:
    """Compare two tokenizers on a corpus — vocab overlap + compression + boundaries."""
    result = compare_vocabularies(tok_a, tok_b)

    # Compression comparison
    result.compression_a = compute_compression(tok_a, texts)
    result.compression_b = compute_compression(tok_b, texts)

    # Boundary agreement
    agree = 0
    total = 0
    for text in texts:
        if not text.strip():
            continue
        total += 1
        spans_a = tok_a.tokenize(text)
        spans_b = tok_b.tokenize(text)
        if _boundaries_match(spans_a, spans_b):
            agree += 1

    result.boundary_agreement = round(agree / total, 4) if total > 0 else 0.0

    return result


def _boundaries_match(spans_a: list, spans_b: list) -> bool:
    """Check if two tokenizations produce the same character boundaries."""
    offsets_a = _boundary_offsets(spans_a)
    offsets_b = _boundary_offsets(spans_b)
    return offsets_a == offsets_b


def _boundary_offsets(spans: list) -> List[int]:
    """Extract character boundary offsets from token spans."""
    offsets = []
    pos = 0
    for span in spans:
        pos += span.char_length
        offsets.append(pos)
    return offsets


def boundary_alignment(
    tok_a: TokenizerWrapper,
    tok_b: TokenizerWrapper,
    text: str,
) -> Dict[str, object]:
    """Detailed boundary alignment analysis for a single text.

    Returns character-level alignment showing where tokenizers agree/disagree.
    """
    spans_a = tok_a.tokenize(text)
    spans_b = tok_b.tokenize(text)

    boundaries_a = set(_boundary_offsets(spans_a))
    boundaries_b = set(_boundary_offsets(spans_b))

    shared = boundaries_a & boundaries_b
    only_a = boundaries_a - boundaries_b
    only_b = boundaries_b - boundaries_a

    total = boundaries_a | boundaries_b
    agreement = len(shared) / len(total) if total else 1.0

    return {
        "text_length": len(text),
        "boundaries_a": len(boundaries_a),
        "boundaries_b": len(boundaries_b),
        "shared_boundaries": len(shared),
        "only_a": sorted(only_a),
        "only_b": sorted(only_b),
        "agreement": round(agreement, 4),
        "tokens_a": [s.text for s in spans_a],
        "tokens_b": [s.text for s in spans_b],
    }


def fragmentation_map(
    tok_a: TokenizerWrapper,
    tok_b: TokenizerWrapper,
    text: str,
) -> List[Dict[str, object]]:
    """Show how each tokenizer fragments the same text, token by token.

    Returns a list of alignment entries showing corresponding tokens.
    """
    spans_a = tok_a.tokenize(text)
    spans_b = tok_b.tokenize(text)

    entries: List[Dict[str, object]] = []
    idx_a = 0
    idx_b = 0
    pos_a = 0
    pos_b = 0

    while idx_a < len(spans_a) or idx_b < len(spans_b):
        entry: Dict[str, object] = {}

        if idx_a < len(spans_a):
            entry["token_a"] = spans_a[idx_a].text
            entry["id_a"] = spans_a[idx_a].token_id
            pos_a += spans_a[idx_a].char_length
            idx_a += 1
        else:
            entry["token_a"] = ""
            entry["id_a"] = -1

        if idx_b < len(spans_b):
            entry["token_b"] = spans_b[idx_b].text
            entry["id_b"] = spans_b[idx_b].token_id
            pos_b += spans_b[idx_b].char_length
            idx_b += 1
        else:
            entry["token_b"] = ""
            entry["id_b"] = -1

        entry["aligned"] = (entry["token_a"] == entry["token_b"])
        entries.append(entry)

    return entries
