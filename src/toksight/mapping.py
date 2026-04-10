"""Token-to-token mapping between tokenizers."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from toksight.loader import TokenizerWrapper


def map_tokens(
    source: TokenizerWrapper,
    target: TokenizerWrapper,
    text: str,
) -> List[Dict[str, object]]:
    """Map tokens from source tokenizer to target tokenizer for a given text.

    For each source token, shows which target token(s) it maps to.
    """
    src_spans = source.tokenize(text)
    tgt_spans = target.tokenize(text)

    # Build character position → target token index mapping
    tgt_char_map: Dict[int, int] = {}
    for idx, span in enumerate(tgt_spans):
        for c in range(span.char_offset, span.char_offset + span.char_length):
            tgt_char_map[c] = idx

    result: List[Dict[str, object]] = []
    for src_span in src_spans:
        # Find which target tokens cover this source token's characters
        tgt_indices = set()
        for c in range(src_span.char_offset, src_span.char_offset + src_span.char_length):
            if c in tgt_char_map:
                tgt_indices.add(tgt_char_map[c])

        mapped_targets = [
            {"text": tgt_spans[i].text, "id": tgt_spans[i].token_id}
            for i in sorted(tgt_indices)
        ]

        result.append({
            "source_token": src_span.text,
            "source_id": src_span.token_id,
            "target_tokens": mapped_targets,
            "expansion": len(mapped_targets),
            "is_one_to_one": len(mapped_targets) == 1,
        })

    return result


def token_expansion_ratio(
    source: TokenizerWrapper,
    target: TokenizerWrapper,
    texts: Sequence[str],
) -> Dict[str, float]:
    """Compute how tokens expand/contract when converting between tokenizers.

    Returns average, min, max expansion ratio (target_tokens / source_tokens).
    """
    ratios: List[float] = []
    for text in texts:
        if not text.strip():
            continue
        src_ids = source.encode(text)
        tgt_ids = target.encode(text)
        if src_ids:
            ratios.append(len(tgt_ids) / len(src_ids))

    if not ratios:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0.0}

    return {
        "mean": sum(ratios) / len(ratios),
        "min": min(ratios),
        "max": max(ratios),
        "count": float(len(ratios)),
    }
