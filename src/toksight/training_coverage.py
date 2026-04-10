"""Training data coverage estimation for tokenizers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from toksight.loader import TokenizerWrapper


@dataclass
class CoverageEstimate:
    """Result of estimating vocabulary coverage over a corpus."""

    vocab_used: int
    vocab_total: int
    coverage_ratio: float
    unused_tokens: List[str]
    rare_tokens: List[Tuple[str, int]]
    most_frequent: List[Tuple[str, int]]


@dataclass
class DomainCoverageResult:
    """Comparison of token usage between domain and reference corpora."""

    domain_vocab_used: int
    reference_vocab_used: int
    shared_tokens: int
    domain_only_tokens: List[str]
    reference_only_tokens: List[str]
    overrepresented: List[Tuple[str, float]]
    underrepresented: List[Tuple[str, float]]


def estimate_coverage(
    tokenizer: TokenizerWrapper,
    corpus_texts: Sequence[str],
    top_k: int = 50,
) -> CoverageEstimate:
    """Tokenize a corpus and measure what fraction of the vocabulary is used.

    Args:
        tokenizer: A loaded tokenizer.
        corpus_texts: Iterable of text strings to analyse.
        top_k: Number of most-frequent / rare tokens to include.

    Returns:
        CoverageEstimate with usage statistics.
    """
    token_counts: Counter[int] = Counter()

    for text in corpus_texts:
        ids = tokenizer.encode(text)
        token_counts.update(ids)

    vocab = tokenizer.vocab
    id_to_token = tokenizer.id_to_token
    vocab_total = tokenizer.vocab_size
    used_ids = set(token_counts.keys())

    # Unused tokens
    all_ids = set(id_to_token.keys())
    unused_ids = all_ids - used_ids
    unused_tokens = sorted(id_to_token[tid] for tid in unused_ids if tid in id_to_token)

    # Rare tokens (used but with lowest counts), excluding unused
    sorted_by_count = token_counts.most_common()
    sorted_by_count_asc = sorted_by_count[::-1]
    rare_tokens = [
        (id_to_token.get(tid, f"<id:{tid}>"), cnt)
        for tid, cnt in sorted_by_count_asc[:top_k]
    ]

    # Most frequent
    most_frequent = [
        (id_to_token.get(tid, f"<id:{tid}>"), cnt)
        for tid, cnt in sorted_by_count[:top_k]
    ]

    vocab_used = len(used_ids)
    coverage_ratio = vocab_used / vocab_total if vocab_total > 0 else 0.0

    return CoverageEstimate(
        vocab_used=vocab_used,
        vocab_total=vocab_total,
        coverage_ratio=round(coverage_ratio, 6),
        unused_tokens=unused_tokens,
        rare_tokens=rare_tokens,
        most_frequent=most_frequent,
    )


def _token_freq_ratio(counts: Counter[int], total: int) -> Dict[int, float]:
    """Return per-token frequency ratio."""
    if total == 0:
        return {}
    return {tid: cnt / total for tid, cnt in counts.items()}


def domain_coverage(
    tokenizer: TokenizerWrapper,
    domain_texts: Sequence[str],
    reference_texts: Sequence[str],
    top_k: int = 50,
) -> DomainCoverageResult:
    """Compare token usage between a domain corpus and a reference corpus.

    Tokens that appear much more in the domain corpus are *overrepresented*;
    tokens that appear much less are *underrepresented*.

    Args:
        tokenizer: A loaded tokenizer.
        domain_texts: Domain-specific texts.
        reference_texts: General reference texts.
        top_k: Number of over/under-represented tokens to return.
    """
    domain_counts: Counter[int] = Counter()
    ref_counts: Counter[int] = Counter()

    for text in domain_texts:
        domain_counts.update(tokenizer.encode(text))
    for text in reference_texts:
        ref_counts.update(tokenizer.encode(text))

    domain_total = sum(domain_counts.values())
    ref_total = sum(ref_counts.values())

    domain_freq = _token_freq_ratio(domain_counts, domain_total)
    ref_freq = _token_freq_ratio(ref_counts, ref_total)

    id_to_token = tokenizer.id_to_token
    domain_ids = set(domain_counts.keys())
    ref_ids = set(ref_counts.keys())

    shared = domain_ids & ref_ids
    domain_only = sorted(
        id_to_token.get(tid, f"<id:{tid}>") for tid in (domain_ids - ref_ids) if tid in id_to_token
    )
    ref_only = sorted(
        id_to_token.get(tid, f"<id:{tid}>") for tid in (ref_ids - domain_ids) if tid in id_to_token
    )

    # Compute log-ratio for shared tokens
    ratios: List[Tuple[int, float]] = []
    for tid in shared:
        d = domain_freq.get(tid, 0.0)
        r = ref_freq.get(tid, 0.0)
        if r > 0:
            ratios.append((tid, d / r))

    ratios.sort(key=lambda x: x[1], reverse=True)

    overrepresented = [
        (id_to_token.get(tid, f"<id:{tid}>"), round(ratio, 4))
        for tid, ratio in ratios[:top_k]
        if ratio > 1.0
    ]
    underrepresented = [
        (id_to_token.get(tid, f"<id:{tid}>"), round(ratio, 4))
        for tid, ratio in ratios[-top_k:]
        if ratio < 1.0
    ]
    # underrepresented sorted from most under-represented
    underrepresented = underrepresented[::-1]

    return DomainCoverageResult(
        domain_vocab_used=len(domain_ids),
        reference_vocab_used=len(ref_ids),
        shared_tokens=len(shared),
        domain_only_tokens=domain_only,
        reference_only_tokens=ref_only,
        overrepresented=overrepresented,
        underrepresented=underrepresented,
    )


def format_coverage_report(estimate: CoverageEstimate) -> str:
    """Return a human-readable coverage report."""
    lines = [
        "=== Vocabulary Coverage Report ===",
        f"Vocabulary size : {estimate.vocab_total}",
        f"Tokens used     : {estimate.vocab_used}",
        f"Coverage ratio  : {estimate.coverage_ratio:.2%}",
        f"Unused tokens   : {len(estimate.unused_tokens)}",
        "",
        "--- Most Frequent Tokens ---",
    ]
    for token, count in estimate.most_frequent[:20]:
        lines.append(f"  {token!r:30s}  {count}")

    lines.append("")
    lines.append("--- Rare Tokens ---")
    for token, count in estimate.rare_tokens[:20]:
        lines.append(f"  {token!r:30s}  {count}")

    return "\n".join(lines)
