"""Vocabulary overlap analysis between tokenizers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Set, Tuple, Union


@dataclass
class OverlapResult:
    """Result of comparing two vocabularies."""

    tokenizer_a: str
    tokenizer_b: str
    shared_count: int
    only_a_count: int
    only_b_count: int
    jaccard: float
    overlap_coefficient: float


class VocabOverlapAnalyzer:
    """Analyze overlap between tokenizer vocabularies."""

    def compare(
        self,
        vocab_a: Union[set, list],
        vocab_b: Union[set, list],
        name_a: str = "A",
        name_b: str = "B",
    ) -> OverlapResult:
        """Compare two vocabularies and return overlap statistics."""
        set_a = set(vocab_a)
        set_b = set(vocab_b)
        shared = set_a & set_b
        only_a = set_a - set_b
        only_b = set_b - set_a
        union = set_a | set_b
        jaccard = len(shared) / len(union) if union else 0.0
        min_size = min(len(set_a), len(set_b))
        overlap_coeff = len(shared) / min_size if min_size else 0.0
        return OverlapResult(
            tokenizer_a=name_a,
            tokenizer_b=name_b,
            shared_count=len(shared),
            only_a_count=len(only_a),
            only_b_count=len(only_b),
            jaccard=jaccard,
            overlap_coefficient=overlap_coeff,
        )

    def multi_compare(
        self, vocabs: Dict[str, Union[set, list]]
    ) -> List[OverlapResult]:
        """All pairwise comparisons between named vocabularies."""
        results: List[OverlapResult] = []
        names = list(vocabs.keys())
        for name_a, name_b in combinations(names, 2):
            results.append(
                self.compare(vocabs[name_a], vocabs[name_b], name_a, name_b)
            )
        return results

    def shared_tokens(
        self,
        vocab_a: Union[set, list],
        vocab_b: Union[set, list],
    ) -> set:
        """Return the intersection of two vocabularies."""
        return set(vocab_a) & set(vocab_b)

    def unique_tokens(
        self,
        vocab_a: Union[set, list],
        vocab_b: Union[set, list],
    ) -> Tuple[set, set]:
        """Return tokens unique to each vocabulary."""
        set_a = set(vocab_a)
        set_b = set(vocab_b)
        return set_a - set_b, set_b - set_a

    def coverage(
        self,
        vocab_subset: Union[set, list],
        vocab_full: Union[set, list],
    ) -> float:
        """What percentage of *vocab_full* is covered by *vocab_subset*."""
        full = set(vocab_full)
        if not full:
            return 0.0
        return len(set(vocab_subset) & full) / len(full)

    def merge_vocabularies(self, vocabs: List[set]) -> set:
        """Union all vocabularies."""
        merged: set = set()
        for v in vocabs:
            merged |= set(v)
        return merged


def overlap_matrix(vocabs: Dict[str, Union[set, list]]) -> Dict[str, Dict[str, float]]:
    """Return an N×N matrix of Jaccard scores as nested dicts.

    Keys are tokenizer names; ``matrix[a][b]`` is the Jaccard index between
    vocab *a* and vocab *b*.
    """
    sets = {k: set(v) for k, v in vocabs.items()}
    names = list(sets.keys())
    matrix: Dict[str, Dict[str, float]] = {}
    for a in names:
        matrix[a] = {}
        for b in names:
            if a == b:
                matrix[a][b] = 1.0
            else:
                union = sets[a] | sets[b]
                matrix[a][b] = len(sets[a] & sets[b]) / len(union) if union else 0.0
    return matrix


def format_overlap_report(results: Union[OverlapResult, List[OverlapResult]]) -> str:
    """Format one or more :class:`OverlapResult` objects as a human-readable report."""
    if isinstance(results, OverlapResult):
        results = [results]
    lines: List[str] = ["Vocabulary Overlap Report", "=" * 40]
    for r in results:
        lines.append(f"\n{r.tokenizer_a} vs {r.tokenizer_b}")
        lines.append("-" * 30)
        lines.append(f"  Shared tokens:       {r.shared_count}")
        lines.append(f"  Only in {r.tokenizer_a}:".ljust(25) + str(r.only_a_count))
        lines.append(f"  Only in {r.tokenizer_b}:".ljust(25) + str(r.only_b_count))
        lines.append(f"  Jaccard index:       {r.jaccard:.4f}")
        lines.append(f"  Overlap coefficient: {r.overlap_coefficient:.4f}")
    return "\n".join(lines)
