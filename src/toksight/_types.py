"""Core types for toksight."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class TokenizerBackend(str, Enum):
    """Detected tokenizer backend."""

    TIKTOKEN = "tiktoken"
    HUGGINGFACE = "huggingface"
    SENTENCEPIECE = "sentencepiece"
    CUSTOM = "custom"


@dataclass
class TokenizerInfo:
    """Basic information about a tokenizer."""

    name: str
    backend: TokenizerBackend
    vocab_size: int
    special_tokens: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenSpan:
    """A single token with its text and byte representation."""

    token_id: int
    text: str
    byte_length: int
    char_offset: int = 0
    char_length: int = 0

    @property
    def bytes_per_char(self) -> float:
        if self.char_length == 0:
            return 0.0
        return self.byte_length / self.char_length


@dataclass
class CompressionStats:
    """Compression statistics for a corpus."""

    total_chars: int
    total_bytes: int
    total_tokens: int
    bytes_per_token: float
    chars_per_token: float
    tokens_per_word: float = 0.0
    fertility: float = 0.0  # tokens per whitespace-separated word

    @property
    def compression_ratio(self) -> float:
        """Bytes compressed into tokens (higher = better compression)."""
        if self.total_tokens == 0:
            return 0.0
        return self.total_bytes / self.total_tokens


@dataclass
class CoverageResult:
    """Unicode coverage analysis result."""

    total_codepoints_tested: int
    codepoints_covered: int
    coverage_ratio: float
    blocks_analyzed: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # block_name -> {tested, covered, ratio, sample_uncovered}

    @property
    def uncovered_count(self) -> int:
        return self.total_codepoints_tested - self.codepoints_covered


@dataclass
class CompareResult:
    """Result of comparing two tokenizers."""

    tokenizer_a: str
    tokenizer_b: str
    vocab_overlap: int
    vocab_only_a: int
    vocab_only_b: int
    jaccard_similarity: float
    compression_a: Optional[CompressionStats] = None
    compression_b: Optional[CompressionStats] = None
    boundary_agreement: float = 0.0  # fraction of texts with identical boundaries
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def overlap_ratio_a(self) -> float:
        total = self.vocab_overlap + self.vocab_only_a
        return self.vocab_overlap / total if total else 0.0

    @property
    def overlap_ratio_b(self) -> float:
        total = self.vocab_overlap + self.vocab_only_b
        return self.vocab_overlap / total if total else 0.0


@dataclass
class AuditFinding:
    """A single finding from tokenizer audit."""

    category: str  # e.g., "glitch_token", "degenerate", "overlong"
    severity: str  # "info", "warning", "critical"
    token_id: int
    token_text: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditResult:
    """Tokenizer audit result."""

    tokenizer_name: str
    findings: List[AuditFinding] = field(default_factory=list)

    @property
    def n_critical(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def n_warnings(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warning")

    @property
    def by_category(self) -> Dict[str, List[AuditFinding]]:
        result: Dict[str, List[AuditFinding]] = {}
        for f in self.findings:
            result.setdefault(f.category, []).append(f)
        return result


@dataclass
class CostEstimate:
    """Token cost estimation for a corpus across providers."""

    corpus_chars: int
    estimates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # tokenizer_name -> {tokens, cost_per_1k_input, cost_per_1k_output, total_input_cost, ...}


@dataclass
class VocabStats:
    """Vocabulary statistics."""

    vocab_size: int
    n_special_tokens: int
    n_byte_tokens: int  # single-byte fallback tokens
    n_single_char: int  # tokens that are a single character
    n_multiword: int  # tokens containing whitespace (multi-word/subword with space)
    avg_token_length: float  # average byte length of vocabulary entries
    max_token_length: int
    length_distribution: Dict[int, int] = field(default_factory=dict)  # length -> count
    script_coverage: Dict[str, int] = field(default_factory=dict)  # unicode_script -> count


class ToksightError(Exception):
    """Base exception for toksight."""
