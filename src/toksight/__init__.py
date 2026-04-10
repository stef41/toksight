"""toksight — tokenizer analysis toolkit."""

__version__ = "0.3.0"

from toksight._types import (
    AuditFinding,
    AuditResult,
    CompareResult,
    CompressionStats,
    CostEstimate,
    CoverageResult,
    TokenizerBackend,
    TokenizerInfo,
    TokenSpan,
    ToksightError,
    VocabStats,
)
from toksight.loader import (
    TokenizerWrapper,
    load_tiktoken,
    load_huggingface,
    load_sentencepiece,
    wrap_custom,
)
from toksight.gguf import load_gguf
from toksight.diff import (
    TokenDiffResult,
    format_diff_html,
    format_diff_inline,
    format_diff_text,
    token_diff,
)
from toksight.training_coverage import (
    CoverageEstimate,
    DomainCoverageResult,
    estimate_coverage,
    domain_coverage,
    format_coverage_report,
)
from toksight.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    TokenizerBenchmark,
    format_benchmark_report,
    generate_benchmark_texts,
)
from toksight.overlap import (
    OverlapResult,
    VocabOverlapAnalyzer,
    format_overlap_report,
    overlap_matrix,
)

__all__ = [
    "AuditFinding",
    "AuditResult",
    "CompareResult",
    "CompressionStats",
    "CostEstimate",
    "CoverageResult",
    "TokenDiffResult",
    "TokenizerBackend",
    "TokenizerInfo",
    "TokenizerWrapper",
    "TokenSpan",
    "ToksightError",
    "VocabStats",
    "format_diff_html",
    "format_diff_inline",
    "format_diff_text",
    "load_gguf",
    "load_huggingface",
    "load_sentencepiece",
    "load_tiktoken",
    "token_diff",
    "wrap_custom",
    "CoverageEstimate",
    "DomainCoverageResult",
    "estimate_coverage",
    "domain_coverage",
    "format_coverage_report",
    "BenchmarkConfig",
    "BenchmarkResult",
    "TokenizerBenchmark",
    "format_benchmark_report",
    "generate_benchmark_texts",
    "OverlapResult",
    "VocabOverlapAnalyzer",
    "format_overlap_report",
    "overlap_matrix",
]
