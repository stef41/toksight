"""Edge-case and hardened tests for toksight modules."""

import pytest
from toksight.audit import audit, find_degenerate_tokens, find_glitch_tokens
from toksight.compare import (
    boundary_alignment,
    compare_on_corpus,
    compare_vocabularies,
    fragmentation_map,
)
from toksight.compression import (
    compare_compression,
    compression_by_language,
    compute_compression,
    fertility_analysis,
)
from toksight.cost import compare_costs, estimate_cost
from toksight.coverage import analyze_coverage, coverage_for_text, detect_script
from toksight.loader import TokenizerWrapper, wrap_custom
from toksight.mapping import map_tokens, token_expansion_ratio
from toksight.stats import token_length_histogram, top_tokens_by_length, vocab_stats
from toksight._types import (
    AuditFinding,
    AuditResult,
    CompareResult,
    CompressionStats,
    TokenSpan,
    VocabStats,
)


# ---- audit edge cases ----


class TestAuditEdgeCases:
    def test_max_tokens_zero_checks_all(self, mock_tokenizer):
        # max_tokens=0 is falsy, so audit falls back to checking the full vocab
        result = audit(mock_tokenizer, max_tokens=0)
        assert isinstance(result.findings, list)

    def test_max_tokens_one(self, mock_tokenizer):
        result = audit(mock_tokenizer, max_tokens=1)
        # Only one token checked — very few findings possible
        assert result.tokenizer_name == "mock"

    def test_overlong_token_detection(self):
        long_text = "x" * 60
        vocab = {"a": 0, long_text: 1}
        tok = wrap_custom(
            name="overlong",
            encode_fn=lambda t: [0] if len(t) < 60 else [1],
            decode_fn=lambda ids: {0: "a", 1: long_text}.get(ids[0], "?"),
            vocab=vocab,
        )
        result = audit(tok)
        overlong = [f for f in result.findings if f.category == "overlong"]
        assert len(overlong) >= 1
        assert overlong[0].metadata["byte_length"] >= 60

    def test_decode_error_is_critical(self):
        vocab = {"a": 0, "b": 1}

        def bad_decode(ids):
            if ids == [1]:
                raise ValueError("cannot decode")
            return "a"

        tok = wrap_custom(
            name="bad_decode",
            encode_fn=lambda t: [0],
            decode_fn=bad_decode,
            vocab=vocab,
        )
        result = audit(tok)
        decode_errors = [f for f in result.findings if f.category == "decode_error"]
        assert any(f.severity == "critical" for f in decode_errors)

    def test_special_tokens_skipped(self):
        vocab = {"<s>": 0, "a": 1}
        tok = wrap_custom(
            name="special_test",
            encode_fn=lambda t: [1],
            decode_fn=lambda ids: {0: "   ", 1: "a"}.get(ids[0], "?"),
            vocab=vocab,
            special_tokens=["<s>"],
        )
        result = audit(tok)
        # <s> decodes to whitespace, but should not be flagged as degenerate since it's special
        degenerate = [f for f in result.findings if f.category == "degenerate"]
        assert all(f.token_id != 0 for f in degenerate)

    def test_find_glitch_tokens_filters(self, mock_tokenizer):
        result = find_glitch_tokens(mock_tokenizer, max_tokens=50)
        assert all(f.category == "glitch_token" for f in result)

    def test_find_degenerate_tokens_filters(self, mock_tokenizer):
        result = find_degenerate_tokens(mock_tokenizer, max_tokens=50)
        assert all(f.category == "degenerate" for f in result)


# ---- compare edge cases ----


class TestCompareEdgeCases:
    def test_zero_overlap_vocabularies(self):
        vocab_a = {"x": 0, "y": 1}
        vocab_b = {"a": 0, "b": 1}
        tok_a = wrap_custom("a", lambda t: [0], lambda ids: "x", vocab_a)
        tok_b = wrap_custom("b", lambda t: [0], lambda ids: "a", vocab_b)
        result = compare_vocabularies(tok_a, tok_b)
        assert result.vocab_overlap == 0
        assert result.jaccard_similarity == 0.0

    def test_boundary_alignment_empty_text(self, mock_tokenizer):
        result = boundary_alignment(mock_tokenizer, mock_tokenizer, "")
        assert result["text_length"] == 0
        assert result["agreement"] == 1.0

    def test_fragmentation_map_unicode(self, mock_tokenizer, mock_tokenizer_b):
        entries = fragmentation_map(mock_tokenizer, mock_tokenizer_b, "日本")
        assert len(entries) > 0

    def test_compare_on_corpus_single_text(self, mock_tokenizer, mock_tokenizer_b):
        result = compare_on_corpus(mock_tokenizer, mock_tokenizer_b, ["hello"])
        assert result.compression_a.total_tokens > 0

    def test_compare_on_corpus_whitespace_only(self, mock_tokenizer, mock_tokenizer_b):
        result = compare_on_corpus(mock_tokenizer, mock_tokenizer_b, ["   "])
        # whitespace-only should not crash
        assert isinstance(result.boundary_agreement, float)


# ---- compression edge cases ----


class TestCompressionEdgeCases:
    def test_single_char(self, mock_tokenizer):
        stats = compute_compression(mock_tokenizer, ["a"])
        assert stats.total_chars == 1
        assert stats.total_tokens == 1

    def test_mixed_empty_and_nonempty(self, mock_tokenizer):
        stats = compute_compression(mock_tokenizer, ["", "hello", ""])
        assert stats.total_chars == 5
        assert stats.total_tokens > 0

    def test_fertility_single_word(self, mock_tokenizer):
        result = fertility_analysis(mock_tokenizer, ["hello"])
        assert result["count"] == 1.0
        assert result["mean"] > 0
        assert result["std"] == 0.0  # single sample => no variance

    def test_compare_compression_single(self, mock_tokenizer):
        result = compare_compression([mock_tokenizer], ["hello world"])
        assert "mock" in result

    def test_large_corpus(self, mock_tokenizer):
        texts = ["hello world is a test"] * 100
        stats = compute_compression(mock_tokenizer, texts)
        assert stats.total_chars == 2100
        assert stats.bytes_per_token > 0


# ---- coverage edge cases ----


class TestCoverageEdgeCases:
    def test_coverage_for_text_unicode(self, mock_tokenizer):
        result = coverage_for_text(mock_tokenizer, "日本語hello")
        assert result["unique_chars"] > 0

    def test_detect_script_digits(self):
        result = detect_script("12345")
        assert "DIGIT" in result
        assert result["DIGIT"] == 5

    def test_detect_script_pure_whitespace(self):
        result = detect_script("   \t\n")
        assert result == {}

    def test_coverage_cyrillic(self, mock_tokenizer):
        result = analyze_coverage(mock_tokenizer, blocks=["Cyrillic"])
        assert result.total_codepoints_tested > 0


# ---- cost edge cases ----


class TestCostEdgeCases:
    def test_large_corpus_cost(self, mock_tokenizer):
        texts = ["hello world"] * 1000
        result = estimate_cost(mock_tokenizer, texts, provider_name="gpt-4o")
        assert result["total_tokens"] > 0
        assert result["input_cost_usd"] > 0

    def test_compare_costs_single_entry(self, mock_tokenizer):
        costs = compare_costs([(mock_tokenizer, "gpt-4o")], ["hello"])
        assert len(costs.estimates) == 1


# ---- loader edge cases ----


class TestLoaderEdgeCases:
    def test_encode_unknown_chars(self, mock_tokenizer):
        # Encoding chars not in vocab should fallback
        ids = mock_tokenizer.encode("\x80\x81")
        assert len(ids) > 0

    def test_tokenize_returns_spans(self, mock_tokenizer):
        spans = mock_tokenizer.tokenize("hello world")
        assert all(isinstance(s, TokenSpan) for s in spans)
        assert sum(s.char_length for s in spans) == len("hello world")

    def test_vocab_mutation_safety(self, mock_tokenizer):
        v = mock_tokenizer.vocab
        original_size = mock_tokenizer.vocab_size
        v["MUTATION_TEST"] = 99999
        assert mock_tokenizer.vocab_size == original_size

    def test_info_metadata(self, mock_tokenizer):
        info = mock_tokenizer.info()
        assert info.metadata == {}


# ---- mapping edge cases ----


class TestMappingEdgeCases:
    def test_unicode_mapping(self, mock_tokenizer, mock_tokenizer_b):
        result = map_tokens(mock_tokenizer, mock_tokenizer_b, "日本")
        assert len(result) > 0

    def test_expansion_ratio_single_text(self, mock_tokenizer, mock_tokenizer_b):
        result = token_expansion_ratio(mock_tokenizer, mock_tokenizer_b, ["hello"])
        assert result["count"] == 1.0


# ---- stats edge cases ----


class TestStatsEdgeCases:
    def test_top_tokens_exceeds_vocab(self, mock_tokenizer):
        result = top_tokens_by_length(mock_tokenizer, n=99999)
        # Result may be <= vocab_size since some id_to_token entries may share keys
        assert len(result) <= mock_tokenizer.vocab_size
        assert len(result) > 0

    def test_histogram_covers_vocab(self, mock_tokenizer):
        hist = token_length_histogram(mock_tokenizer)
        total = sum(hist.values())
        assert total <= mock_tokenizer.vocab_size
        assert total > 0


# ---- types edge cases ----


class TestTypesEdgeCases:
    def test_compression_stats_high_ratio(self):
        stats = CompressionStats(
            total_chars=10000, total_bytes=10000, total_tokens=100,
            bytes_per_token=100.0, chars_per_token=100.0,
        )
        assert stats.compression_ratio == 100.0

    def test_compare_result_symmetric(self):
        r = CompareResult(
            tokenizer_a="a", tokenizer_b="b",
            vocab_overlap=100, vocab_only_a=100, vocab_only_b=100,
            jaccard_similarity=1 / 3,
        )
        assert r.overlap_ratio_a == r.overlap_ratio_b

    def test_audit_finding_metadata(self):
        f = AuditFinding("test", "info", 0, "t", "desc", metadata={"key": "val"})
        assert f.metadata["key"] == "val"

    def test_audit_result_no_findings(self):
        r = AuditResult(tokenizer_name="test")
        assert r.by_category == {}
        assert r.n_critical == 0
        assert r.n_warnings == 0

    def test_vocab_stats_defaults(self):
        vs = VocabStats(
            vocab_size=100, n_special_tokens=0, n_byte_tokens=0,
            n_single_char=50, n_multiword=10, avg_token_length=3.0, max_token_length=10,
        )
        assert vs.length_distribution == {}
        assert vs.script_coverage == {}
