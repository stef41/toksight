"""Tests for coverage module."""

import pytest
from toksight.coverage import analyze_coverage, coverage_for_text, detect_script


class TestAnalyzeCoverage:
    def test_basic_latin(self, mock_tokenizer):
        result = analyze_coverage(mock_tokenizer, blocks=["Basic Latin"])
        assert result.total_codepoints_tested > 0
        assert result.codepoints_covered > 0
        assert "Basic Latin" in result.blocks_analyzed

    def test_multiple_blocks(self, mock_tokenizer):
        result = analyze_coverage(mock_tokenizer, blocks=["Basic Latin", "Cyrillic"])
        assert len(result.blocks_analyzed) == 2
        assert result.total_codepoints_tested > 0

    def test_all_blocks(self, mock_tokenizer):
        result = analyze_coverage(mock_tokenizer)
        assert len(result.blocks_analyzed) > 5

    def test_basic_latin_high_coverage(self, mock_tokenizer):
        """Mock tokenizer has full ASCII, should cover Basic Latin well."""
        result = analyze_coverage(mock_tokenizer, blocks=["Basic Latin"])
        block = result.blocks_analyzed["Basic Latin"]
        assert block["ratio"] > 0.5  # ASCII chars should mostly work

    def test_unknown_block(self, mock_tokenizer):
        result = analyze_coverage(mock_tokenizer, blocks=["NonExistentBlock"])
        assert result.total_codepoints_tested == 0

    def test_coverage_ratio(self, mock_tokenizer):
        result = analyze_coverage(mock_tokenizer, blocks=["Basic Latin"])
        assert 0.0 <= result.coverage_ratio <= 1.0

    def test_uncovered_count(self, mock_tokenizer):
        result = analyze_coverage(mock_tokenizer, blocks=["Basic Latin"])
        assert result.uncovered_count == (
            result.total_codepoints_tested - result.codepoints_covered
        )


class TestCoverageForText:
    def test_ascii(self, mock_tokenizer):
        result = coverage_for_text(mock_tokenizer, "hello world")
        assert result["unique_chars"] > 0
        assert result["coverage_ratio"] > 0

    def test_empty(self, mock_tokenizer):
        result = coverage_for_text(mock_tokenizer, "")
        assert result["unique_chars"] == 0
        assert result["coverage_ratio"] == 1.0

    def test_whitespace_only(self, mock_tokenizer):
        result = coverage_for_text(mock_tokenizer, "   ")
        assert result["unique_chars"] == 0


class TestDetectScript:
    def test_latin(self):
        result = detect_script("hello world")
        assert "LATIN" in result

    def test_mixed(self):
        result = detect_script("hello 日本")
        assert len(result) >= 2

    def test_empty(self):
        result = detect_script("")
        assert result == {}

    def test_whitespace(self):
        result = detect_script("   ")
        assert result == {}

    def test_numbers(self):
        result = detect_script("123")
        assert "DIGIT" in result
