"""Tests for toksight.overlap module."""

import pytest

from toksight.overlap import (
    OverlapResult,
    VocabOverlapAnalyzer,
    format_overlap_report,
    overlap_matrix,
)


@pytest.fixture
def analyzer():
    return VocabOverlapAnalyzer()


@pytest.fixture
def vocab_a():
    return {"cat", "dog", "bird", "fish"}


@pytest.fixture
def vocab_b():
    return {"dog", "fish", "snake", "lizard", "frog"}


# --- OverlapResult ---

def test_overlap_result_fields():
    r = OverlapResult("X", "Y", 3, 1, 2, 0.5, 0.75)
    assert r.tokenizer_a == "X"
    assert r.tokenizer_b == "Y"
    assert r.shared_count == 3
    assert r.only_a_count == 1
    assert r.only_b_count == 2
    assert r.jaccard == 0.5
    assert r.overlap_coefficient == 0.75


# --- compare ---

def test_compare_basic(analyzer, vocab_a, vocab_b):
    r = analyzer.compare(vocab_a, vocab_b, "A", "B")
    assert r.shared_count == 2  # dog, fish
    assert r.only_a_count == 2  # cat, bird
    assert r.only_b_count == 3  # snake, lizard, frog
    assert r.jaccard == pytest.approx(2 / 7)
    assert r.overlap_coefficient == pytest.approx(2 / 4)


def test_compare_identical(analyzer):
    v = {"a", "b", "c"}
    r = analyzer.compare(v, v)
    assert r.shared_count == 3
    assert r.only_a_count == 0
    assert r.only_b_count == 0
    assert r.jaccard == pytest.approx(1.0)
    assert r.overlap_coefficient == pytest.approx(1.0)


def test_compare_disjoint(analyzer):
    r = analyzer.compare({"a"}, {"b"})
    assert r.shared_count == 0
    assert r.jaccard == pytest.approx(0.0)
    assert r.overlap_coefficient == pytest.approx(0.0)


def test_compare_empty(analyzer):
    r = analyzer.compare(set(), set())
    assert r.jaccard == 0.0
    assert r.overlap_coefficient == 0.0


def test_compare_accepts_lists(analyzer):
    r = analyzer.compare(["a", "b"], ["b", "c"])
    assert r.shared_count == 1


# --- multi_compare ---

def test_multi_compare(analyzer):
    vocabs = {"X": {"a", "b"}, "Y": {"b", "c"}, "Z": {"a", "c"}}
    results = analyzer.multi_compare(vocabs)
    assert len(results) == 3  # C(3,2) pairs
    names = {(r.tokenizer_a, r.tokenizer_b) for r in results}
    assert ("X", "Y") in names
    assert ("X", "Z") in names
    assert ("Y", "Z") in names


def test_multi_compare_single_vocab(analyzer):
    results = analyzer.multi_compare({"only": {1, 2, 3}})
    assert results == []


# --- shared_tokens / unique_tokens ---

def test_shared_tokens(analyzer, vocab_a, vocab_b):
    shared = analyzer.shared_tokens(vocab_a, vocab_b)
    assert shared == {"dog", "fish"}


def test_unique_tokens(analyzer, vocab_a, vocab_b):
    only_a, only_b = analyzer.unique_tokens(vocab_a, vocab_b)
    assert only_a == {"cat", "bird"}
    assert only_b == {"snake", "lizard", "frog"}


# --- coverage ---

def test_coverage_full(analyzer):
    assert analyzer.coverage({"a", "b", "c"}, {"a", "b", "c"}) == pytest.approx(1.0)


def test_coverage_partial(analyzer):
    assert analyzer.coverage({"a"}, {"a", "b"}) == pytest.approx(0.5)


def test_coverage_empty_full(analyzer):
    assert analyzer.coverage({"a"}, set()) == 0.0


# --- merge_vocabularies ---

def test_merge_vocabularies(analyzer):
    merged = analyzer.merge_vocabularies([{"a", "b"}, {"b", "c"}, {"d"}])
    assert merged == {"a", "b", "c", "d"}


def test_merge_empty(analyzer):
    assert analyzer.merge_vocabularies([]) == set()


# --- overlap_matrix ---

def test_overlap_matrix():
    vocabs = {"X": {"a", "b"}, "Y": {"b", "c"}}
    m = overlap_matrix(vocabs)
    assert m["X"]["X"] == pytest.approx(1.0)
    assert m["Y"]["Y"] == pytest.approx(1.0)
    expected = 1 / 3  # intersection=1, union=3
    assert m["X"]["Y"] == pytest.approx(expected)
    assert m["Y"]["X"] == pytest.approx(expected)


def test_overlap_matrix_single():
    m = overlap_matrix({"A": {"x"}})
    assert m["A"]["A"] == pytest.approx(1.0)


# --- format_overlap_report ---

def test_format_report_single(analyzer, vocab_a, vocab_b):
    r = analyzer.compare(vocab_a, vocab_b, "GPT4", "Llama")
    text = format_overlap_report(r)
    assert "GPT4 vs Llama" in text
    assert "Shared tokens" in text
    assert "Jaccard" in text


def test_format_report_list(analyzer):
    results = analyzer.multi_compare({"A": {1, 2}, "B": {2, 3}, "C": {3, 4}})
    text = format_overlap_report(results)
    assert "A vs B" in text
    assert "B vs C" in text
