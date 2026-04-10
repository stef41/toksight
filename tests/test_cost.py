"""Tests for cost module."""

import pytest
from toksight.cost import compare_costs, estimate_cost, DEFAULT_PRICING


class TestEstimateCost:
    def test_basic(self, mock_tokenizer):
        result = estimate_cost(mock_tokenizer, ["hello world"])
        assert result["total_tokens"] > 0
        assert result["total_chars"] == 11

    def test_with_provider(self, mock_tokenizer):
        result = estimate_cost(mock_tokenizer, ["hello world"], provider_name="gpt-4o")
        assert result["input_cost_usd"] >= 0
        assert result["pricing"] == DEFAULT_PRICING["gpt-4o"]

    def test_with_custom_pricing(self, mock_tokenizer):
        pricing = {"input": 1.0, "output": 3.0}
        result = estimate_cost(mock_tokenizer, ["hello world"], pricing=pricing)
        assert result["pricing"] == pricing

    def test_empty_corpus(self, mock_tokenizer):
        result = estimate_cost(mock_tokenizer, [])
        assert result["total_tokens"] == 0
        assert result["input_cost_usd"] == 0.0

    def test_unknown_provider(self, mock_tokenizer):
        result = estimate_cost(mock_tokenizer, ["hello"], provider_name="unknown_model")
        # Falls back to zero pricing
        assert result["input_cost_usd"] == 0.0


class TestCompareCosts:
    def test_basic(self, mock_tokenizer, mock_tokenizer_b):
        costs = compare_costs(
            [(mock_tokenizer, "gpt-4o"), (mock_tokenizer_b, "gpt-4o-mini")],
            ["hello world"],
        )
        assert costs.corpus_chars == 11
        assert len(costs.estimates) == 2
        assert "mock" in costs.estimates
        assert "mock_b" in costs.estimates

    def test_with_custom_pricing(self, mock_tokenizer):
        costs = compare_costs(
            [(mock_tokenizer, {"input": 1.0, "output": 3.0})],
            ["hello"],
        )
        assert len(costs.estimates) == 1

    def test_empty_corpus(self, mock_tokenizer):
        costs = compare_costs([(mock_tokenizer, "gpt-4o")], [])
        assert costs.corpus_chars == 0


class TestDefaultPricing:
    def test_has_major_models(self):
        assert "gpt-4o" in DEFAULT_PRICING
        assert "claude-3.5-sonnet" in DEFAULT_PRICING
        assert "gemini-1.5-pro" in DEFAULT_PRICING

    def test_pricing_structure(self):
        for model, prices in DEFAULT_PRICING.items():
            assert "input" in prices
            assert "output" in prices
            assert prices["input"] >= 0
            assert prices["output"] >= prices["input"]
