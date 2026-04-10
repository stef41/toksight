"""Token cost estimation — compare provider costs based on tokenization differences."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from toksight._types import CostEstimate
from toksight.loader import TokenizerWrapper

# Approximate pricing per 1M tokens (USD) as of early 2026
# Input / Output prices
DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
}


def estimate_cost(
    tokenizer: TokenizerWrapper,
    texts: Sequence[str],
    pricing: Optional[Dict[str, float]] = None,
    provider_name: Optional[str] = None,
) -> Dict[str, object]:
    """Estimate the cost of tokenizing a corpus.

    Args:
        tokenizer: Tokenizer to use for counting.
        texts: Corpus of texts.
        pricing: Dict with 'input' and 'output' price per 1M tokens.
        provider_name: If given, look up pricing from DEFAULT_PRICING.
    """
    total_tokens = 0
    for text in texts:
        if text:
            total_tokens += len(tokenizer.encode(text))

    if pricing is None and provider_name and provider_name in DEFAULT_PRICING:
        pricing = DEFAULT_PRICING[provider_name]

    if pricing is None:
        pricing = {"input": 0.0, "output": 0.0}

    input_cost = (total_tokens / 1_000_000) * pricing.get("input", 0.0)
    output_cost = (total_tokens / 1_000_000) * pricing.get("output", 0.0)

    return {
        "tokenizer": tokenizer.name,
        "total_tokens": total_tokens,
        "total_chars": sum(len(t) for t in texts),
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "pricing": pricing,
    }


def compare_costs(
    tokenizers_with_pricing: Sequence[tuple],
    texts: Sequence[str],
) -> CostEstimate:
    """Compare costs across multiple tokenizer/pricing combinations.

    Args:
        tokenizers_with_pricing: List of (tokenizer, provider_name) or
            (tokenizer, pricing_dict) tuples.
        texts: Corpus of texts.
    """
    total_chars = sum(len(t) for t in texts)
    estimates: Dict[str, Dict] = {}

    for item in tokenizers_with_pricing:
        tokenizer = item[0]
        pricing_arg = item[1]

        if isinstance(pricing_arg, str):
            result = estimate_cost(tokenizer, texts, provider_name=pricing_arg)
        elif isinstance(pricing_arg, dict):
            result = estimate_cost(tokenizer, texts, pricing=pricing_arg)
        else:
            result = estimate_cost(tokenizer, texts)

        name = result.get("tokenizer", tokenizer.name)
        estimates[str(name)] = dict(result)

    return CostEstimate(
        corpus_chars=total_chars,
        estimates=estimates,
    )
