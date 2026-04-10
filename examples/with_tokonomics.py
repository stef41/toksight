#!/usr/bin/env python3
"""Integration: toksight + tokonomics — analyze tokenizer efficiency, estimate costs.

Flow: Use toksight to compare two tokenizers on a sample corpus, then use
tokonomics to show the dollar cost implications of the token count differences.

Install: pip install toksight tokonomics
"""

try:
    from toksight import (
        load_tiktoken, load_huggingface, TokenizerWrapper,
    )
    from toksight.compression import compute_compression
    from toksight.compare import compare_on_corpus
    from toksight.stats import vocab_stats
except ImportError:
    raise SystemExit("pip install toksight  # required for this example")

try:
    from tokonomics import estimate_cost, compare_models, format_comparison
except ImportError:
    raise SystemExit("pip install tokonomics  # required for this example")


SAMPLE_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Implement a binary search tree in Python with insert, delete, and search methods.",
    "La inteligencia artificial está transformando la industria tecnológica global.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "Explain the difference between supervised and unsupervised learning in detail.",
] * 20  # 100 samples


def main() -> None:
    # ── 1. Load two tokenizers with toksight ─────────────────────────
    print("=" * 60)
    print("STEP 1: Load and inspect tokenizers (toksight)")
    print("=" * 60)
    tok_a = load_tiktoken("cl100k_base")   # GPT-4o family
    tok_b = load_tiktoken("o200k_base")    # GPT-4o-mini / newer
    print(f"  Tokenizer A: {tok_a.name} (vocab: {len(tok_a.vocab):,})")
    print(f"  Tokenizer B: {tok_b.name} (vocab: {len(tok_b.vocab):,})")

    # ── 2. Compare compression on the corpus ─────────────────────────
    print("\nSTEP 2: Compression & efficiency comparison")
    comp_a = compute_compression(tok_a, SAMPLE_CORPUS)
    comp_b = compute_compression(tok_b, SAMPLE_CORPUS)
    print(f"  {tok_a.name}: {comp_a.total_tokens:,} tokens "
          f"({comp_a.bytes_per_token:.2f} bytes/tok)")
    print(f"  {tok_b.name}: {comp_b.total_tokens:,} tokens "
          f"({comp_b.bytes_per_token:.2f} bytes/tok)")
    delta = comp_a.total_tokens - comp_b.total_tokens
    direction = "fewer" if delta > 0 else "more"
    print(f"  → {tok_b.name} uses {abs(delta):,} {direction} tokens on this corpus")

    # ── 3. Cost those token counts via tokonomics ────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Estimate dollar cost difference (tokonomics)")
    print("=" * 60)
    cost_a = estimate_cost(model="gpt-4o", input_tokens=comp_a.total_tokens, output_tokens=0)
    cost_b = estimate_cost(model="gpt-4o-mini", input_tokens=comp_b.total_tokens, output_tokens=0)
    print(f"  gpt-4o    ({tok_a.name}): ${cost_a.total_cost:.6f} for {comp_a.total_tokens:,} tokens")
    print(f"  gpt-4o-mini ({tok_b.name}): ${cost_b.total_cost:.6f} for {comp_b.total_tokens:,} tokens")

    # ── 4. Scale to production ───────────────────────────────────────
    print("\nSTEP 4: Projected daily cost at 10M requests/day")
    scale = 10_000_000 / len(SAMPLE_CORPUS)
    daily_a = cost_a.total_cost * scale
    daily_b = cost_b.total_cost * scale
    print(f"  gpt-4o:      ${daily_a:,.2f}/day")
    print(f"  gpt-4o-mini: ${daily_b:,.2f}/day")
    print(f"  Savings:     ${daily_a - daily_b:,.2f}/day ({(1 - daily_b/daily_a)*100:.0f}%)")
    print("\nTokenizer analysis + cost projection complete.")


if __name__ == "__main__":
    main()
