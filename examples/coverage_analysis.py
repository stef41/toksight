"""Estimate vocabulary coverage on a text corpus.

Demonstrates estimate_coverage() and format_coverage_report() to find
how much of a tokenizer's vocabulary is actually used by a given corpus.
"""

from toksight import estimate_coverage, format_coverage_report, wrap_custom


def _build_toy_tokenizer():
    """Build a small tokenizer with extra unused vocab entries."""
    words = [
        "the", "cat", "sat", "on", "mat", "dog", "ran", "fast",
        "hello", "world", "python", "code", "data", "model", "train",
        "unused_a", "unused_b", "unused_c", "unused_d", "unused_e",
    ]
    vocab = {w: i for i, w in enumerate(words)}
    vocab["<unk>"] = len(vocab)
    id_to_tok = {v: k for k, v in vocab.items()}
    unk_id = vocab["<unk>"]

    def encode(text: str):
        return [vocab.get(w, unk_id) for w in text.lower().split()]

    def decode(ids):
        return " ".join(id_to_tok.get(i, "<unk>") for i in ids)

    return wrap_custom("toy-tokenizer", encode, decode, vocab, ["<unk>"])


def main() -> None:
    tokenizer = _build_toy_tokenizer()
    print(f"Tokenizer: {tokenizer.name}  (vocab size: {tokenizer.vocab_size})")
    print()

    # A small corpus that only uses some vocabulary entries
    corpus = [
        "the cat sat on the mat",
        "the dog ran fast",
        "the cat ran on the mat",
        "hello world hello world",
        "python code data model train",
    ]

    print(f"Corpus: {len(corpus)} texts")
    print()

    # -- Estimate coverage ------------------------------------------------
    result = estimate_coverage(tokenizer, corpus, top_k=10)

    print(f"=== Coverage Results ===")
    print(f"  Vocab used:  {result.vocab_used} / {result.vocab_total}")
    print(f"  Coverage:    {result.coverage_ratio:.1%}")
    print()

    print(f"  Most frequent tokens:")
    for token, count in result.most_frequent[:10]:
        print(f"    {token!r:>15}: {count}")

    print(f"\n  Rare tokens:")
    for token, count in result.rare_tokens[:5]:
        print(f"    {token!r:>15}: {count}")

    print(f"\n  Unused tokens ({len(result.unused_tokens)}):")
    for token in result.unused_tokens[:10]:
        print(f"    {token!r}")

    # -- Formatted report -------------------------------------------------
    print("\n=== Formatted Report ===")
    print(format_coverage_report(result))


if __name__ == "__main__":
    main()
