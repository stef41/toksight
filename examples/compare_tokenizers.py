"""Load two tokenizers and compare their behavior on the same text.

Demonstrates load_tiktoken and wrap_custom to build tokenizer wrappers,
then compares vocabulary sizes, encoding lengths, and token boundaries.
"""

from toksight import load_tiktoken, wrap_custom


def _build_simple_tokenizer():
    """Build a trivial whitespace tokenizer for demonstration."""
    vocab = {}
    counter = 0
    corpus = "the quick brown fox jumps over lazy dog hello world".split()
    for word in corpus:
        if word not in vocab:
            vocab[word] = counter
            counter += 1
    # Add an unknown token
    vocab["<unk>"] = counter

    id_to_tok = {v: k for k, v in vocab.items()}
    unk_id = vocab["<unk>"]

    def encode(text: str):
        return [vocab.get(w, unk_id) for w in text.lower().split()]

    def decode(ids):
        return " ".join(id_to_tok.get(i, "<unk>") for i in ids)

    return wrap_custom("whitespace", encode, decode, vocab, ["<unk>"])


def main() -> None:
    # -- Load tokenizers --------------------------------------------------
    print("Loading tokenizers...")
    try:
        tok_a = load_tiktoken("cl100k_base")
    except Exception as exc:
        print(f"tiktoken not available: {exc}")
        print("Install with: pip install toksight[tiktoken]")
        return

    tok_b = _build_simple_tokenizer()

    print(f"  A: {tok_a.name} (vocab={tok_a.vocab_size:,})")
    print(f"  B: {tok_b.name} (vocab={tok_b.vocab_size:,})")
    print()

    # -- Compare on sample texts ------------------------------------------
    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world! This is a test of tokenization.",
        "Machine learning models use subword tokenization for efficiency.",
    ]

    print(f"{'Text':<55} {'A tokens':>9} {'B tokens':>9}")
    print("-" * 75)
    for text in samples:
        ids_a = tok_a.encode(text)
        ids_b = tok_b.encode(text)
        print(f"{text[:53]:<55} {len(ids_a):>9} {len(ids_b):>9}")

    # -- Detailed tokenization view ---------------------------------------
    print("\n=== Detailed Tokenization ===")
    sample = "The quick brown fox"
    spans_a = tok_a.tokenize(sample)
    spans_b = tok_b.tokenize(sample)
    print(f"Text: '{sample}'")
    print(f"  A: {[s.text for s in spans_a]}")
    print(f"  B: {[s.text for s in spans_b]}")


if __name__ == "__main__":
    main()
