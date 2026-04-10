"""Visualize how two tokenizers split the same text differently.

Demonstrates token_diff(), format_diff_text(), and format_diff_html().
"""

from toksight import format_diff_html, format_diff_text, token_diff, wrap_custom


def _char_tokenizer():
    """Character-level tokenizer for demonstration."""
    chars = [chr(i) for i in range(32, 127)]
    vocab = {c: i for i, c in enumerate(chars)}
    vocab["<unk>"] = len(vocab)
    id_to_tok = {v: k for k, v in vocab.items()}
    unk_id = vocab["<unk>"]

    def encode(text: str):
        return [vocab.get(c, unk_id) for c in text]

    def decode(ids):
        return "".join(id_to_tok.get(i, "?") for i in ids)

    return wrap_custom("char-level", encode, decode, vocab)


def _word_tokenizer():
    """Word-level tokenizer for demonstration."""
    words = [
        "hello", "world", "the", "quick", "brown", "fox", "jumps",
        "over", "lazy", "dog", "!", ".", ",", " ",
    ]
    vocab = {w: i for i, w in enumerate(words)}
    vocab["<unk>"] = len(vocab)
    id_to_tok = {v: k for k, v in vocab.items()}
    unk_id = vocab["<unk>"]

    def encode(text: str):
        tokens = []
        for word in text.split():
            tokens.append(vocab.get(word.lower().rstrip(".,!"), unk_id))
        return tokens

    def decode(ids):
        return " ".join(id_to_tok.get(i, "<unk>") for i in ids)

    return wrap_custom("word-level", encode, decode, vocab)


def main() -> None:
    tok_a = _char_tokenizer()
    tok_b = _word_tokenizer()

    print(f"Tokenizer A: {tok_a.name} (vocab={tok_a.vocab_size})")
    print(f"Tokenizer B: {tok_b.name} (vocab={tok_b.vocab_size})")
    print()

    text = "Hello world! The quick brown fox."

    # -- Compute diff -----------------------------------------------------
    result = token_diff(text, tok_a, tok_b)

    print(f"Text: '{result.text}'")
    print(f"A tokens: {len(result.tokens_a)}  B tokens: {len(result.tokens_b)}")
    print(f"Common boundaries: {result.common_count}  Different: {result.diff_count}")
    print()

    # -- Plain text diff --------------------------------------------------
    print("=== Text Diff ===")
    print(format_diff_text(result))

    # -- HTML diff (save to file) -----------------------------------------
    html = format_diff_html(result)
    out_path = "/tmp/toksight_diff.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\nHTML diff saved to: {out_path}")


if __name__ == "__main__":
    main()
