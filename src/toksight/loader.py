"""Tokenizer loading and wrapping — unified interface over multiple backends."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from toksight._types import TokenizerBackend, TokenizerInfo, TokenSpan, ToksightError


class TokenizerWrapper:
    """Unified tokenizer interface wrapping different backends."""

    def __init__(
        self,
        name: str,
        backend: TokenizerBackend,
        encode_fn: Callable[[str], List[int]],
        decode_fn: Callable[[List[int]], str],
        vocab: Dict[str, int],
        special_tokens: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.backend = backend
        self._encode = encode_fn
        self._decode = decode_fn
        self._vocab = vocab
        self._id_to_token = {v: k for k, v in vocab.items()}
        self._special_tokens = special_tokens or []
        self._metadata = metadata or {}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def vocab(self) -> Dict[str, int]:
        return dict(self._vocab)

    @property
    def id_to_token(self) -> Dict[int, str]:
        return dict(self._id_to_token)

    @property
    def special_tokens(self) -> List[str]:
        return list(self._special_tokens)

    def info(self) -> TokenizerInfo:
        return TokenizerInfo(
            name=self.name,
            backend=self.backend,
            vocab_size=self.vocab_size,
            special_tokens=self._special_tokens,
            metadata=self._metadata,
        )

    def encode(self, text: str) -> List[int]:
        return self._encode(text)

    def decode(self, ids: List[int]) -> str:
        return self._decode(ids)

    def decode_single(self, token_id: int) -> str:
        """Decode a single token ID to text."""
        return self._decode([token_id])

    def tokenize(self, text: str) -> List[TokenSpan]:
        """Tokenize text and return detailed spans."""
        ids = self.encode(text)
        spans: List[TokenSpan] = []
        offset = 0
        for tid in ids:
            token_text = self.decode_single(tid)
            byte_len = len(token_text.encode("utf-8", errors="replace"))
            char_len = len(token_text)
            spans.append(
                TokenSpan(
                    token_id=tid,
                    text=token_text,
                    byte_length=byte_len,
                    char_offset=offset,
                    char_length=char_len,
                )
            )
            offset += char_len
        return spans


def load_tiktoken(encoding_name: str) -> TokenizerWrapper:
    """Load a tiktoken tokenizer by encoding name (e.g., 'cl100k_base', 'o200k_base')."""
    try:
        import tiktoken
    except ImportError:
        raise ToksightError("tiktoken not installed: pip install toksight[tiktoken]")

    enc = tiktoken.get_encoding(encoding_name)

    # Build vocab from token bytes
    vocab: Dict[str, int] = {}
    for token_id in range(enc.n_vocab):
        try:
            token_bytes = enc.decode_single_token_bytes(token_id)
            token_str = token_bytes.decode("utf-8", errors="replace")
            vocab[token_str] = token_id
        except (KeyError, ValueError):
            pass

    special = list(enc._special_tokens.keys()) if hasattr(enc, "_special_tokens") else []

    return TokenizerWrapper(
        name=encoding_name,
        backend=TokenizerBackend.TIKTOKEN,
        encode_fn=enc.encode,
        decode_fn=enc.decode,
        vocab=vocab,
        special_tokens=special,
        metadata={"n_vocab": enc.n_vocab},
    )


def load_huggingface(model_name_or_path: str) -> TokenizerWrapper:
    """Load a HuggingFace tokenizer."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ToksightError(
            "transformers not installed: pip install toksight[transformers]"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    vocab = dict(tokenizer.get_vocab())
    special = list(tokenizer.all_special_tokens) if hasattr(tokenizer, "all_special_tokens") else []

    return TokenizerWrapper(
        name=model_name_or_path,
        backend=TokenizerBackend.HUGGINGFACE,
        encode_fn=lambda text: tokenizer.encode(text, add_special_tokens=False),
        decode_fn=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
        vocab=vocab,
        special_tokens=special,
        metadata={"model_max_length": getattr(tokenizer, "model_max_length", None)},
    )


def load_sentencepiece(model_path: str) -> TokenizerWrapper:
    """Load a SentencePiece model (.model file)."""
    try:
        import sentencepiece as spm
    except ImportError:
        raise ToksightError(
            "sentencepiece not installed: pip install toksight[sentencepiece]"
        )

    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)

    vocab: Dict[str, int] = {}
    for i in range(sp.GetPieceSize()):
        vocab[sp.IdToPiece(i)] = i

    return TokenizerWrapper(
        name=model_path,
        backend=TokenizerBackend.SENTENCEPIECE,
        encode_fn=lambda text: sp.Encode(text),
        decode_fn=lambda ids: sp.Decode(ids),
        vocab=vocab,
        special_tokens=["<s>", "</s>", "<unk>", "<pad>"],
        metadata={"piece_size": sp.GetPieceSize()},
    )


def wrap_custom(
    name: str,
    encode_fn: Callable[[str], List[int]],
    decode_fn: Callable[[List[int]], str],
    vocab: Dict[str, int],
    special_tokens: Optional[List[str]] = None,
) -> TokenizerWrapper:
    """Wrap a custom tokenizer with encode/decode functions."""
    return TokenizerWrapper(
        name=name,
        backend=TokenizerBackend.CUSTOM,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        vocab=vocab,
        special_tokens=special_tokens,
    )
