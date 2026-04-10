"""GGUF tokenizer extraction — load vocab from GGUF model files."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from toksight._types import TokenizerBackend, ToksightError
from toksight.loader import TokenizerWrapper

# GGUF metadata value types
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

_GGUF_MAGIC = 0x46554747  # b"GGUF"


def _read_string(data: bytes, offset: int) -> Tuple[str, int]:
    """Read a GGUF string (uint64 length + bytes)."""
    (length,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    value = data[offset : offset + length].decode("utf-8", errors="replace")
    return value, offset + length


def _read_value(data: bytes, offset: int, vtype: int) -> Tuple[Any, int]:
    """Read a single GGUF metadata value."""
    if vtype == _GGUF_TYPE_UINT8:
        (v,) = struct.unpack_from("<B", data, offset)
        return v, offset + 1
    elif vtype == _GGUF_TYPE_INT8:
        (v,) = struct.unpack_from("<b", data, offset)
        return v, offset + 1
    elif vtype == _GGUF_TYPE_UINT16:
        (v,) = struct.unpack_from("<H", data, offset)
        return v, offset + 2
    elif vtype == _GGUF_TYPE_INT16:
        (v,) = struct.unpack_from("<h", data, offset)
        return v, offset + 2
    elif vtype == _GGUF_TYPE_UINT32:
        (v,) = struct.unpack_from("<I", data, offset)
        return v, offset + 4
    elif vtype == _GGUF_TYPE_INT32:
        (v,) = struct.unpack_from("<i", data, offset)
        return v, offset + 4
    elif vtype == _GGUF_TYPE_FLOAT32:
        (v,) = struct.unpack_from("<f", data, offset)
        return v, offset + 4
    elif vtype == _GGUF_TYPE_BOOL:
        (v,) = struct.unpack_from("<B", data, offset)
        return bool(v), offset + 1
    elif vtype == _GGUF_TYPE_STRING:
        return _read_string(data, offset)
    elif vtype == _GGUF_TYPE_UINT64:
        (v,) = struct.unpack_from("<Q", data, offset)
        return v, offset + 8
    elif vtype == _GGUF_TYPE_INT64:
        (v,) = struct.unpack_from("<q", data, offset)
        return v, offset + 8
    elif vtype == _GGUF_TYPE_FLOAT64:
        (v,) = struct.unpack_from("<d", data, offset)
        return v, offset + 8
    elif vtype == _GGUF_TYPE_ARRAY:
        (elem_type,) = struct.unpack_from("<I", data, offset)
        offset += 4
        (count,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        items = []
        for _ in range(count):
            item, offset = _read_value(data, offset, elem_type)
            items.append(item)
        return items, offset
    else:
        raise ToksightError(f"Unknown GGUF metadata type: {vtype}")


def _parse_gguf_metadata(data: bytes) -> Dict[str, Any]:
    """Parse GGUF header and return all metadata key-value pairs."""
    if len(data) < 24:
        raise ToksightError("File too small to be a valid GGUF file")

    magic, version, tensor_count, metadata_count = struct.unpack_from("<IIqq", data, 0)

    if magic != _GGUF_MAGIC:
        raise ToksightError(
            f"Invalid GGUF magic: expected 0x{_GGUF_MAGIC:08X}, "
            f"got 0x{magic:08X}"
        )
    if version not in (2, 3):
        raise ToksightError(f"Unsupported GGUF version: {version}")

    offset = 4 + 4 + 8 + 8  # magic(4) + version(4) + tensor_count(8) + metadata_count(8)

    metadata: Dict[str, Any] = {}
    for _ in range(metadata_count):
        key, offset = _read_string(data, offset)
        (vtype,) = struct.unpack_from("<I", data, offset)
        offset += 4
        value, offset = _read_value(data, offset, vtype)
        metadata[key] = value

    metadata["_gguf_version"] = version
    metadata["_tensor_count"] = tensor_count
    return metadata


def load_gguf(path: str | Path) -> TokenizerWrapper:
    """Load a tokenizer from a GGUF file.

    Extracts vocabulary from ``tokenizer.ggml.tokens`` metadata and
    optional scores from ``tokenizer.ggml.scores``.

    Parameters
    ----------
    path:
        Path to the ``.gguf`` file.

    Returns
    -------
    TokenizerWrapper
        A tokenizer wrapper with encode/decode backed by the extracted vocab.
    """
    path = Path(path)
    if not path.exists():
        raise ToksightError(f"GGUF file not found: {path}")

    data = path.read_bytes()
    metadata = _parse_gguf_metadata(data)

    tokens: Optional[List[str]] = metadata.get("tokenizer.ggml.tokens")
    if tokens is None:
        raise ToksightError("GGUF file does not contain tokenizer.ggml.tokens")

    scores: Optional[List[float]] = metadata.get("tokenizer.ggml.scores")

    # Build vocab mapping
    vocab: Dict[str, int] = {}
    for idx, token in enumerate(tokens):
        vocab[token] = idx

    id_to_token = {v: k for k, v in vocab.items()}

    # Detect special tokens from metadata
    special_tokens: List[str] = []
    for key in ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
                "tokenizer.ggml.unknown_token_id", "tokenizer.ggml.padding_token_id"):
        tid = metadata.get(key)
        if tid is not None and isinstance(tid, int) and tid in id_to_token:
            special_tokens.append(id_to_token[tid])

    # Simple encode: greedy longest-prefix match
    def encode_fn(text: str) -> List[int]:
        ids: List[int] = []
        i = 0
        while i < len(text):
            best_len = 0
            best_id = vocab.get("<unk>", 0)  # fallback to unk or 0
            for length in range(min(len(text) - i, 64), 0, -1):
                candidate = text[i : i + length]
                if candidate in vocab:
                    best_len = length
                    best_id = vocab[candidate]
                    break
            if best_len == 0:
                best_len = 1  # skip one character
            ids.append(best_id)
            i += best_len
        return ids

    def decode_fn(ids: List[int]) -> str:
        parts: List[str] = []
        for tid in ids:
            parts.append(id_to_token.get(tid, "<unk>"))
        return "".join(parts)

    extra_meta: Dict[str, Any] = {
        "gguf_version": metadata.get("_gguf_version"),
        "tensor_count": metadata.get("_tensor_count"),
    }
    if scores is not None:
        extra_meta["has_scores"] = True
        extra_meta["vocab_size"] = len(tokens)

    model_name = metadata.get("general.name", path.stem)

    return TokenizerWrapper(
        name=str(model_name),
        backend=TokenizerBackend.CUSTOM,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        vocab=vocab,
        special_tokens=special_tokens,
        metadata=extra_meta,
    )
