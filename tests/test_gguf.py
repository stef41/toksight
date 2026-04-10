"""Tests for GGUF tokenizer extraction."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import pytest

from toksight._types import ToksightError
from toksight.gguf import load_gguf


def _pack_string(s: str) -> bytes:
    """Pack a GGUF string: uint64 length + utf-8 bytes."""
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _pack_metadata_entry(key: str, vtype: int, value_bytes: bytes) -> bytes:
    """Pack a single GGUF metadata key-value entry."""
    return _pack_string(key) + struct.pack("<I", vtype) + value_bytes


def _pack_string_array(strings: list[str]) -> bytes:
    """Pack an array of strings (type=8, count, then items)."""
    data = struct.pack("<I", 8)  # element type = STRING
    data += struct.pack("<Q", len(strings))
    for s in strings:
        data += _pack_string(s)
    return data


def _pack_float_array(floats: list[float]) -> bytes:
    """Pack an array of float32 (type=6, count, then items)."""
    data = struct.pack("<I", 6)  # element type = FLOAT32
    data += struct.pack("<Q", len(floats))
    for f in floats:
        data += struct.pack("<f", f)
    return data


def _build_gguf(tokens: list[str], scores: list[float] | None = None,
                bos_id: int | None = None, eos_id: int | None = None,
                model_name: str | None = None) -> bytes:
    """Build a minimal valid GGUF file with tokenizer metadata."""
    metadata_entries: list[bytes] = []

    # tokenizer.ggml.tokens (required)
    metadata_entries.append(
        _pack_metadata_entry("tokenizer.ggml.tokens", 9, _pack_string_array(tokens))
    )

    # tokenizer.ggml.scores (optional)
    if scores is not None:
        metadata_entries.append(
            _pack_metadata_entry("tokenizer.ggml.scores", 9, _pack_float_array(scores))
        )

    # bos/eos token IDs
    if bos_id is not None:
        metadata_entries.append(
            _pack_metadata_entry("tokenizer.ggml.bos_token_id", 5,
                                 struct.pack("<i", bos_id))
        )
    if eos_id is not None:
        metadata_entries.append(
            _pack_metadata_entry("tokenizer.ggml.eos_token_id", 5,
                                 struct.pack("<i", eos_id))
        )

    # general.name
    if model_name is not None:
        name_bytes = _pack_string(model_name)
        metadata_entries.append(
            _pack_metadata_entry("general.name", 8, name_bytes)
        )

    metadata_count = len(metadata_entries)

    # Header: magic(4) + version(4) + tensor_count(8) + metadata_count(8)
    header = struct.pack("<I", 0x46554747)  # GGUF magic
    header += struct.pack("<I", 3)  # version 3
    header += struct.pack("<q", 0)  # tensor_count
    header += struct.pack("<q", metadata_count)

    return header + b"".join(metadata_entries)


def _write_gguf(tmp_path: Path, **kwargs) -> Path:
    """Write a GGUF file and return its path."""
    p = tmp_path / "test.gguf"
    p.write_bytes(_build_gguf(**kwargs))
    return p


class TestLoadGGUF:
    def test_basic_vocab(self, tmp_path: Path) -> None:
        tokens = ["<unk>", "<s>", "</s>", "hello", "world"]
        path = _write_gguf(tmp_path, tokens=tokens)

        wrapper = load_gguf(path)
        assert wrapper.vocab_size == 5
        assert wrapper.vocab["hello"] == 3
        assert wrapper.vocab["world"] == 4

    def test_encode_decode(self, tmp_path: Path) -> None:
        tokens = ["<unk>", "h", "e", "l", "o"]
        path = _write_gguf(tmp_path, tokens=tokens)

        wrapper = load_gguf(path)
        ids = wrapper.encode("hello")
        assert ids == [1, 2, 3, 3, 4]  # h=1, e=2, l=3, l=3, o=4
        assert wrapper.decode(ids) == "hello"

    def test_greedy_longest_match(self, tmp_path: Path) -> None:
        tokens = ["<unk>", "h", "he", "hel", "lo", "l", "o"]
        path = _write_gguf(tmp_path, tokens=tokens)

        wrapper = load_gguf(path)
        ids = wrapper.encode("hello")
        # Should prefer "hel" (3) + "lo" (4) over character-by-character
        assert ids == [3, 4]

    def test_with_scores(self, tmp_path: Path) -> None:
        tokens = ["<unk>", "a", "b", "c"]
        scores = [0.0, 1.0, 2.0, 3.0]
        path = _write_gguf(tmp_path, tokens=tokens, scores=scores)

        wrapper = load_gguf(path)
        assert wrapper._metadata["has_scores"] is True
        assert wrapper._metadata["vocab_size"] == 4

    def test_special_tokens(self, tmp_path: Path) -> None:
        tokens = ["<unk>", "<s>", "</s>", "a", "b"]
        path = _write_gguf(tmp_path, tokens=tokens, bos_id=1, eos_id=2)

        wrapper = load_gguf(path)
        assert "<s>" in wrapper.special_tokens
        assert "</s>" in wrapper.special_tokens

    def test_model_name(self, tmp_path: Path) -> None:
        tokens = ["<unk>", "a"]
        path = _write_gguf(tmp_path, tokens=tokens, model_name="test-model")

        wrapper = load_gguf(path)
        assert wrapper.name == "test-model"

    def test_invalid_magic(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.gguf"
        p.write_bytes(b"\x00" * 100)

        with pytest.raises(ToksightError, match="Invalid GGUF magic"):
            load_gguf(p)

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(ToksightError, match="not found"):
            load_gguf(tmp_path / "nonexistent.gguf")

    def test_no_vocab(self, tmp_path: Path) -> None:
        """GGUF file with no tokenizer.ggml.tokens should raise."""
        # Build a file with only a dummy metadata entry
        entry = _pack_metadata_entry(
            "general.name", 8, _pack_string("no-vocab-model")
        )
        header = struct.pack("<I", 0x46554747)
        header += struct.pack("<I", 3)
        header += struct.pack("<q", 0)
        header += struct.pack("<q", 1)
        p = tmp_path / "novocab.gguf"
        p.write_bytes(header + entry)

        with pytest.raises(ToksightError, match="does not contain"):
            load_gguf(p)

    def test_file_too_small(self, tmp_path: Path) -> None:
        p = tmp_path / "tiny.gguf"
        p.write_bytes(b"GGUF")

        with pytest.raises(ToksightError, match="too small"):
            load_gguf(p)
