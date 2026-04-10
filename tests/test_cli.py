"""Tests for CLI module — covers all 4 commands with mocked tokenizer loading."""

import json
import os

import pytest
from click.testing import CliRunner

from toksight.loader import wrap_custom


def _mock_tokenizer():
    """Minimal tokenizer for CLI testing."""
    vocab = {chr(i): i for i in range(32, 127)}
    vocab.update({"hello": 200, "world": 201, " ": 32})
    reverse = {v: k for k, v in vocab.items()}

    def encode(text):
        ids = []
        i = 0
        sorted_tokens = sorted(vocab.keys(), key=len, reverse=True)
        while i < len(text):
            for tok in sorted_tokens:
                if text[i : i + len(tok)] == tok:
                    ids.append(vocab[tok])
                    i += len(tok)
                    break
            else:
                ids.append(vocab.get("?", 63))
                i += 1
        return ids

    def decode(ids):
        return "".join(reverse.get(i, "?") for i in ids)

    return wrap_custom(
        name="test_enc",
        encode_fn=encode,
        decode_fn=decode,
        vocab=vocab,
        special_tokens=[],
    )


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cli_app(monkeypatch):
    """Return CLI with load_tiktoken patched to return mock tokenizer."""
    mock_tok = _mock_tokenizer()

    import toksight.cli as cli_mod

    monkeypatch.setattr(cli_mod, "cli", None)  # force rebuild

    import importlib

    # Patch load_tiktoken at the module it's imported from
    import toksight.loader as loader_mod

    original = loader_mod.load_tiktoken
    monkeypatch.setattr(loader_mod, "load_tiktoken", lambda name: mock_tok)

    # Rebuild CLI so it picks up the patched loader
    new_cli = cli_mod._build_cli()
    yield new_cli

    monkeypatch.setattr(loader_mod, "load_tiktoken", original)


class TestInfoCommand:
    def test_basic(self, runner, cli_app):
        result = runner.invoke(cli_app, ["info", "test_enc"])
        assert result.exit_code == 0
        assert "Vocab" in result.output or "vocab" in result.output.lower()

    def test_json_out(self, runner, cli_app, tmp_path):
        out = tmp_path / "stats.json"
        result = runner.invoke(cli_app, ["info", "test_enc", "--json-out", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "vocab_size" in data


class TestCompressCommand:
    def test_with_text(self, runner, cli_app):
        result = runner.invoke(cli_app, ["compress", "test_enc", "--text", "hello world"])
        assert result.exit_code == 0
        assert "token" in result.output.lower() or "Compression" in result.output

    def test_with_corpus_file(self, runner, cli_app, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("hello world\nthe test is a test\n")
        result = runner.invoke(cli_app, ["compress", "test_enc", "--corpus", str(corpus)])
        assert result.exit_code == 0

    def test_no_input_fails(self, runner, cli_app):
        result = runner.invoke(cli_app, ["compress", "test_enc"])
        assert result.exit_code != 0


class TestCoverageCommand:
    def test_basic(self, runner, cli_app):
        result = runner.invoke(cli_app, ["coverage", "test_enc"])
        assert result.exit_code == 0

    def test_with_blocks(self, runner, cli_app):
        result = runner.invoke(cli_app, ["coverage", "test_enc", "-b", "Basic Latin"])
        assert result.exit_code == 0

    def test_json_out(self, runner, cli_app, tmp_path):
        out = tmp_path / "cov.json"
        result = runner.invoke(
            cli_app, ["coverage", "test_enc", "-b", "Basic Latin", "--json-out", str(out)]
        )
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "coverage_ratio" in data


class TestAuditCommand:
    def test_basic(self, runner, cli_app):
        result = runner.invoke(cli_app, ["audit", "test_enc"])
        assert result.exit_code == 0

    def test_with_max_tokens(self, runner, cli_app):
        result = runner.invoke(cli_app, ["audit", "test_enc", "--max-tokens", "10"])
        assert result.exit_code == 0
