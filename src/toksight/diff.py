"""Visual token diff between two tokenizers."""

from __future__ import annotations

import html as _html
from dataclasses import dataclass
from typing import List

from toksight.loader import TokenizerWrapper


@dataclass
class TokenDiffResult:
    """Result of diffing tokenization of the same text by two tokenizers."""

    text: str
    tokens_a: List[str]
    tokens_b: List[str]
    boundaries_a: List[int]
    boundaries_b: List[int]
    common_count: int
    diff_count: int


def token_diff(
    text: str,
    tokenizer_a: TokenizerWrapper,
    tokenizer_b: TokenizerWrapper,
) -> TokenDiffResult:
    """Tokenize *text* with two tokenizers and produce a visual diff result.

    Boundaries are cumulative character offsets where each token ends.
    ``common_count`` is the number of boundary positions shared by both
    tokenizers; ``diff_count`` is the number present in only one.
    """
    spans_a = tokenizer_a.tokenize(text)
    spans_b = tokenizer_b.tokenize(text)

    tokens_a = [s.text for s in spans_a]
    tokens_b = [s.text for s in spans_b]

    boundaries_a = _cumulative_boundaries(spans_a)
    boundaries_b = _cumulative_boundaries(spans_b)

    set_a = set(boundaries_a)
    set_b = set(boundaries_b)
    common = set_a & set_b
    diff = (set_a | set_b) - common

    return TokenDiffResult(
        text=text,
        tokens_a=tokens_a,
        tokens_b=tokens_b,
        boundaries_a=boundaries_a,
        boundaries_b=boundaries_b,
        common_count=len(common),
        diff_count=len(diff),
    )


def _cumulative_boundaries(spans: list) -> List[int]:
    boundaries: List[int] = []
    pos = 0
    for span in spans:
        pos += span.char_length
        boundaries.append(pos)
    return boundaries


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def format_diff_text(result: TokenDiffResult) -> str:
    """Plain-text visualisation showing tokens side by side with markers."""
    set_a = set(result.boundaries_a)
    set_b = set(result.boundaries_b)

    def _fmt_tokens(tokens: List[str]) -> str:
        return " ".join(f"[{t}]" for t in tokens)

    lines = [
        f"Text: {result.text!r}",
        "",
        f"Tokenizer A ({len(result.tokens_a)} tokens):",
        f"  {_fmt_tokens(result.tokens_a)}",
        f"Tokenizer B ({len(result.tokens_b)} tokens):",
        f"  {_fmt_tokens(result.tokens_b)}",
        "",
        f"Boundaries A: {result.boundaries_a}",
        f"Boundaries B: {result.boundaries_b}",
        f"Common: {result.common_count}  Different: {result.diff_count}",
    ]

    all_bounds = sorted(set_a | set_b)
    if all_bounds:
        markers: List[str] = []
        for b in all_bounds:
            if b in set_a and b in set_b:
                markers.append(f"  {b}:=")
            elif b in set_a:
                markers.append(f"  {b}:A")
            else:
                markers.append(f"  {b}:B")
        lines.append("Alignment:" + "".join(markers))

    return "\n".join(lines)


def format_diff_html(result: TokenDiffResult) -> str:
    """HTML output with colour-coded token boundaries.

    Tokens whose right boundary is shared with the other tokenizer are
    shown in green; tokens whose boundary is unique are shown in red.
    """
    set_a = set(result.boundaries_a)
    set_b = set(result.boundaries_b)

    def _render_row(
        label: str,
        tokens: List[str],
        boundaries: List[int],
        other: set,
    ) -> str:
        spans: List[str] = []
        for token, boundary in zip(tokens, boundaries):
            bg = "#c8e6c9" if boundary in other else "#ffcdd2"
            escaped = _html.escape(token).replace(" ", "&nbsp;")
            spans.append(
                f'<span style="background:{bg};padding:2px 4px;'
                f'border-radius:2px;margin:1px">{escaped}</span>'
            )
        return (
            f"<div><strong>{_html.escape(label)}</strong>: "
            f"{''.join(spans)}</div>"
        )

    parts = [
        '<div style="font-family:monospace">',
        f"<div>Text: {_html.escape(result.text)}</div>",
        _render_row("A", result.tokens_a, result.boundaries_a, set_b),
        _render_row("B", result.tokens_b, result.boundaries_b, set_a),
        f"<div>Common: {result.common_count} | Different: {result.diff_count}</div>",
        "</div>",
    ]
    return "\n".join(parts)


def format_diff_inline(result: TokenDiffResult) -> str:
    """Single-line visualisation with ``|`` markers at token boundaries."""

    def _render(tokens: List[str]) -> str:
        if not tokens:
            return "||"
        return "|" + "|".join(tokens) + "|"

    return f"A: {_render(result.tokens_a)}\nB: {_render(result.tokens_b)}"
