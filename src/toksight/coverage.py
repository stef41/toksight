"""Unicode coverage analysis — how well does a tokenizer cover different scripts?"""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Optional, Set, Sequence, Tuple

from toksight._types import CoverageResult
from toksight.loader import TokenizerWrapper

# Major Unicode blocks with representative ranges
UNICODE_BLOCKS: Dict[str, List[Tuple[int, int]]] = {
    "Basic Latin": [(0x0020, 0x007E)],
    "Latin Extended": [(0x00C0, 0x024F)],
    "Cyrillic": [(0x0400, 0x04FF)],
    "Greek": [(0x0370, 0x03FF)],
    "Arabic": [(0x0600, 0x06FF)],
    "Devanagari": [(0x0900, 0x097F)],
    "CJK Unified": [(0x4E00, 0x9FFF)],
    "Hangul Syllables": [(0xAC00, 0xD7AF)],
    "Hiragana": [(0x3040, 0x309F)],
    "Katakana": [(0x30A0, 0x30FF)],
    "Thai": [(0x0E00, 0x0E7F)],
    "Hebrew": [(0x0590, 0x05FF)],
    "Bengali": [(0x0980, 0x09FF)],
    "Tamil": [(0x0B80, 0x0BFF)],
    "Emoji": [(0x1F600, 0x1F64F), (0x1F300, 0x1F5FF)],
    "Mathematical": [(0x2200, 0x22FF)],
    "CJK Symbols": [(0x3000, 0x303F)],
}

# Smaller sample sizes for very large blocks
_MAX_SAMPLE_PER_BLOCK = 500


def analyze_coverage(
    tokenizer: TokenizerWrapper,
    blocks: Optional[Sequence[str]] = None,
    sample_size: int = _MAX_SAMPLE_PER_BLOCK,
) -> CoverageResult:
    """Analyze which Unicode codepoints a tokenizer can handle without loss.

    A codepoint is 'covered' if encode→decode produces the original character.
    """
    target_blocks = blocks or list(UNICODE_BLOCKS.keys())
    total_tested = 0
    total_covered = 0
    block_results: Dict[str, Dict] = {}

    for block_name in target_blocks:
        if block_name not in UNICODE_BLOCKS:
            continue
        ranges = UNICODE_BLOCKS[block_name]

        # Collect valid codepoints
        codepoints: List[int] = []
        for start, end in ranges:
            for cp in range(start, min(end + 1, start + sample_size)):
                ch = chr(cp)
                # Skip unassigned/control characters
                cat = unicodedata.category(ch)
                if cat.startswith("C") and cat != "Co":
                    continue
                codepoints.append(cp)

        tested = 0
        covered = 0
        sample_uncovered: List[str] = []

        for cp in codepoints[:sample_size]:
            ch = chr(cp)
            tested += 1
            try:
                ids = tokenizer.encode(ch)
                decoded = tokenizer.decode(ids)
                # Check if roundtrip preserves the character
                if ch in decoded:
                    covered += 1
                else:
                    if len(sample_uncovered) < 5:
                        sample_uncovered.append(f"U+{cp:04X} ({ch})")
            except Exception:
                if len(sample_uncovered) < 5:
                    sample_uncovered.append(f"U+{cp:04X} (encode error)")

        ratio = covered / tested if tested > 0 else 0.0
        block_results[block_name] = {
            "tested": tested,
            "covered": covered,
            "ratio": round(ratio, 4),
            "sample_uncovered": sample_uncovered,
        }
        total_tested += tested
        total_covered += covered

    overall_ratio = total_covered / total_tested if total_tested > 0 else 0.0

    return CoverageResult(
        total_codepoints_tested=total_tested,
        codepoints_covered=total_covered,
        coverage_ratio=round(overall_ratio, 4),
        blocks_analyzed=block_results,
    )


def coverage_for_text(
    tokenizer: TokenizerWrapper,
    text: str,
) -> Dict[str, object]:
    """Analyze how well a tokenizer handles the actual characters in a given text."""
    unique_chars = set(text)
    tested = 0
    covered = 0
    uncovered_chars: List[str] = []

    for ch in sorted(unique_chars):
        if ch.isspace():
            continue
        tested += 1
        try:
            ids = tokenizer.encode(ch)
            decoded = tokenizer.decode(ids)
            if ch in decoded:
                covered += 1
            else:
                uncovered_chars.append(ch)
        except Exception:
            uncovered_chars.append(ch)

    return {
        "unique_chars": tested,
        "covered": covered,
        "uncovered": len(uncovered_chars),
        "coverage_ratio": round(covered / tested, 4) if tested > 0 else 1.0,
        "uncovered_sample": uncovered_chars[:20],
    }


def detect_script(text: str) -> Dict[str, int]:
    """Count characters per Unicode script in a text."""
    scripts: Dict[str, int] = {}
    for ch in text:
        if ch.isspace():
            continue
        try:
            script = unicodedata.name(ch, "").split()[0]
        except (ValueError, IndexError):
            script = "UNKNOWN"
        scripts[script] = scripts.get(script, 0) + 1
    return scripts
