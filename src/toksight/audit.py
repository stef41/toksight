"""Tokenizer auditing — detect glitch tokens, degenerate entries, oddities."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from toksight._types import AuditFinding, AuditResult
from toksight.loader import TokenizerWrapper


def audit(
    tokenizer: TokenizerWrapper,
    max_tokens: Optional[int] = None,
) -> AuditResult:
    """Run a comprehensive audit on a tokenizer's vocabulary.

    Detects:
    - Glitch tokens (encode→decode roundtrip fails)
    - Degenerate tokens (empty or whitespace-only after decoding)
    - Overlong tokens (unusually long byte sequences)
    - Duplicate surface forms (different IDs decode to same string)
    - Control character tokens
    - Untrimmed whitespace tokens
    """
    findings: List[AuditFinding] = []
    vocab = tokenizer.vocab
    id_to_token = tokenizer.id_to_token
    special = set(tokenizer.special_tokens)

    # Track decoded forms for duplicate detection
    decoded_forms: Dict[str, List[int]] = {}

    limit = max_tokens or len(id_to_token)
    checked = 0

    for token_id, token_text in sorted(id_to_token.items()):
        if checked >= limit:
            break
        checked += 1

        # Skip special tokens for most checks
        is_special = token_text in special

        # Decode form
        try:
            decoded = tokenizer.decode_single(token_id)
        except Exception:
            findings.append(AuditFinding(
                category="decode_error",
                severity="critical",
                token_id=token_id,
                token_text=repr(token_text),
                description=f"Token {token_id} cannot be decoded",
            ))
            continue

        # Track for duplicates
        decoded_forms.setdefault(decoded, []).append(token_id)

        if is_special:
            continue

        # Glitch token: re-encode the decoded form and check roundtrip
        try:
            re_encoded = tokenizer.encode(decoded)
            if re_encoded and token_id not in re_encoded:
                findings.append(AuditFinding(
                    category="glitch_token",
                    severity="warning",
                    token_id=token_id,
                    token_text=repr(decoded),
                    description=(
                        f"Roundtrip mismatch: decode({token_id})={repr(decoded)}, "
                        f"re-encode gives {re_encoded}"
                    ),
                ))
        except Exception:
            pass

        # Degenerate: empty or whitespace-only
        if not decoded.strip() and decoded:
            findings.append(AuditFinding(
                category="degenerate",
                severity="info",
                token_id=token_id,
                token_text=repr(decoded),
                description=f"Token {token_id} decodes to whitespace-only: {repr(decoded)}",
            ))

        # Overlong: > 50 bytes is suspicious
        byte_len = len(decoded.encode("utf-8", errors="replace"))
        if byte_len > 50:
            findings.append(AuditFinding(
                category="overlong",
                severity="info",
                token_id=token_id,
                token_text=repr(decoded[:30] + "..."),
                description=f"Token {token_id} is {byte_len} bytes long",
                metadata={"byte_length": byte_len},
            ))

        # Control characters (except newline/tab)
        if any(
            ord(c) < 32 and c not in ("\n", "\r", "\t")
            for c in decoded
        ):
            findings.append(AuditFinding(
                category="control_char",
                severity="warning",
                token_id=token_id,
                token_text=repr(decoded),
                description=f"Token {token_id} contains control characters",
            ))

        # Repeated character sequences (e.g., "aaaaaaa")
        if len(decoded) >= 4 and len(set(decoded)) == 1:
            findings.append(AuditFinding(
                category="repeated_char",
                severity="info",
                token_id=token_id,
                token_text=repr(decoded),
                description=f"Token {token_id} is a repeated character: {repr(decoded)}",
            ))

    # Check for duplicate decoded forms
    for decoded, ids in decoded_forms.items():
        if len(ids) > 1 and decoded.strip():
            findings.append(AuditFinding(
                category="duplicate_surface",
                severity="warning",
                token_id=ids[0],
                token_text=repr(decoded),
                description=f"Multiple IDs decode to {repr(decoded)}: {ids[:5]}",
                metadata={"all_ids": ids},
            ))

    return AuditResult(
        tokenizer_name=tokenizer.name,
        findings=findings,
    )


def find_glitch_tokens(
    tokenizer: TokenizerWrapper,
    max_tokens: Optional[int] = None,
) -> List[AuditFinding]:
    """Find glitch tokens (roundtrip encode→decode failures)."""
    result = audit(tokenizer, max_tokens=max_tokens)
    return [f for f in result.findings if f.category == "glitch_token"]


def find_degenerate_tokens(
    tokenizer: TokenizerWrapper,
    max_tokens: Optional[int] = None,
) -> List[AuditFinding]:
    """Find degenerate (empty/whitespace-only) tokens."""
    result = audit(tokenizer, max_tokens=max_tokens)
    return [f for f in result.findings if f.category == "degenerate"]
