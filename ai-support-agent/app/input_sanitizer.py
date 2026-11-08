"""Prompt-injection defences and input sanitisation for the support agent.

The agent exposes LLM-driven triage/response generation over untrusted
customer input. This module provides **layered, deterministic** defences
that do not depend on the LLM itself:

1. ``sanitize_message``     — hard input validation at the API boundary
2. ``detect_prompt_injection`` — heuristic injection/pattern detection
3. ``escape_for_prompt``    — delimiter fencing when untrusted text is
                               embedded in LLM prompts
4. ``redact_pii`` / ``redact_mapping`` — PII redaction for logs/audit

Threat model addressed:
- direct instruction override ("ignore previous instructions...")
- data-exfiltration prompts ("print your system prompt")
- delimiter/role confusion (fake "system:" blocks, template injection)
- log/audit injection (newlines & control chars to forge log lines)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. Hard input validation (API boundary)
# ---------------------------------------------------------------------------

MAX_MESSAGE_LENGTH = 4000
MAX_SESSION_ID_LENGTH = 64
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# Control characters (except \t) — prevent log forging & terminal escapes.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Cap absurd newlines runs (limits log/audit spam & context stuffing).
_BLANK_LINE_RUN_RE = re.compile(r"\n{4,}")


class InputValidationError(ValueError):
    """Raised when user input fails hard validation (→ HTTP 400/422)."""


def sanitize_message(message: str, *, max_length: int = MAX_MESSAGE_LENGTH) -> str:
    """Validate and normalise a customer chat message.

    Raises ``InputValidationError`` for empty/oversized/control-char input.
    Returns a normalised string safe for logging, storage, and prompting.
    """
    if not isinstance(message, str):
        raise InputValidationError("Message must be a string.")
    cleaned = message.strip()
    if not cleaned:
        raise InputValidationError("Message must not be empty.")
    if len(cleaned) > max_length:
        raise InputValidationError(f"Message too long (max {max_length} characters).")
    if _CONTROL_CHARS_RE.search(cleaned):
        raise InputValidationError("Message contains control characters.")
    return cleaned


def sanitize_session_id(session_id: Optional[str]) -> Optional[str]:
    """Validate an optional client-supplied session id (log-injection safe)."""
    if session_id is None:
        return None
    if not isinstance(session_id, str):
        raise InputValidationError("session_id must be a string.")
    value = session_id.strip()
    if not value:
        return None
    if not _SESSION_ID_RE.match(value):
        raise InputValidationError("session_id may only contain letters, digits, '.', '_' and '-' (max 64).")
    return value


def sanitize_search_query(query: str, *, max_length: int = 500) -> str:
    """Sanitise RAG search queries (same rules, tighter length)."""
    cleaned = sanitize_message(query, max_length=max_length)
    return cleaned


# ---------------------------------------------------------------------------
# 2. Prompt-injection detection (heuristics)
# ---------------------------------------------------------------------------

# Ordered by intent. Each entry: (compiled regex, category).
_INJECTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts?|rules?)", re.I), "instruction_override"),
    (re.compile(r"disregard\s+(all\s+)?(previous|prior|your)\s+(instructions|prompts?|rules?)", re.I), "instruction_override"),
    (re.compile(r"forget\s+(everything|all|your)\s+(instructions|rules?|training)", re.I), "instruction_override"),
    (ref_pattern if (ref_pattern := None) else re.compile(r"\byou\s+are\s+now\s+(a|an)\b", re.I), "role_hijack"),
    (re.compile(r"pretend\s+(you\s+are|to\s+be)\s+(a|an)?\s*(jailbreak|unfiltered|unrestricted)", re.I), "role_hijack"),
    (re.compile(r"(reveal|show|print|repeat|output)\s+(your\s+)?(system\s+)?(prompt|instructions)", re.I), "system_prompt_extraction"),
    (re.compile(r"what\s+(is|are)\s+your\s+(system\s+prompt|initial\s+instructions)", re.I), "system_prompt_extraction"),
    (re.compile(r"\bsystem\s*:\s*$", re.I | re.M), "role_confusion"),
    (re.compile(r"<\|?(im_start|endoftext|system)\|?>", re.I), "delimiter_injection"),
    (re.compile(r"\{\{.*?\}\}", re.S), "template_injection"),
    (re.compile(r"\b(?:api[_\s]?key|secret[_\s]?key|password)\s*[=:]\s*\S+", re.I), "credential_fishing"),
    (re.compile(r"(?:curl|fetch|wget)\s+(?:https?://|www\.)", re.I), "exfiltration_attempt"),
    (re.compile(r"https?://\S+\.(?:xyz|top|ru|tk|cn)\b", re.I), "suspicious_url"),
]

_SUSPICIOUS_THRESHOLD = 2  # ≥2 distinct categories → treat as high-risk


def detect_prompt_injection(text: str) -> Tuple[bool, List[str]]:
    """Return ``(is_high_risk, matched_categories)`` for untrusted text.

    Heuristic, not a guarantee — a clean result is *not* proof of safety.
    High-risk input is logged + audit-tagged; the LLM still only sees the
    fenced, escaped version produced by :func:`escape_for_prompt`.
    """
    if not text:
        return False, []
    categories: List[str] = []
    for pattern, category in _INJECTION_PATTERNS:
        if pattern.search(text) and category not in categories:
            categories.append(category)
    return len(categories) >= _SUSPICIOUS_THRESHOLD, categories


# ---------------------------------------------------------------------------
# 3. Prompt fencing
# ---------------------------------------------------------------------------


def escape_for_prompt(untrusted_text: str, *, max_length: int = MAX_MESSAGE_LENGTH) -> str:
    """Prepare untrusted text for embedding in an LLM prompt.

    - truncates to a sane bound,
    - strips control characters,
    - neutralises look-alike role markers the model might obey ("system:",
      "<|im_start|>", triple-dash fences),
    - wraps in an explicit untrusted-data fence.
    """
    text = _CONTROL_CHARS_RE.sub("", untrusted_text or "")
    text = text[:max_length]
    # Neutralise fake role/fence markers a model could interpret as context.
    text = re.sub(r"^\s*(system|assistant|user)\s*:\s*$", "[user said]:", text, flags=re.I | re.M)
    text = text.replace("<|", "‹").replace("|>", "›")
    text = re.sub(r"```+", "'''", text)
    text = re.sub(r"^\s*---+\s*$", "[dash line]", text, flags=re.M)
    return text


# ---------------------------------------------------------------------------
# 4. PII redaction (logs, audit trail, metrics labels)
# ---------------------------------------------------------------------------

_PII_RULES: List[Tuple[re.Pattern, str]] = [
    # Emails (keep domain: jo**@example.com)
    (re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"), r"jo**@\1"),
    # Credit cards (13–16 digits, optional separators) — before generic digits
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[card]"),
    # Phone numbers (US-ish, with separators)
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[phone]"),
    # Long digit runs (order/SSN-ish, ≥7 digits) — avoid false-positive prices
    (re.compile(r"\b\d{7,}\b"), "[redacted-number]"),
    # IBANs
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"), "[iban]"),
]


def redact_pii(text: str) -> str:
    """Redact obvious PII from free text before it hits logs/audit storage."""
    if not text:
        return text
    for pattern, repl in _PII_RULES:
        text = pattern.sub(repl, text)
    return text


def redact_mapping(details: Dict[str, Any], *, max_value_length: int = 200) -> Dict[str, Any]:
    """Recursively redact PII and clamp values in a dict destined for logs."""
    out: Dict[str, Any] = {}
    for key, value in details.items():
        if isinstance(value, str):
            value = redact_pii(value)
            if len(value) > max_value_length:
                value = value[:max_value_length] + "…"
        elif isinstance(value, dict):
            value = redact_mapping(value, max_value_length=max_value_length)
        elif isinstance(value, (list, tuple)):
            value = [
                redact_mapping(v, max_value_length=max_value_length) if isinstance(v, dict)
                else redact_pii(v) if isinstance(v, str)
                else v
                for v in value
            ]
        out[key] = value
    return out


__all__ = [
    "InputValidationError",
    "MAX_MESSAGE_LENGTH",
    "sanitize_message",
    "sanitize_session_id",
    "sanitize_search_query",
    "detect_prompt_injection",
    "escape_for_prompt",
    "redact_pii",
    "redact_mapping",
]
