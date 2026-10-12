"""Strict response parser, a versioned alternative to the registered parser (STRICT_PARSER_VERSION).

The registered parser, fairscreen.runtime.parse_response, is unchanged and produced every registered result. It
tolerates trailing commas, takes the last JSON object that carries a decision, normalizes label spelling, and falls
back to a standalone upper-case label token in prose. This parser has none of those allowances:

    parse_strict(text) -> "MEETS" | "DOES_NOT_MEET" | "INSUFFICIENT_EVIDENCE" | None

It accepts only a response that is, after stripping surrounding whitespace, one complete JSON object (optionally
wrapped in a single ```json fence) with a top-level "decision" key whose value is exactly one of the three labels.
Anything else returns None: prose around the object, a label in prose ("I cannot conclude MEETS."), a truncated
object, trailing commas, duplicate keys, a lower-case or misspelled label, or a non-string decision.
"""
import json
import re

STRICT_PARSER_VERSION = "1.0"
LABELS = ("MEETS", "DOES_NOT_MEET", "INSUFFICIENT_EVIDENCE")  # same strings as fairscreen.runtime.LABELS

_FENCE = re.compile(r"```(?:json|JSON)?[ \t]*\r?\n(.*?)\r?\n?[ \t]*```", re.DOTALL)


def _no_duplicate_keys(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key")
    return dict(pairs)


def _unwrap_fence(text):
    """Return the body of a single fenced block that is the whole text; otherwise the text unchanged."""
    if not text.startswith("```"):
        return text
    m = _FENCE.fullmatch(text)
    return m.group(1).strip() if m else None


def parse_strict(text):
    """Return the decision label if text is one complete JSON object with a valid "decision", else None."""
    if not isinstance(text, str):
        return None
    body = _unwrap_fence(text.strip())
    if body is None or not body.startswith("{"):
        return None
    try:
        obj = json.loads(body, object_pairs_hook=_no_duplicate_keys)
    except ValueError:  # includes JSONDecodeError (truncated or malformed) and duplicate keys
        return None
    if not isinstance(obj, dict):
        return None
    decision = obj.get("decision")
    return decision if isinstance(decision, str) and decision in LABELS else None
