"""Tests for analysis/strict_parser.py, the versioned strict alternative to the registered runtime.parse_response."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fairscreen import runtime as RT  # noqa: E402


@pytest.fixture(scope="module")
def parse_strict():
    """analysis/strict_parser.py as a module (analysis/ is a script folder, not a package)."""
    spec = importlib.util.spec_from_file_location("strict_parser_under_test", ROOT / "analysis" / "strict_parser.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.LABELS == tuple(RT.LABELS)
    return mod.parse_strict


@pytest.mark.parametrize("label", ["MEETS", "DOES_NOT_MEET", "INSUFFICIENT_EVIDENCE"])
def test_returns_label_for_valid_object(parse_strict, label):
    assert parse_strict('{"decision": "%s", "reason": "x"}' % label) == label


def test_returns_label_for_fenced_valid_object(parse_strict):
    text = '```json\n{"decision": "DOES_NOT_MEET", "facts": {"years": 2}}\n```'
    assert parse_strict(text) == "DOES_NOT_MEET"


def test_accepts_surrounding_whitespace(parse_strict):
    assert parse_strict('\n  {"decision": "MEETS"}  \n') == "MEETS"


def test_returns_none_for_label_in_prose(parse_strict):
    assert parse_strict("I cannot conclude MEETS.") is None


def test_returns_none_for_truncated_object(parse_strict):
    assert parse_strict('{"decision": "MEETS", "reason": "The applicant has') is None
    assert parse_strict('{"decision": "MEETS"') is None


@pytest.mark.parametrize("bad", ["MAYBE", "meets", "Does Not Meet", "", None, 1, ["MEETS"]])
def test_returns_none_for_invalid_label(parse_strict, bad):
    import json
    assert parse_strict(json.dumps({"decision": bad})) is None


@pytest.mark.parametrize("text", [None, "", "   ", 5, "[]", '["MEETS"]', '"MEETS"', "{}", '{"verdict": "MEETS"}'])
def test_returns_none_for_non_object_or_missing_decision(parse_strict, text):
    assert parse_strict(text) is None


def test_returns_none_for_prose_around_object(parse_strict):
    assert parse_strict('Answer: {"decision": "MEETS"}') is None
    assert parse_strict('{"decision": "MEETS"} Hope this helps.') is None


def test_returns_none_for_trailing_comma_and_duplicate_keys(parse_strict):
    assert parse_strict('{"decision": "MEETS",}') is None
    assert parse_strict('{"decision": "MEETS", "decision": "DOES_NOT_MEET"}') is None


def test_returns_none_for_unclosed_or_non_json_fence(parse_strict):
    assert parse_strict('```json\n{"decision": "MEETS"}') is None
    assert parse_strict('```json\n{"decision": "MEETS"}\n``` and more text') is None


def test_registered_parser_is_unchanged_in_its_tolerances():
    """The strict parser is an alternative; the registered one still takes the fallback and the trailing-comma repair."""
    assert RT.parse_response("I cannot conclude MEETS.")["decision"] == "MEETS"
    assert RT.parse_response('{"decision": "MEETS",}')["decision"] == "MEETS"
