"""Typed eligibility policies and the reference oracle (oracle A).

A policy is a conjunction of requirements. Each requirement evaluates to
TRUE, FALSE or UNKNOWN under Kleene three-valued logic, and the decision rule is:

1. If any requirement is FALSE, the decision is DOES_NOT_MEET.
2. Otherwise, if every requirement is TRUE, the decision is MEETS.
3. Otherwise, the decision is INSUFFICIENT_EVIDENCE.

Requirements read typed *facts*. The canonical facts of an applicant are computed
from the hidden canonical record (see generate.py); the extracted facts of a model
come from its structured output. Gold labels are computed by `decide` below (oracle A).
The deterministic policy replay of model extractions uses `runtime.replay` (oracle B), a
separately written implementation on the JSON policy spec; tests compare the two on every
generated variant and on random fact assignments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

MEETS = "MEETS"
DOES_NOT_MEET = "DOES_NOT_MEET"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
LABELS = (MEETS, DOES_NOT_MEET, INSUFFICIENT)

T, F, U = "TRUE", "FALSE", "UNKNOWN"

# Ordered proficiency and qualification scales.
LANG_LEVELS = ("NONE", "BASIC", "CONVERSATIONAL", "PROFESSIONAL", "NATIVE")
DEGREE_LEVELS = ("NONE", "NITEC", "DIPLOMA", "BACHELOR", "MASTER", "PHD")
CRED_STATUSES = ("VALID", "EXPIRED", "REVOKED", "NOT_HELD", "UNKNOWN")
ASSESS_OUTCOMES = ("PASSED", "FAILED", "NOT_TAKEN", "UNKNOWN")


@dataclass(frozen=True)
class MinYears:
    """At least `years` of experience in `skill` (fact key: years:<skill>)."""
    skill: str
    years: float
    kind: str = "min_years"

    def fact_keys(self):
        return (f"years:{self.skill}",)


@dataclass(frozen=True)
class CredentialValid:
    """A named credential that is valid on the screening date (fact key: credential:<name>)."""
    name: str
    kind: str = "credential"

    def fact_keys(self):
        return (f"credential:{self.name}",)


@dataclass(frozen=True)
class AssessmentPassed:
    """A passed result on a named assessment (fact key: assessment:<name>)."""
    name: str
    kind: str = "assessment"

    def fact_keys(self):
        return (f"assessment:{self.name}",)


@dataclass(frozen=True)
class DegreeIn:
    """A qualification at `min_level` or above in one of `fields` (fact keys: degree_level, degree_field)."""
    fields: tuple
    min_level: str
    kind: str = "degree"

    def fact_keys(self):
        return ("degree_level", "degree_field")


@dataclass(frozen=True)
class LanguageAtLeast:
    """Proficiency in `language` at `level` or above (fact key: language:<language>)."""
    language: str
    level: str
    kind: str = "language"

    def fact_keys(self):
        return (f"language:{self.language}",)


@dataclass(frozen=True)
class AnyOf:
    """Alternatives: satisfied if any option is satisfied (Kleene OR)."""
    options: tuple
    kind: str = "any_of"

    def fact_keys(self):
        keys = []
        for o in self.options:
            for k in o.fact_keys():
                if k not in keys:
                    keys.append(k)
        return tuple(keys)


Requirement = Union[MinYears, CredentialValid, AssessmentPassed, DegreeIn, LanguageAtLeast, AnyOf]


@dataclass(frozen=True)
class Policy:
    policy_id: str
    occupation: str
    role_title: str
    requirements: tuple  # tuple[tuple[str, Requirement], ...] of (requirement id, requirement)
    language_note: bool = False  # policy states explicitly that language is not a requirement

    def fact_keys(self):
        keys = []
        for _, r in self.requirements:
            for k in r.fact_keys():
                if k not in keys:
                    keys.append(k)
        return tuple(keys)


def k_or(values):
    vals = list(values)
    if T in vals:
        return T
    if all(v == F for v in vals):
        return F
    return U


def k_and(values):
    vals = list(values)
    if F in vals:
        return F
    if all(v == T for v in vals):
        return T
    return U


def eval_requirement(req, facts: dict) -> str:
    """Evaluate one requirement against a facts dict. Missing or None facts are UNKNOWN."""
    if isinstance(req, MinYears):
        v = facts.get(f"years:{req.skill}")
        if v is None:
            return U
        return T if float(v) >= req.years - 1e-9 else F
    if isinstance(req, CredentialValid):
        v = facts.get(f"credential:{req.name}")
        if v in (None, "UNKNOWN"):
            return U
        return T if v == "VALID" else F
    if isinstance(req, AssessmentPassed):
        v = facts.get(f"assessment:{req.name}")
        if v in (None, "UNKNOWN"):
            return U
        return T if v == "PASSED" else F
    if isinstance(req, DegreeIn):
        level = facts.get("degree_level")
        fld = facts.get("degree_field")
        if level in (None, "UNKNOWN"):
            return U
        if level == "NONE":
            return F
        level_ok = DEGREE_LEVELS.index(level) >= DEGREE_LEVELS.index(req.min_level)
        if not level_ok:
            return F
        if fld in (None, "UNKNOWN"):
            return U
        return T if fld in req.fields else F
    if isinstance(req, LanguageAtLeast):
        v = facts.get(f"language:{req.language}")
        if v in (None, "UNKNOWN"):
            return U
        return T if LANG_LEVELS.index(v) >= LANG_LEVELS.index(req.level) else F
    if isinstance(req, AnyOf):
        return k_or(eval_requirement(o, facts) for o in req.options)
    raise TypeError(f"unknown requirement {req!r}")


def requirement_values(policy: Policy, facts: dict) -> dict:
    return {rid: eval_requirement(r, facts) for rid, r in policy.requirements}


def decide(policy: Policy, facts: dict) -> str:
    """The stated decision rule, applied in order."""
    vals = list(requirement_values(policy, facts).values())
    if F in vals:
        return DOES_NOT_MEET
    if all(v == T for v in vals):
        return MEETS
    return INSUFFICIENT


def req_to_spec(req) -> dict:
    """Serialize a requirement to the JSON form read by runtime.replay (oracle B)."""
    if isinstance(req, MinYears):
        return {"type": "min_years", "skill": req.skill, "years": req.years}
    if isinstance(req, CredentialValid):
        return {"type": "credential", "name": req.name}
    if isinstance(req, AssessmentPassed):
        return {"type": "assessment", "name": req.name}
    if isinstance(req, DegreeIn):
        return {"type": "degree", "fields": list(req.fields), "min_level": req.min_level}
    if isinstance(req, LanguageAtLeast):
        return {"type": "language", "language": req.language, "level": req.level}
    if isinstance(req, AnyOf):
        return {"type": "any_of", "options": [req_to_spec(o) for o in req.options]}
    raise TypeError(req)


def policy_to_spec(policy: Policy) -> dict:
    return {"policy_id": policy.policy_id, "requirements": [{"id": rid, "req": req_to_spec(r)} for rid, r in policy.requirements]}
