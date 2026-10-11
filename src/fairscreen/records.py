"""Canonical applicant records and their canonical facts.

The record holds only qualification content. Identity signals are attached at
render time, so an identity edit cannot change a qualification field.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .policy import AnyOf, AssessmentPassed, CredentialValid, DegreeIn, LanguageAtLeast, MinYears, Policy
from .world import SCREENING_DATE

NOW = SCREENING_DATE[0] * 12 + (SCREENING_DATE[1] - 1)  # month index of the screening date


def month_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


@dataclass(frozen=True)
class Role:
    title: str
    employer: str
    start: int            # month index (inclusive)
    end: int | None       # month index (inclusive); None means "present" (up to the screening month)
    line: str             # description sentence
    skills: tuple = ()    # skills the description explicitly says were used
    dates_stated: bool = True
    part_time: bool = False

    def months(self) -> int:
        end = NOW if self.end is None else self.end
        return end - self.start  # whole months elapsed; a role from Mar 2021 to present (Oct 2026) = 67


@dataclass(frozen=True)
class Record:
    roles: tuple                       # most recent first
    degree_level: str = "BACHELOR"     # NONE / NITEC / DIPLOMA / BACHELOR / MASTER / PHD
    degree_field: str = "Business Administration"
    degree_year: int = 2015
    institution: str = "National University of Singapore"
    education_stated: bool = True
    credentials: tuple = ()            # (name, status, month_index) status VALID / EXPIRED / REVOKED
    credentials_stated: bool = True
    assessments: tuple = ()            # (name, outcome, score, month_index) PASSED / FAILED / NOT_TAKEN / PENDING
    languages: tuple = (("English", "NATIVE"),)
    languages_stated: bool = True
    conflicts: tuple = ()              # fact keys under an explicit unresolved conflict
    notes: tuple = ()                  # recruiter notes rendered verbatim (used for conflicts)
    total_claim: str = ""              # optional summary claim about total experience (realistic suite)
    extra_lines: tuple = ()            # qualification-irrelevant lines rendered in Additional information


def skill_months(record: Record, skill: str):
    """Return (known_months, has_undated) for experience in `skill`, merging overlaps."""
    intervals = []
    undated = False
    for r in record.roles:
        if skill not in r.skills:
            continue
        if not r.dates_stated:
            undated = True
            continue
        end = NOW if r.end is None else r.end
        intervals.append((r.start, end))
    intervals.sort()
    total = 0
    cur_s = cur_e = None
    for s, e in intervals:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None:
        total += cur_e - cur_s
    return total, undated


def _skills_in_policy(policy: Policy):
    out = []

    def visit(req):
        if isinstance(req, MinYears):
            out.append(req.skill)
        elif isinstance(req, AnyOf):
            for o in req.options:
                visit(o)
    for _, r in policy.requirements:
        visit(r)
    return out


def _names_in_policy(policy: Policy, cls):
    out = []

    def visit(req):
        if isinstance(req, cls):
            out.append(req)
        elif isinstance(req, AnyOf):
            for o in req.options:
                visit(o)
    for _, r in policy.requirements:
        visit(r)
    return out


def canonical_facts(record: Record, policy: Policy) -> dict:
    """The facts a perfect reader would extract, with None/UNKNOWN where the policy says 'not established'."""
    facts = {}
    for skill in _skills_in_policy(policy):
        key = f"years:{skill}"
        months, undated = skill_months(record, skill)
        if key in record.conflicts or undated:
            facts[key] = None
        else:
            facts[key] = round(months / 12.0, 2)
    for req in _names_in_policy(policy, CredentialValid):
        key = f"credential:{req.name}"
        if key in record.conflicts or not record.credentials_stated:
            facts[key] = "UNKNOWN"
            continue
        match = [c for c in record.credentials if c[0] == req.name]
        facts[key] = match[0][1] if match else "NOT_HELD"
    for req in _names_in_policy(policy, AssessmentPassed):
        key = f"assessment:{req.name}"
        match = [a for a in record.assessments if a[0] == req.name]
        if key in record.conflicts or not match or match[0][1] == "PENDING":
            facts[key] = "UNKNOWN"
        else:
            facts[key] = match[0][1]
    degs = _names_in_policy(policy, DegreeIn)
    if degs:
        allowed = degs[0].fields
        if not record.education_stated:
            facts["degree_level"], facts["degree_field"] = "UNKNOWN", "UNKNOWN"
        else:
            facts["degree_level"] = record.degree_level
            if record.degree_level == "NONE":
                facts["degree_field"] = "NONE"
            elif "degree_field" in record.conflicts:
                facts["degree_field"] = "UNKNOWN"
            else:
                facts["degree_field"] = record.degree_field if record.degree_field in allowed else "OTHER"
    for req in _names_in_policy(policy, LanguageAtLeast):
        key = f"language:{req.language}"
        if key in record.conflicts or not record.languages_stated:
            facts[key] = "UNKNOWN"
            continue
        match = [lv for lang, lv in record.languages if lang == req.language]
        facts[key] = match[0] if match else "NONE"
    return facts
