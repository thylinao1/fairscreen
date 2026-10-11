"""Benchmark validity tests: oracle agreement, edit invariants, suite contracts, leakage, parser, baselines."""
import itertools
import json
import random
import re
from collections import defaultdict

import pytest

from fairscreen import world
from fairscreen.policy import DEGREE_LEVELS, LANG_LEVELS, decide, policy_to_spec
from fairscreen.runtime import (DOES_NOT_MEET, INSUFFICIENT, MEETS, family_pass, parse_facts, parse_response, replay)

SPLITS = ("dev", "final")


def load(split):
    vs = [json.loads(l) for l in open(f"data/{split}/variants.jsonl")]
    fams = [json.loads(l) for l in open(f"data/{split}/families.jsonl")]
    return vs, fams


@pytest.fixture(scope="module", params=SPLITS)
def data(request):
    return load(request.param)


def test_oracles_agree_on_every_generated_variant(data):
    vs, _ = data
    assert len(vs) > 100
    for v in vs:
        assert replay(v["policy_spec"], v["gold_facts"]) == v["gold_decision"], v["variant_id"]


ALL_POLICIES = [world.make_policy(o, language=l) for o in world.OCCUPATIONS for l in ("Malay", "Mandarin", "Tamil")] + \
    [world.make_policy("customer_support", language_note=True)]


def random_value(key, rng):
    if key.startswith("years:"):
        return rng.choice([None, round(rng.uniform(0, 8), 2), 1.0, 2.0, 3.0, 4.0, 5.0])
    if key.startswith("credential:"):
        return rng.choice(["VALID", "EXPIRED", "REVOKED", "NOT_HELD", "UNKNOWN"])
    if key.startswith("assessment:"):
        return rng.choice(["PASSED", "FAILED", "NOT_TAKEN", "UNKNOWN"])
    if key == "degree_level":
        return rng.choice(list(DEGREE_LEVELS) + ["UNKNOWN"])
    if key == "degree_field":
        return rng.choice(list(world.DATA_FIELDS + world.LAB_FIELDS) + ["OTHER", "NONE", "UNKNOWN"])
    if key.startswith("language:"):
        return rng.choice(list(LANG_LEVELS) + ["UNKNOWN"])
    raise ValueError(key)


@pytest.mark.parametrize("policy", ALL_POLICIES, ids=lambda p: p.policy_id)
def test_oracles_agree_on_random_facts(policy):
    rng = random.Random(7)
    spec = policy_to_spec(policy)
    for _ in range(3000):
        facts = {k: random_value(k, rng) for k in policy.fact_keys()}
        assert decide(policy, facts) == replay(spec, facts), facts


def test_decision_rule_ordering_false_beats_unknown():
    p = world.make_policy("backend")
    assert decide(p, {"years:Python": None, "assessment:SQL assessment": "FAILED"}) == DOES_NOT_MEET
    assert decide(p, {"years:Python": 5.0, "assessment:SQL assessment": "UNKNOWN"}) == INSUFFICIENT
    assert decide(p, {"years:Python": 5.0, "assessment:SQL assessment": "PASSED"}) == MEETS
    assert decide(p, {"years:Python": 2.9, "assessment:SQL assessment": "PASSED"}) == DOES_NOT_MEET


def by_family(vs):
    out = defaultdict(list)
    for v in vs:
        out[v["family_id"]].append(v)
    return out


def test_core_contract(data):
    vs, _ = data
    for fid, fam in by_family([v for v in vs if v["suite"] == "core"]).items():
        hi = [v for v in fam if v["tags"]["q"] == "hi"]
        lo = [v for v in fam if v["tags"]["q"] == "lo"]
        assert {v["gold_decision"] for v in hi} == {MEETS}, fid
        assert {v["gold_decision"] for v in lo} == {DOES_NOT_MEET}, fid
        # every identity, context and control variant shares the canonical facts of its qualification state
        assert len({json.dumps(v["gold_facts"], sort_keys=True) for v in hi}) == 1, fid
        assert len({json.dumps(v["gold_facts"], sort_keys=True) for v in lo}) == 1, fid
        core8 = [v for v in fam if "control" not in v["tags"]]
        assert len(core8) == 8, fid


def _strip_identity(v):
    text = v["application_text"]
    text = text.replace(v["identity_label"], "<ID>")
    text = re.sub(r"Applicant ID A-\d+", "<ID>", text)
    for m in world.MARKERS.values():
        text = text.replace(m, "<MARKER>")
    return text


def test_identity_edit_changes_only_identity(data):
    vs, _ = data
    for fid, fam in by_family([v for v in vs if v["suite"] in ("core", "realistic", "audit")]).items():
        groups = defaultdict(list)
        for v in fam:
            if v["tags"].get("control") == "length":
                continue
            key = (v["tags"].get("q"), v["context"], v["tags"].get("frame"))
            groups[key].append(v)
        for key, g in groups.items():
            # marker variants use a neutral ID; name variants use names; both reduce to the same skeleton
            names = [x for x in g if x["identity_kind"] != "neutral" or True]
            skel = {_strip_identity(x).replace("<MARKER> (2021 to present)", "").replace("\n- \n", "\n")
                    for x in names if x["identity_kind"] == names[0]["identity_kind"]}
            assert len(skel) == 1, (fid, key)


def test_sufficiency_contract(data):
    vs, _ = data
    want = {"complete": MEETS, "missing": INSUFFICIENT, "conflict": INSUFFICIENT, "missing_other_fails": DOES_NOT_MEET}
    fams = by_family([v for v in vs if v["suite"] == "sufficiency"])
    assert fams
    for fid, fam in fams.items():
        assert {v["tags"]["ev"]: v["gold_decision"] for v in fam} == want, fid


def test_relevance_contract(data):
    vs, fams_meta = data
    meta = {f["family_id"]: f for f in fams_meta}
    for fid, fam in by_family([v for v in vs if v["suite"] == "relevance"]).items():
        got = {(v["tags"]["job"], v["tags"]["skill"]): v["gold_decision"] for v in fam}
        assert got[("relevant", "present")] == MEETS and got[("relevant", "absent")] == DOES_NOT_MEET, fid
        irr = meta[fid]["irrelevant_answer"]
        assert got[("irrelevant", "present")] == got[("irrelevant", "absent")] == irr, fid
        # both jobs see the identical application
        apps = {v["tags"]["skill"]: set() for v in fam}
        for v in fam:
            apps[v["tags"]["skill"]].add(v["application_text"].split("\n", 2)[2])
        assert all(len(a) == 1 for a in apps.values()), fid


def test_realistic_and_audit_contract(data):
    vs, _ = data
    for v in vs:
        if v["suite"] in ("realistic", "audit"):
            assert v["gold_decision"] == (MEETS if v["tags"]["q"] == "hi" else DOES_NOT_MEET), v["variant_id"]


def test_no_leakage_into_model_visible_text(data):
    vs, _ = data
    for v in vs:
        for field in ("application_text", "policy_text", "framing_text"):
            t = v[field] or ""
            assert "core-" not in t and "suff-" not in t and "gold" not in t.lower() and "variant" not in t.lower()
        assert not re.search(r"\b(MEETS|DOES_NOT_MEET|INSUFFICIENT_EVIDENCE)\b", v["application_text"])


def test_margins_keep_threshold_crossings_unambiguous(data):
    vs, _ = data
    for v in vs:
        if v["suite"] != "core":
            continue
        for k, val in v["gold_facts"].items():
            if k.startswith("years:") and val is not None:
                for _, req in v["policy_spec"]["requirements"] and [(r["id"], r["req"]) for r in v["policy_spec"]["requirements"]]:
                    leaves = req.get("options", [req])
                    for leaf in leaves:
                        if leaf["type"] == "min_years" and "years:" + leaf["skill"] == k:
                            assert abs(val - leaf["years"]) >= 0.4, (v["variant_id"], val)


# ---------------------------------------------------------------- parser
@pytest.mark.parametrize("text,status,dec", [
    ('{"decision": "MEETS"}', "json", MEETS),
    ('```json\n{"decision": "DOES_NOT_MEET"}\n```', "json", DOES_NOT_MEET),
    ('Reasoning... {"decision": "insufficient evidence",}', "json", INSUFFICIENT),
    ('{"decision": "Does not meet"}', "json", DOES_NOT_MEET),
    ("The decision is DOES_NOT_MEET.", "fallback", DOES_NOT_MEET),
    ("R1 is met and R2 is met, so MEETS. Not DOES_NOT_MEET.", "invalid", None),
    ("The applicant meets the requirements.", "invalid", None),
    ('{"decision": "ELIGIBLE"}', "invalid", None),
    ("", "empty", None),
    ('{"a": {"b": "}"}, "decision": "MEETS"}', "json", MEETS),
])
def test_parser(text, status, dec):
    r = parse_response(text)
    assert (r["status"], r["decision"]) == (status, dec)


def test_parse_facts_and_replay():
    p = world.make_policy("data_analyst")
    spec = policy_to_spec(p)
    keys = list(p.fact_keys())
    obj = {"facts": {"highest_qualification_level": "BACHELOR", "qualification_field": "Economics",
                     "data_analysis_experience_years": "1.5", "data_case_study_result": "passed"}, "decision": "MEETS"}
    facts, ev, invalid = parse_facts(obj, keys, spec)
    assert invalid == [] and facts["degree_field"] == "Economics" and facts["years:data analysis"] == 1.5
    assert replay(spec, facts) == MEETS
    facts2, _, invalid2 = parse_facts({"facts": {"qualification_field": "Basket weaving"}}, keys, spec)
    assert "degree_field" in invalid2 and replay(spec, facts2) == INSUFFICIENT


# ---------------------------------------------------------------- baselines
def test_constant_baselines_fail_core_and_oracle_passes(data):
    vs, fams = data
    core = [f for f in fams if f["suite"] == "core"]
    gold = {v["variant_id"]: v["gold_decision"] for v in vs}
    prim = lambda f: [i for i in f["variant_ids"] if "control=" not in i]
    for const in (MEETS, DOES_NOT_MEET, INSUFFICIENT):
        recs = [{"variant_id": i, "decision": const, "gold": g} for i, g in gold.items()]
        assert sum(family_pass(recs, prim(f)) for f in core) == 0
    recs = [{"variant_id": i, "decision": g, "gold": g} for i, g in gold.items()]
    assert all(family_pass(recs, prim(f)) for f in core)
