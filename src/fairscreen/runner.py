"""Execution and summary logic shared by local runs and the hosted Kaggle tasks (stdlib only).

Inlined into each Kaggle task file after runtime.py.
"""
import json
import random
import time

RECIPES = ("direct", "fairness", "structured", "extract_only")
MAX_ATTEMPTS = 3  # infrastructure errors only; answer-independent


def plan_calls(variants, families, suites=None, seed=20261012):
    """Every variant once, plus the prespecified exact repeats; interleaved in a fixed random order."""
    by_fam = {f["family_id"]: f for f in families}
    calls = []
    for v in variants:
        if suites and v["suite"] not in suites:
            continue
        calls.append((v, 0))
        fam = by_fam[v["family_id"]]
        if fam.get("repeats") and "control" not in v["tags"]:
            for r in range(1, fam["repeats"] + 1):
                calls.append((v, r))
    random.Random(seed).shuffle(calls)
    return calls


def evaluate_variant(v, recipe, call, rep=0):
    """call(prompt, tag) -> (text, error, usage dict). Returns one response record."""
    prompt = build_prompt(v, recipe)  # noqa: F821 (defined in runtime, which precedes this module when inlined)
    text, err, usage, attempts = None, None, {}, []
    for a in range(MAX_ATTEMPTS):
        text, err, usage = call(prompt, f"{recipe}|{v['variant_id']}|r{rep}|a{a}")
        attempts.append(err)
        if err is None:
            break
        time.sleep(3 * (a + 1))
    if err is not None:
        parsed = {"status": "transport_error", "decision": None, "obj": None}
    else:
        parsed = parse_response(text)  # noqa: F821
    rec = {"variant_id": v["variant_id"], "family_id": v["family_id"], "suite": v["suite"], "recipe": recipe,
           "rep": rep, "status": parsed["status"], "decision": parsed["decision"], "gold": v["gold_decision"],
           "raw": text, "errors": [e for e in attempts if e], "n_attempts": len(attempts), "usage": usage}
    if recipe in ("structured", "extract_only"):
        facts, evidence, invalid = parse_facts(parsed.get("obj"), v["fact_keys"], v["policy_spec"])  # noqa: F821
        rec["facts"], rec["evidence"], rec["invalid_facts"] = facts, evidence, invalid
        rec["replay"] = replay(v["policy_spec"], facts) if parsed.get("obj") is not None else None  # noqa: F821
        if recipe == "extract_only":
            rec["model_decision"] = rec["decision"]
            rec["decision"] = rec["replay"]
            if parsed.get("obj") is not None:
                rec["status"] = "json"  # no decision is requested in this recipe; a parsed facts object is valid
    return rec


def core_family_ids(families):
    return [f for f in families if f["suite"] == "core"]


def primary_ids(fam):
    return [i for i in fam["variant_ids"] if "control=" not in i]


def summarize(records, families, decision_key="decision"):
    """Primary score (robust policy fidelity on core families) with a stratified bootstrap radius."""
    rec0 = {r["variant_id"]: r for r in records if r["rep"] == 0}
    view = [{"variant_id": k, "decision": r.get(decision_key), "gold": r["gold"]} for k, r in rec0.items()]
    out = {}
    core = [f for f in families if f["suite"] == "core"]
    passes, strata = [], []
    for f in core:
        ids = primary_ids(f)
        passes.append(1.0 if family_pass(view, ids) else 0.0)  # noqa: F821
        strata.append(f["occupation"])
    m, lo, hi = bootstrap_mean(passes, strata)  # noqa: F821
    out["robust_policy_fidelity"] = {"mean": m, "lo": lo, "hi": hi, "n_families": len(passes)}
    for suite in ("sufficiency", "relevance", "realistic", "audit"):
        fs = [f for f in families if f["suite"] == suite]
        if fs:
            vals = [1.0 if family_pass(view, f["variant_ids"]) else 0.0 for f in fs]  # noqa: F821
            out[f"{suite}_family_pass"] = sum(vals) / len(vals)
    graded = [x for x in view if x["variant_id"] in rec0]
    out["decision_accuracy"] = sum(x["decision"] == x["gold"] for x in graded) / max(1, len(graded))
    out["invalid_rate"] = sum(r["status"] in ("invalid", "empty", "transport_error") for r in rec0.values()) / max(1, len(rec0))
    out["n_records"] = len(records)
    return out
