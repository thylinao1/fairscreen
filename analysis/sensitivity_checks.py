"""Sensitivity and completeness checks on the saved outputs (no model calls).

1. C3 sensitivity: the open-weight panel's C3 recomputed with Qwen3-32B's thinking-on run in place of the panel run
   (the protocol registered default reasoning, which is thinking on for Qwen3; the panel run had it off).
2. Generation-limit check promised in the protocol deviations: responses stopped by the token limit, per model and recipe.
3. Method-selection test: the frozen choice versus the other frozen candidates on the
   final changed layouts (descriptive regret).
4. Signed group-by-qualification cells: how many intervals would exclude zero by chance at 95 percent.

python analysis/sensitivity_checks.py   writes results/sensitivity_checks.json
"""
import csv
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from analyze import View, joint_contrast, load_data, metrics  # noqa: E402

CLUSTER = "results/cluster_records"
THINK = "results/exploratory/qwen3-32b-thinking"


def load(path):
    return [json.loads(line) for line in open(path)]


def c3_sensitivity(vs, core):
    per = {}
    for model in sorted(os.listdir(CLUSTER)):
        path = f"{THINK}/structured.jsonl" if model == "qwen3-32b" else f"{CLUSTER}/{model}/structured.jsonl"
        recs = load(path)
        per[model] = (metrics(View(recs, key="replay"), vs, core)["_fam_pass"],
                      metrics(View(recs), vs, core)["_fam_pass"])
    c = joint_contrast({m: v[0] for m, v in per.items()}, {m: v[1] for m, v in per.items()}, core)
    return {k: c[k] for k in ("estimate", "lo", "hi", "p", "per_model")}


def length_stops():
    out = {}
    for path in sorted(glob.glob(f"{CLUSTER}/*/*.jsonl")):
        model, recipe = path.split("/")[-2], os.path.basename(path)[:-6]
        fin = Counter(r.get("finish_reason") for r in load(path))
        out[f"{model}/{recipe}"] = {"length": fin.get("length", 0), "n": sum(fin.values())}
    return out


def method_selection():
    frozen = json.load(open("configs/method_selection_frozen.json"))
    rows = {(r["model"], r["config"]): r for r in csv.DictReader(open("results/final_kaggle/main_table.csv"))}
    out = {}
    for model, f in frozen.items():
        cands = list(f["dev_clean_quartet"])
        changed = {c: float(rows[(model, c)]["changed"]) for c in cands}
        code = float(rows[(model, "policy_code_same_extraction")]["changed"])
        out[model] = {"selected": f["selected"], "changed_quartet": changed,
                      "regret_among_frozen": max(changed.values()) - changed[f["selected"]],
                      "policy_code_changed_quartet_not_a_candidate": code}
    return out


def signed_chance():
    cells = [x for p in ("final_cluster", "final_kaggle") for cfg in ("direct", "fairness", "structured")
             for x in json.load(open(f"results/{p}/contrasts.json"))[f"S_signed_by_group_{cfg}"].values()]
    ex = [x for x in cells if x["lo"] > 0 or x["hi"] < 0]
    largest = min(ex, key=lambda x: x["estimate"])
    return {"n_cells": len(cells), "n_exclude_zero": len(ex), "expected_by_chance": 0.05 * len(cells),
            "largest_negative": largest}


def c3_kaggle_strict_json(vs, core):
    """Kaggle C3 when the model's own structured decision also requires valid JSON (same usable-output rule as the
    code arm, which needs the parsed facts). Not registered: the registered parser accepts a fallback decision."""
    code, strict = {}, {}
    for path in sorted(glob.glob("results/records/fairscreen-structured/*.jsonl")):
        recs = load(path)
        model = os.path.basename(path)[:-6]
        code[model] = metrics(View(recs, key="replay"), vs, core)["_fam_pass"]
        strict[model] = metrics(View([dict(r, decision=r["decision"] if r["status"] == "json" else None) for r in recs]), vs, core)["_fam_pass"]
    c = joint_contrast(code, strict, core)
    return {k: c[k] for k in ("estimate", "lo", "hi", "p", "per_model")}


def main():
    vs, fams = load_data("final")
    core = [f for f in fams if f["suite"] == "core"]
    res = {"c3_with_qwen_thinking_on": c3_sensitivity(vs, core), "length_stops": length_stops(),
           "method_selection": method_selection(), "signed_cells": signed_chance(),
           "c3_kaggle_strict_json": c3_kaggle_strict_json(vs, [f for f in fams if f["suite"] == "core"])}
    json.dump(res, open("results/sensitivity_checks.json", "w"), indent=1)
    c = res["c3_with_qwen_thinking_on"]
    print(f"C3 with Qwen thinking on: {100*c['estimate']:+.1f} [{100*c['lo']:+.1f}, {100*c['hi']:+.1f}] p={c['p']:.3f}")
    print("per model", {m: round(100 * v, 1) for m, v in c["per_model"].items()})
    print("length stops (nonzero)", {k: v for k, v in res["length_stops"].items() if v["length"]})
    print("method selection", json.dumps(res["method_selection"], indent=0))
    print("signed", {k: v for k, v in res["signed_cells"].items() if k != "largest_negative"})


if __name__ == "__main__":
    main()
