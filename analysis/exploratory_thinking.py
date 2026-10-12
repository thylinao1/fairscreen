"""Exploratory (not registered): Qwen3-32B with thinking on versus the registered thinking-off run.

Same model, same core-suite prompts (direct and structured recipes, controls and repeats included); the thinking run
uses Qwen's documented thinking-mode sampling (temperature 0.6, top_p 0.95, top_k 20, seed 0), the registered run
used temperature 0 with thinking off. Raw output: results/exploratory/qwen3-32b-thinking.jsonl.gz.

python analysis/exploratory_thinking.py
"""
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from fairscreen import runtime as RT  # noqa: E402
from analyze import View, joint_contrast, load_data, metrics  # noqa: E402

RAW = "results/exploratory/qwen3-32b-thinking.jsonl.gz"
OUT = "results/exploratory/qwen3-32b-thinking"


def to_records(vs):
    out = {}
    for line in gzip.open(RAW, "rt"):
        o = json.loads(line)
        recipe, vid, rep = o["id"].split("|")
        v = vs[vid]
        text = o.get("text")
        parsed = RT.parse_response(text) if text is not None else {"status": "transport_error", "decision": None, "obj": None}
        rec = {"variant_id": vid, "family_id": v["family_id"], "suite": v["suite"], "recipe": recipe, "rep": int(rep[1:]),
               "status": parsed["status"], "decision": parsed["decision"], "gold": v["gold_decision"], "raw": text,
               "finish_reason": o.get("finish_reason"), "usage": {"in": o.get("prompt_tokens"), "out": o.get("completion_tokens")}}
        if recipe == "structured":
            facts, evidence, invalid = RT.parse_facts(parsed.get("obj"), v["fact_keys"], v["policy_spec"])
            rec["facts"], rec["evidence"], rec["invalid_facts"] = facts, evidence, invalid
            rec["replay"] = RT.replay(v["policy_spec"], facts) if parsed.get("obj") is not None else None
        out.setdefault(recipe, []).append(rec)
    os.makedirs(OUT, exist_ok=True)
    for recipe, recs in out.items():
        with open(f"{OUT}/{recipe}.jsonl", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
    return out


def main():
    vs, fams = load_data("final")
    core = [f for f in fams if f["suite"] == "core"]
    think = to_records(vs)
    base = {r: [json.loads(l) for l in open(f"results/cluster_records/qwen3-32b/{r}.jsonl") if '"suite": "core"' in l]
            for r in ("direct", "structured")}
    res = {"note": "Exploratory: not registered. Thinking run uses Qwen thinking-mode sampling; registered run used temperature 0."}
    for label, recipe, key in (("direct", "direct", "decision"), ("structured", "structured", "decision"),
                               ("code_same_extraction", "structured", "replay")):
        mt = metrics(View(think[recipe], key=key), vs, core)
        mb = metrics(View(base[recipe], key=key), vs, core)
        c = joint_contrast({"q": mt["_fam_pass"]}, {"q": mb["_fam_pass"]}, core)
        res[label] = {
            "thinking_on": mt["robust"], "thinking_off": mb["robust"],
            "difference": {k: c[k] for k in ("estimate", "lo", "hi", "p")},
            "false_accept": (mt["core_false_accept"], mb["core_false_accept"]),
            "false_reject": (mt["core_false_reject"], mb["core_false_reject"]),
            "abstain": (mt["core_abstention"], mb["core_abstention"]),
            "invalid": (mt["invalid"], mb["invalid"]),
            "by_type_on": mt["robust_by_decisive_type"], "by_type_off": mb["robust_by_decisive_type"],
            "identity_flip": (mt["identity_flip"], mb["identity_flip"]),
            "same_group_flip": (mt["same_group_flip"], mb["same_group_flip"]),
            "repeat_flip": (mt["repeat_flip"], mb["repeat_flip"]),
        }
    fin = [r.get("finish_reason") for rs in think.values() for r in rs]
    toks = sorted(r["usage"]["out"] or 0 for rs in think.values() for r in rs)
    res["finish_reasons"] = {k: fin.count(k) for k in set(fin)}
    res["completion_tokens_median"] = toks[len(toks) // 2]
    json.dump(res, open("results/exploratory/qwen3-32b-thinking_summary.json", "w"), indent=1, default=str)
    for k in ("direct", "structured", "code_same_extraction"):
        r = res[k]
        print(f"{k:22s} on {100*r['thinking_on'][0]:.1f} off {100*r['thinking_off'][0]:.1f} diff {100*r['difference']['estimate']:+.1f} "
              f"[{100*r['difference']['lo']:+.1f}, {100*r['difference']['hi']:+.1f}] FA {r['false_accept'][0]:.2f}/{r['false_accept'][1]:.2f} "
              f"invalid {r['invalid'][0]:.3f}")
    print("finish", res["finish_reasons"], "median completion tokens", res["completion_tokens_median"])


if __name__ == "__main__":
    main()
