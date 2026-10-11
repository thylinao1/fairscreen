"""Convert cluster vLLM outputs to response records with the same runtime parser.

python analysis/cluster_records.py  (reads results/cluster_raw/*.jsonl.gz, writes results/cluster_records/<model>/<recipe>.jsonl)
"""
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fairscreen import runtime as RT  # noqa: E402


def main():
    vs = {json.loads(l)["variant_id"]: json.loads(l) for l in open("data/final/variants.jsonl")}
    for path in sorted(glob.glob("results/cluster_raw/*.jsonl.gz")):
        model = os.path.basename(path)[:-9]
        out = {}
        n = 0
        for line in gzip.open(path, "rt"):
            o = json.loads(line)
            recipe, vid, rep = o["id"].split("|")
            v = vs[vid]
            text = o.get("text")
            parsed = RT.parse_response(text) if text is not None else {"status": "transport_error", "decision": None, "obj": None}
            rec = {"variant_id": vid, "family_id": v["family_id"], "suite": v["suite"], "recipe": recipe,
                   "rep": int(rep[1:]), "status": parsed["status"], "decision": parsed["decision"],
                   "gold": v["gold_decision"], "raw": text, "finish_reason": o.get("finish_reason"),
                   "usage": {"in": o.get("prompt_tokens"), "out": o.get("completion_tokens")}}
            if recipe in ("structured", "extract_only"):
                facts, evidence, invalid = RT.parse_facts(parsed.get("obj"), v["fact_keys"], v["policy_spec"])
                rec["facts"], rec["evidence"], rec["invalid_facts"] = facts, evidence, invalid
                rec["replay"] = RT.replay(v["policy_spec"], facts) if parsed.get("obj") is not None else None
                if recipe == "extract_only":
                    rec["model_decision"], rec["decision"] = rec["decision"], rec["replay"]
            out.setdefault(recipe, []).append(rec)
            n += 1
        for recipe, recs in out.items():
            d = f"results/cluster_records/{model}"
            os.makedirs(d, exist_ok=True)
            with open(f"{d}/{recipe}.jsonl", "w") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
        print(model, n, {k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
