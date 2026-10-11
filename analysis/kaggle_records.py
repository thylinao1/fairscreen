"""Rebuild response records from downloaded Kaggle run files.

Each conversation in a run file is named `recipe|variant_id|rN|aK-<hash>`. The assistant text of the last
successful attempt is re-parsed with the same runtime code, so these records are recomputed from raw outputs and
never read the task's own summary.

python analysis/kaggle_records.py results/kaggle_runs  -> results/records/<task>/<model>.jsonl
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fairscreen import runtime as RT  # noqa: E402

TAG = re.compile(r"^(?P<recipe>direct|fairness|structured|extract_only)\|(?P<vid>.+)\|r(?P<rep>\d+)\|a(?P<att>\d+)-[0-9a-f]{8}$")


def load_variants(split="final"):
    return {json.loads(l)["variant_id"]: json.loads(l) for l in open(f"data/{split}/variants.jsonl")}


def records_from_run(run, variants):
    by_call = {}
    for conv in run.get("conversations", []):
        m = TAG.match(conv.get("id", ""))
        if not m:
            continue
        key = (m["recipe"], m["vid"], int(m["rep"]))
        texts = []
        metrics = {}
        for req in conv.get("requests", []):
            for c in req.get("contents", []):
                if c.get("role") == "CONTENT_ROLE_ASSISTANT":
                    texts.append("".join(p.get("text", "") for p in c.get("parts", [])))
            metrics = req.get("metrics", metrics)
        by_call.setdefault(key, []).append((int(m["att"]), texts[-1] if texts else None, metrics))
    out = []
    for (recipe, vid, rep), attempts in by_call.items():
        attempts.sort()
        ok = [a for a in attempts if a[1] is not None]
        text, metrics = (ok[-1][1], ok[-1][2]) if ok else (None, {})
        v = variants[vid]
        parsed = RT.parse_response(text) if ok else {"status": "transport_error", "decision": None, "obj": None}
        rec = {"variant_id": vid, "family_id": v["family_id"], "suite": v["suite"], "recipe": recipe, "rep": rep,
               "status": parsed["status"], "decision": parsed["decision"], "gold": v["gold_decision"], "raw": text,
               "n_attempts": len(attempts),
               "usage": {"in": metrics.get("inputTokens"), "out": metrics.get("outputTokens"),
                         "cost_nd": int(metrics.get("inputTokensCostNanodollars", 0) or 0)
                         + int(metrics.get("outputTokensCostNanodollars", 0) or 0),
                         "ms": int(metrics.get("totalBackendLatencyMs", 0) or 0)}}
        if recipe in ("structured", "extract_only"):
            facts, evidence, invalid = RT.parse_facts(parsed.get("obj"), v["fact_keys"], v["policy_spec"])
            rec["facts"], rec["evidence"], rec["invalid_facts"] = facts, evidence, invalid
            rec["replay"] = RT.replay(v["policy_spec"], facts) if parsed.get("obj") is not None else None
            if recipe == "extract_only":
                rec["model_decision"], rec["decision"] = rec["decision"], rec["replay"]
        out.append(rec)
    return out


def main(root):
    variants = load_variants()
    for path in glob.glob(os.path.join(root, "**", "*.run.json"), recursive=True):
        run = json.load(open(path))
        task = run["taskVersion"]["name"]
        model = run["modelVersion"]["slug"].replace("/", "_")
        recs = records_from_run(run, variants)
        dest = f"results/records/{task}/{model}.jsonl"
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        res = run.get("results", [{}])[0].get("numericResult", {})
        print(f"{task:40s} {model:45s} records={len(recs):5d} hosted_score={res.get('value')} radius={res.get('confidenceInterval')} state={run.get('state')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/kaggle_runs")
