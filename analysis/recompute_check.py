"""Independent recomputation of the primary score (robust policy fidelity) from raw response text.

Written separately from analyze.py and runtime.family_pass: it re-parses the raw text with its own minimal
rule (last JSON object with a decision key), groups by family itself, and compares with analyze.py's table.
python analysis/recompute_check.py <records dir, --local layout> <split> <analyze main_table.csv>
"""
import csv
import glob
import json
import os
import re
import sys

LABELS = {"MEETS", "DOES_NOT_MEET", "INSUFFICIENT_EVIDENCE"}


def decision_from_raw(text, extracted_only=None):
    if extracted_only is not None:
        return extracted_only
    if not text:
        return None
    best = None
    depth, start, in_s, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_s:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_s = False
            continue
        if ch == '"' and depth:
            in_s = True
        elif ch == "{":
            start = i if depth == 0 else start
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                frag = text[start:i + 1]
                for cand in (frag, re.sub(r",\s*([}\]])", r"\1", frag)):
                    try:
                        o = json.loads(cand)
                    except Exception:
                        continue
                    for k, v in o.items():
                        if k.lower() == "decision" and isinstance(v, str):
                            lab = re.sub(r"[\s\-]+", "_", v.strip().upper())
                            if lab in LABELS:
                                best = lab
                    break
    if best:
        return best
    found = set(re.findall(r"\b(DOES_NOT_MEET|INSUFFICIENT_EVIDENCE|MEETS)\b", text))
    return found.pop() if len(found) == 1 else None


def main(root, split, table):
    gold = {}
    fam_of = {}
    for l in open(f"data/{split}/variants.jsonl"):
        v = json.loads(l)
        gold[v["variant_id"]] = v["gold_decision"]
        fam_of[v["variant_id"]] = v["family_id"]
    core_fams = {}
    for l in open(f"data/{split}/families.jsonl"):
        f = json.loads(l)
        if f["suite"] == "core":
            core_fams[f["family_id"]] = [i for i in f["variant_ids"] if "control=" not in i]
    expected = {(r["model"], r["config"]): float(r["robust"]) for r in csv.DictReader(open(table))}
    bad = 0
    for path in sorted(glob.glob(os.path.join(root, "*", "*.jsonl"))):
        model = os.path.basename(os.path.dirname(path))
        recipe = os.path.basename(path)[:-6]
        dec = {}
        for l in open(path):
            r = json.loads(l)
            if r["rep"] != 0:
                continue
            if recipe == "extract_only":
                d = r.get("replay")
            else:
                d = decision_from_raw(r.get("raw"))
            if r["variant_id"] not in dec or d is not None:
                dec[r["variant_id"]] = d
        ran = {fam_of[v] for v in dec}
        fams = [f for f in core_fams if f in ran]
        score = sum(all(dec.get(v) == gold[v] for v in core_fams[f]) for f in fams) / max(1, len(fams))
        exp = expected.get((model, recipe))
        ok = exp is not None and abs(exp - score) < 1e-12
        bad += not ok
        print(f"{model:45s} {recipe:12s} independent={score:.6f} analyze={exp} {'OK' if ok else 'MISMATCH'}")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(*sys.argv[1:4]) else 0)
