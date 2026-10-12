"""Cross-platform check: gpt-oss-120b on Kaggle (FairScreen-48) versus the NUS cluster on the same 48 families."""
import json
import sys
from collections import Counter


def fam_scores(path, fids, key="decision", gold=None, valid_only=False):
    recs = [json.loads(l) for l in open(path)]
    by = {r["variant_id"]: r for r in recs if r["rep"] == 0}
    out, n_bad = {}, 0
    for fid in fids:
        ids = [f"{fid}/q={q}-id={i}-ctx={c}" for q in ("hi", "lo") for i in ("A", "B") for c in ("clean", "changed")]
        rs = [by.get(i) for i in ids]
        bad = any(r is None or r["status"] in ("invalid", "empty", "transport_error") or (key == "replay" and r.get("replay") is None) for r in rs)
        n_bad += bad
        if valid_only and bad:
            continue
        out[fid] = all(r is not None and r.get(key) == r["gold"] for r in rs)
    return out, n_bad


fids = [f"core-final-{i:04d}" for i in range(48)]
rows = []
for cfg, task, rec, key in (("direct", "fairscreen-direct", "direct", "decision"),
                            ("fairness", "fairscreen-fairness-instruction", "fairness", "decision"),
                            ("structured", "fairscreen-structured", "structured", "decision"),
                            ("policy code, same extraction", "fairscreen-structured", "structured", "replay")):
    k, kb = fam_scores(f"results/records/{task}/openai_gpt-oss-120b.jsonl", fids, key)
    kv, _ = fam_scores(f"results/records/{task}/openai_gpt-oss-120b.jsonl", fids, key, valid_only=True)
    c, cb = fam_scores(f"results/cluster_records/gpt-oss-120b/{rec}.jsonl", fids, key)
    agree = sum(k[f] == c[f] for f in fids)
    rows.append((cfg, sum(k.values()) / 48, kb, sum(kv.values()) / max(1, len(kv)), len(kv), sum(c.values()) / 48, cb, agree))
# same-family comparison on the families where every Kaggle direct answer is complete
k, _ = fam_scores("results/records/fairscreen-direct/openai_gpt-oss-120b.jsonl", fids, "decision", valid_only=True)
c, _ = fam_scores("results/cluster_records/gpt-oss-120b/direct.jsonl", fids, "decision")
same = sorted(k)
print(f"SAME_FAMILIES direct: n={len(same)} kaggle_pass={sum(k[f] for f in same)} cluster_pass={sum(c[f] for f in same)}")
print("| Configuration | Kaggle (all 48) | Kaggle families with a malformed answer | Kaggle (complete families) | Cluster (all 48) | Same family outcome |")
print("|---|---|---|---|---|---|")
for cfg, k, kb, kv, nkv, c, cb, agree in rows:
    print(f"| {cfg} | {round(100*k+1e-9) if (100*k)%1!=0.5 else int(100*k)+1}% | {kb} | {100*kv:.0f}% (n={nkv}) | {100*c:.0f}% | {agree}/48 |")
