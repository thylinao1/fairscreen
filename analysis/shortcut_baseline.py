"""Text-only shortcut baseline: TF-IDF + logistic regression trained on development items, scored on the
held-out confirmatory set with the same family rule. High scores here would mean lexical shortcuts.

python analysis/shortcut_baseline.py
"""
import json
import os
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fairscreen import runtime as RT  # noqa: E402


def load(split):
    return ([json.loads(l) for l in open(f"data/{split}/variants.jsonl")],
            [json.loads(l) for l in open(f"data/{split}/families.jsonl")])


def main():
    dv, _ = load("dev")
    fv, ff = load("final")
    text = lambda v: v["policy_text"] + "\n" + v["application_text"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform([text(v) for v in dv])
    clf = LogisticRegression(max_iter=2000, C=4.0).fit(X, [v["gold_decision"] for v in dv])
    pred = clf.predict(vec.transform([text(v) for v in fv]))
    recs = [{"variant_id": v["variant_id"], "decision": p, "gold": v["gold_decision"]} for v, p in zip(fv, pred)]
    core = [f for f in ff if f["suite"] == "core"]
    prim = lambda f: [i for i in f["variant_ids"] if "control=" not in i]
    fam = sum(RT.family_pass(recs, prim(f)) for f in core) / len(core)
    acc_core = sum(r["decision"] == r["gold"] for r in recs if "core-" in r["variant_id"] and "control=" not in r["variant_id"])
    n_core = sum(1 for r in recs if "core-" in r["variant_id"] and "control=" not in r["variant_id"])
    suff = [f for f in ff if f["suite"] == "sufficiency"]
    sfam = sum(RT.family_pass(recs, f["variant_ids"]) for f in suff) / len(suff)
    out = {"core_decision_accuracy": acc_core / n_core, "core_robust_fidelity": fam, "sufficiency_family_pass": sfam,
           "train_items": len(dv), "test_items": len(fv)}
    print(json.dumps(out, indent=1))
    os.makedirs("results/baselines", exist_ok=True)
    json.dump(out, open("results/baselines/shortcut_tfidf.json", "w"), indent=1)


if __name__ == "__main__":
    main()
