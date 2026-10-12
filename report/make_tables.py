"""Generate the article's tables from analysis outputs, so no number is typed by hand.

python report/make_tables.py results/final_kaggle results/final_cluster > report/tables.md
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
from figures import pretty  # noqa: E402

LABEL = {"direct": "Direct", "fairness": "Fairness instruction", "structured": "Extract, then model decides",
         "policy_code_same_extraction": "Same extraction, code decides", "extract_only": "Extract only, code decides"}


def pct(x):
    return "n/a" if x is None or x != x else f"{100 * x:.0f}%"


def ci(t):
    return f"{100 * t[0]:.0f}% [{100 * t[1]:.0f}, {100 * t[2]:.0f}]"


def main_table(metrics, configs):
    models = sorted({k.split("|")[0] for k in metrics})
    head = "| Model | " + " | ".join(LABEL[c] for c in configs) + " |"
    lines = [head, "|---|" + "---|" * len(configs)]
    rows = []
    for m in models:
        cells = []
        for c in configs:
            v = metrics.get(f"{m}|{c}")
            cells.append(ci(v["robust"]) if v else "not run")
        best = max((metrics[f"{m}|{c}"]["robust"][0] for c in configs if f"{m}|{c}" in metrics), default=0)
        rows.append((-best, f"| {pretty(m)} | " + " | ".join(cells) + " |"))
    return "\n".join(lines + [r for _, r in sorted(rows)])


def controls_table(metrics, config="direct"):  # noqa: C901
    lines = ["| Model | Qualification response | False reject | False accept | Abstain on complete files | Identity swap flips | Same-group name flips | Neutral ID flips | Exact repeat flips | Invalid |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(metrics.items()):
        m, c = k.split("|")
        if c != config:
            continue
        lines.append(f"| {pretty(m)} | {pct(v['qual_response'])} | {pct(v.get('core_false_reject'))} | {pct(v.get('core_false_accept'))} | {pct(v.get('core_abstention'))} | {pct(v['identity_flip'][0])} (n={v['identity_flip'][1]}) | "
                     f"{pct(v['same_group_flip'][0])} (n={v['same_group_flip'][1]}) | {pct(v['neutral_id_flip'][0])} (n={v['neutral_id_flip'][1]}) | "
                     f"{pct(v['repeat_flip'][0])} (n={v['repeat_flip'][1]}) | {pct(v['invalid'])} |")
    return "\n".join(lines)


def suites_table(metrics, config="direct"):
    lines = ["| Model | Sufficiency families | Appropriate uncertainty | Unnecessary abstention | Relevance families | Realistic families | Audit families |",
             "|---|---|---|---|---|---|---|"]
    for k, v in sorted(metrics.items()):
        m, c = k.split("|")
        if c != config or "sufficiency_pass" not in v:
            continue
        g = lambda key: pct(v[key][0]) if key in v else "n/a"
        lines.append(f"| {pretty(m)} | {g('sufficiency_pass')} | {pct(v.get('appropriate_uncertainty'))} | "
                     f"{pct(v.get('unnecessary_abstention'))} | {g('relevance_pass')} | {g('realistic_pass')} | {g('audit_pass')} |")
    return "\n".join(lines)


def contrasts_table(contrasts):
    lines = ["| Contrast | Estimate (pp) | 95% interval | p | Holm p | Models |", "|---|---|---|---|---|---|"]
    for k, v in contrasts.items():
        if v and "models" in v:
            lines.append(f"| {k} | {100 * v['estimate']:+.1f} | [{100 * v['lo']:+.1f}, {100 * v['hi']:+.1f}] | {v['p']:.3f} | "
                         f"{v.get('p_holm', float('nan')):.3f} | {len(v['models'])} |")
    return "\n".join(lines)


def main(*dirs):
    for d in dirs:
        if not os.path.exists(f"{d}/metrics.json"):
            continue
        metrics = json.load(open(f"{d}/metrics.json"))
        contrasts = json.load(open(f"{d}/contrasts.json"))
        configs = [c for c in LABEL if any(k.endswith("|" + c) for k in metrics)]
        print(f"## {d}\n\n### Robust policy fidelity\n\n{main_table(metrics, configs)}\n")
        print(f"### Controls (direct)\n\n{controls_table(metrics)}\n")
        s = suites_table(metrics)
        if s.count("\n") > 1:
            print(f"### Diagnostic suites (direct)\n\n{s}\n")
        print(f"### Contrasts\n\n{contrasts_table(contrasts)}\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
