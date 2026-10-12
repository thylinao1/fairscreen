"""Figures for the FairScreen write-up, drawn from analysis outputs (metrics.json), never from model calls.

python analysis/figures.py results/final_cluster/metrics.json results/figures/cluster "Cluster panel"
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette (dataviz skill, light mode), fixed order; text never wears series colour.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
CONFIG = {  # configuration -> (label, colour slot, marker)
    "direct": ("Direct", "#2a78d6", "o"),
    "fairness": ("Fairness instruction", "#eb6834", "s"),
    "structured": ("Extract, then model decides", "#1baf7a", "D"),
    "policy_code_same_extraction": ("Same extraction, code decides", "#6250d6", "^"),
    "extract_only": ("Extract only, code decides", "#eda100", "v"),
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False})


def pretty(model):
    m = model
    for org in ("openai_", "google_", "anthropic_", "qwen_", "meta-llama_", "mistralai_", "Qwen_", "deepseek-ai_", "ibm_"):
        if m.startswith(org):
            m = m[len(org):]
    m = m.split("/")[-1].replace("qwen3.6_35b", "qwen3.6-35b")
    for a, b in (("-2026-03-17", ""), ("-20251001", ""), ("@default", ""), ("-instruct-2507", ""), ("-preview", ""),
                 (".jsonl", "")):
        m = m.replace(a, b)
    return m


def load(path):
    d = json.load(open(path))
    out = {}
    for k, v in d.items():
        model, config = k.split("|")
        out[(model, config)] = v
    return out


def fig_fidelity(M, out, title):
    models = sorted({m for m, _ in M}, key=lambda m: -max((M[(m, c)]["robust"][0] for c in CONFIG if (m, c) in M), default=0))
    configs = [c for c in CONFIG if any((m, c) in M for m in models)]
    fig, ax = plt.subplots(figsize=(8.2, 1.7 + 0.95 * len(models)))
    step = 0.76 / max(1, len(configs))
    for j, c in enumerate(configs):
        label, col, mk = CONFIG[c]
        ys, xs, lo, hi = [], [], [], []
        for i, m in enumerate(models):
            if (m, c) not in M:
                continue
            r = M[(m, c)]["robust"]
            ys.append(i + (j - (len(configs) - 1) / 2) * step)
            xs.append(100 * r[0]); lo.append(100 * (r[0] - r[1])); hi.append(100 * (r[2] - r[0]))
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt=mk, color=col, ecolor=col, elinewidth=1.4, capsize=0, ms=7,
                    mec=SURFACE, mew=1.5, label=label)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([pretty(m) for m in models], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-2, 102)
    ax.set_xlabel("Robust policy fidelity: families with all eight paired decisions correct (%)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_controls(M, out, title, config="direct"):
    models = [m for (m, c) in M if c == config]
    models.sort(key=lambda m: -M[(m, config)]["identity_flip_ctrl"][0])
    keys = [("identity_flip_ctrl", "Identity swapped", "#2a78d6"),
            ("same_group_flip", "Same-group name swapped", "#eb6834"),
            ("repeat_flip", "Identical prompt repeated", "#1baf7a")]
    fig, ax = plt.subplots(figsize=(8.2, 0.6 + 0.6 * len(models)))
    h = 0.26
    for j, (k, label, col) in enumerate(keys):
        vals = [100 * M[(m, config)][k][0] if M[(m, config)][k][0] == M[(m, config)][k][0] else 0 for m in models]
        ax.barh([i + (j - 1) * h for i in range(len(models))], vals, height=h - 0.04, color=col, label=label)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([pretty(m) for m in models], color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Share of matched pairs whose decisions differ (%)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_transfer(M, out, title):
    configs = [c for c in ("direct", "fairness", "structured", "policy_code_same_extraction") if any(k[1] == c for k in M)]
    fig, axes = plt.subplots(1, len(configs), figsize=(2.6 * len(configs) + 0.6, 3.6), sharey=True)
    if len(configs) == 1:
        axes = [axes]
    for ax, c in zip(axes, configs):
        label, col, mk = CONFIG[c]
        for (m, cc), v in M.items():
            if cc != c:
                continue
            a, b = 100 * v["clean_quartet"][0], 100 * v["changed_quartet"][0]
            ax.plot([0, 1], [a, b], color=col, lw=2, marker=mk, ms=6, mec=SURFACE, mew=1.2, alpha=0.9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Clean", "Changed"], color=INK)
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(-3, 103)
        ax.set_title(label, fontsize=9.5, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0].set_ylabel("Quartet pass rate (%)")
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_decomposition(M, out, title):
    rows = [(m, v["decomposition"]) for (m, c), v in M.items() if c == "structured" and "decomposition" in v]
    rows.sort(key=lambda r: -sum(v for k, v in r[1]["tab"].items() if k.startswith("extract_ok=True|verdict_ok=False")) / max(1, r[1]["n"]))
    cats = [("extract_ok=True|verdict_ok=True", "Decisive facts right, decision right", "#1baf7a"),
            ("extract_ok=True|verdict_ok=False", "Decisive facts right, decision wrong", "#eb6834"),
            ("extract_ok=False|verdict_ok=True", "Decisive facts wrong, decision right", "#2a78d6"),
            ("extract_ok=False|verdict_ok=False", "Decisive facts wrong, decision wrong", "#e34948")]
    fig, ax = plt.subplots(figsize=(8.2, 0.7 + 0.5 * len(rows)))
    for i, (m, d) in enumerate(rows):
        n = d["n"] or 1
        left = 0
        for key, label, col in cats:
            val = 100 * sum(v for k, v in d["tab"].items() if k.startswith(key)) / n
            ax.barh(i, val, left=left, color=col, height=0.6, label=label if i == 0 else None, edgecolor=SURFACE, linewidth=1)
            left += val
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([pretty(m) for m, _ in rows], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Core decisions in the structured configuration (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(metrics_path, outdir, label):
    os.makedirs(outdir, exist_ok=True)
    M = load(metrics_path)
    fig_fidelity(M, f"{outdir}/fig1_fidelity.png", f"Robust policy fidelity by configuration, {label} (95% intervals)")
    fig_controls(M, f"{outdir}/fig2_controls.png", f"Decision changes: identity swap versus controls, {label}")
    fig_transfer(M, f"{outdir}/fig3_transfer.png", f"Clean versus changed presentation, {label}")
    fig_decomposition(M, f"{outdir}/fig4_decomposition.png", f"Where structured decisions go wrong, {label}")
    print("figures in", outdir)


if __name__ == "__main__":
    main(*sys.argv[1:4])
