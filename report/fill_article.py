"""Fill every [[placeholder]] in report/article.md from analysis outputs (claim trace by construction).

python report/fill_article.py  -> report/article_filled.md and report/article_values.json; lists unfilled placeholders.
[[KAGGLE LINK]] is filled in by hand with the public benchmark URL.
"""
import glob
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from figures import pretty  # noqa: E402

KM = json.load(open("results/final_kaggle/metrics.json"))
KC = json.load(open("results/final_kaggle/contrasts.json"))
CM = json.load(open("results/final_cluster/metrics.json"))
CC = json.load(open("results/final_cluster/contrasts.json"))
VS = {json.loads(l)["variant_id"]: json.loads(l) for l in open("data/final/variants.jsonl")}


def pct(x, d=0):
    from decimal import ROUND_HALF_UP, Decimal
    q = Decimal(1).scaleb(-d)
    return f"{Decimal(str(100 * x)).quantize(q, rounding=ROUND_HALF_UP)}%"


def pp(x, d=1):
    return f"{100 * x:+.{d}f}"


def ci(lo, hi, d=1):
    return f"{100 * lo:+.{d}f} to {100 * hi:+.{d}f}"


def recs(path):
    return [json.loads(l) for l in open(path)]


V = {}
# ---------------------------------------------------------------- registered contrasts
for tag, C in (("cluster", CC), ("kaggle", KC)):
    for k, name in (("c1", "C1_fairness_minus_direct"), ("c2", "C2_transfer"), ("c3", "C3_policy_code_minus_model_verdict")):
        V[f"{k}_{tag}"] = pp(C[name]["estimate"])
        V[f"{k}_{tag}_ci"] = ci(C[name]["lo"], C[name]["hi"])
        hp = C[name]["p_holm"]
        V[f"{k}_{tag}_p"] = "< 0.001" if hp < 0.001 else f"= {hp:.2f}"
V["llama_struct"] = pct(CM["llama-3.1-8b-instruct|structured"]["robust"][0])
V["llama_code"] = pct(CM["llama-3.1-8b-instruct|policy_code_same_extraction"]["robust"][0])

# ---------------------------------------------------------------- identity controls
idc, idf = CC["S_identity_vs_same_group_direct"], CC["S_identity_vs_same_group_fairness"]
V["id_flip"], V["sg_flip"] = pct(idc["identity_flip"], 1), pct(idc["same_group_flip"], 1)
V["id_minus_sg"], V["id_minus_sg_ci"] = pp(idc["difference"]), ci(idc["lo"], idc["hi"])
V["n_ctrl_fam"] = str(idc["n_families"])
V["id_flip_fair"], V["sg_flip_fair"] = pct(idf["identity_flip"], 1), pct(idf["same_group_flip"], 1)
rep = [(CM[k]["repeat_flip"][0], CM[k]["repeat_flip"][1]) for k in CM if k.endswith("|direct")]
V["repeat_flip"] = pct(sum(r * n for r, n in rep) / sum(n for _, n in rep), 1)
V["nano_flip"] = pct(KM["openai_gpt-5.4-nano-2026-03-17|direct"]["identity_flip"][0], 1)
V["fl_flip"] = pct(KM["google_gemini-3.1-flash-lite-preview|direct"]["identity_flip"][0], 1)
cells = [x for C in (CC, KC) for cfg in ("direct", "fairness", "structured") for x in C[f"S_signed_by_group_{cfg}"].values()]
V["n_cells"] = str(len(cells))
V["n_cells_ex"] = str(sum(x["lo"] > 0 or x["hi"] < 0 for x in cells))
V["n_cells_chance"] = f"{0.05 * len(cells):.0f}"
_keyed = [(tag, cfg, k, x) for tag, C in (("cluster", CC), ("kaggle", KC)) for cfg in ("direct", "fairness", "structured")
          for k, x in C[f"S_signed_by_group_{cfg}"].items()]
_neg = sorted([t for t in _keyed if t[3]["hi"] < 0], key=lambda t: t[3]["estimate"])
assert len(_neg) == 2 and all(tag == "kaggle" and k.startswith("indian_") and k.endswith("|hi") for tag, _, k, _ in _neg), \
    "rewrite the signed-cell sentence"
assert _neg[0][1:3] == ("direct", "indian_female|hi"), "rewrite the signed-cell sentence"
V["signed_max"], V["signed_max_ci"] = pp(_neg[0][3]["estimate"]), ci(_neg[0][3]["lo"], _neg[0][3]["hi"])
RC = json.load(open("results/sensitivity_checks.json"))
V["c3_think"], V["c3_think_ci"] = pp(RC["c3_with_qwen_thinking_on"]["estimate"]), ci(RC["c3_with_qwen_thinking_on"]["lo"], RC["c3_with_qwen_thinking_on"]["hi"])
q = CC["S_qual_response_fairness_minus_direct"]
V["qual_cluster"], V["qual_cluster_ci"] = pp(q["estimate"]), "interval " + ci(q["lo"], q["hi"])


# ---------------------------------------------------------------- model table (exact bounds where the bootstrap collapses)
def exact_bounds(k, n, a=0.05):
    if k == n:
        return (a / 2) ** (1 / n), 1.0
    if k == 0:
        return 0.0, 1 - (a / 2) ** (1 / n)
    return None


NOTES = {("gpt-oss-20b", "cluster"): "reads a Higher Nitec as meeting \"diploma or above\"",
         ("gpt-oss-120b", "Kaggle"): "hosted answers cut off mid-word (finding 5)"}
rows = []
for M, plat, n in ((KM, "Kaggle", 48), (CM, "cluster", 192)):
    for key, v in M.items():
        m, c = key.split("|")
        if c != "direct":
            continue
        r, lo, hi = v["robust"]
        eb = exact_bounds(round(r * n), n)
        rng = f"{100 * eb[0]:.0f} to {100 * eb[1]:.0f}, exact" if eb else f"{100 * lo:.0f} to {100 * hi:.0f}"
        if (pretty(m), plat) in NOTES:
            err = NOTES[(pretty(m), plat)]
        elif r >= 0.97:
            err = "none"
        else:
            kind, val = max((("rejects qualified applicants", v["core_false_reject"]),
                             ("approves unqualified applicants", v["core_false_accept"]),
                             ("says INSUFFICIENT_EVIDENCE on complete files", v["core_abstention"])), key=lambda t: t[1])
            err = f"{kind} ({pct(val)})"
        rows.append((plat, -r, f"| {pretty(m)} | {plat} ({n}) | {pct(r)} ({rng}) | {err} |"))
sb = json.load(open("results/baselines/shortcut_tfidf.json"))
rows.append(("zz", 0, f"| bag-of-words baseline, no LLM | cluster set (192) | {pct(sb['core_robust_fidelity'])} | trained on development items |"))
V["model_table"] = ("| Model | Platform (families) | Direct: all eight right (95% interval) | Main error on the core |\n"
                    "|---|---|---|---|\n" + "\n".join(r for _, _, r in sorted(rows)))
V["shortcut_core"] = pct(sb["core_robust_fidelity"])
qt = CM["qwen3-32b|direct"]["robust_by_decisive_type"]
V["q_years"], V["q_assess"] = pct(qt["min_years"][0]), pct(qt["assessment"][0])

# ---------------------------------------------------------------- code versus model decision
nit_n = nit_m = 0
for r in recs("results/cluster_records/gpt-oss-20b/structured.jsonl"):
    v = VS[r["variant_id"]]
    if r["rep"] == 0 and v["suite"] == "core" and v["gold_facts"].get("degree_level") == "NITEC" and v["gold_decision"] == "DOES_NOT_MEET":
        nit_n += 1
        nit_m += r["decision"] == "MEETS"
V["oss20_nitec"] = f"{nit_m} of {nit_n}"
crow = []
for M, plat in ((CM, "open-weight"), (KM, "Kaggle")):
    for key in M:
        m, c = key.split("|")
        if c != "structured":
            continue
        a, b = M[key]["robust"][0], M[f"{m}|policy_code_same_extraction"]["robust"][0]
        note = " (answers cut off, finding 5)" if plat == "Kaggle" and "gpt-oss" in m else ""
        crow.append((plat != "open-weight", -(b - a), f"| {pretty(m)}, {plat}{note} | {pct(a)} | {pct(b)} |"))
V["code_table"] = "\n".join(r for *_, r in sorted(crow))
ko = recs("results/records/fairscreen-structured/openai_gpt-oss-120b.jsonl")
V["n_trunc_struct"] = str(sum(r["rep"] == 0 and r.get("replay") is None for r in ko))
V["nano_code_gain"] = f"{100 * (KM['openai_gpt-5.4-nano-2026-03-17|policy_code_same_extraction']['robust'][0] - KM['openai_gpt-5.4-nano-2026-03-17|structured']['robust'][0]):+.0f}"
ex = []
for r in recs("results/cluster_records/gpt-oss-120b/structured.jsonl"):
    v = VS[r["variant_id"]]
    if r["rep"] == 0 and v["suite"] == "sufficiency" and v["tags"]["ev"] == "missing" and r["decision"] == "INSUFFICIENT_EVIDENCE" \
            and r.get("replay") == "DOES_NOT_MEET":
        yk = [k for k in v["fact_keys"] if k.startswith("years:")]
        if yk and r["facts"].get(yk[0]) is not None:
            ex.append((r["variant_id"], r["facts"][yk[0]]))
V["partial_example"] = f"{sorted(ex)[0][1]:.1f} years"
V["oss120_suff"] = pct(CM["gpt-oss-120b|structured"]["sufficiency_pass"][0])
V["oss120_suff_code"] = pct(CM["gpt-oss-120b|policy_code_same_extraction"]["sufficiency_pass"][0])

# ---------------------------------------------------------------- cross-platform
xp = open("results/cross_platform_gptoss120b.md").read().splitlines()
same = re.search(r"n=(\d+) kaggle_pass=(\d+) cluster_pass=(\d+)", xp[0])
V["same_n"], V["same_k"] = same.group(1), same.group(2)
assert same.group(2) == same.group(3), "platforms disagree on the complete families; rewrite finding 5"
direct = [l for l in xp if l.startswith("| direct |")][0].split("|")
V["oss120_kaggle48"] = pct(KM["openai_gpt-oss-120b|direct"]["robust"][0])  # 30 of 48, half-up rounding
V["oss120_cluster48"] = direct[5].strip()
kd = recs("results/records/fairscreen-direct/openai_gpt-oss-120b.jsonl")
bad = [r["usage"]["out"] for r in kd if r["status"] == "invalid"]
good = [r["usage"]["out"] for r in kd if r["status"] == "json"]
V["trunc_lo"], V["trunc_hi"], V["complete_hi"] = f"{min(bad):,}", f"{max(bad):,}", f"{max(good):,}"
cnt = Counter(r["variant_id"].split("ctx=")[1] for r in kd if r["status"] == "invalid")
V["trunc_ratio"] = f"{cnt['changed'] / max(1, cnt['clean']):.1f} times as common"

# ---------------------------------------------------------------- suites (open-weight panel, direct)
dm = {k.split("|")[0]: v for k, v in CM.items() if k.endswith("|direct")}
conf = [v["suff_cells"]["conflict"]["INSUFFICIENT_EVIDENCE"] for v in dm.values()]
V["conflict_lo"], V["conflict_hi"] = pct(min(conf)), pct(max(conf))
miss = sorted((v["suff_cells"]["missing"]["INSUFFICIENT_EVIDENCE"], pretty(m)) for m, v in dm.items())
V["missing_range"] = f"from {pct(miss[0][0])} for {miss[0][1]} to {pct(miss[-1][0])} for {miss[-1][1]}"
V["llama_unnec"] = pct(dm["llama-3.1-8b-instruct"]["unnecessary_abstention"])
mids = [dm[m]["realistic_pass"][0] for m in ("gemma-3-27b-it", "qwen3-32b", "mistral-small-3.2-24b")]
V["real_mid_range"] = f"{100 * min(mids):.0f} to {100 * max(mids):.0f} percent"
V["mis_neutral"] = pct(dm["mistral-small-3.2-24b"]["audit_neutral"]["quartet_pass"])
V["mis_oper"] = pct(dm["mistral-small-3.2-24b"]["audit_operational"]["quartet_pass"])
# Kaggle evidence-sufficiency task, if its runs have been downloaded and rebuilt
fams = [json.loads(l) for l in open("data/final/families.jsonl") if '"suite": "sufficiency"' in l]
ks = []
for path in sorted(glob.glob("results/records/fairscreen-evidence-sufficiency/*.jsonl")):
    rs = {r["variant_id"]: r for r in recs(path) if r["rep"] == 0}
    if len(rs) < 192:
        continue
    ok = [f for f in fams if all(rs[i]["status"] not in ("transport_error", "invalid", "empty") for i in f["variant_ids"])]
    passed = sum(all(rs[i]["decision"] == rs[i]["gold"] for i in f["variant_ids"]) for f in fams)
    failed_calls = sum(r["status"] == "transport_error" for r in rs.values())
    passed_ok = sum(all(rs[i]["decision"] == rs[i]["gold"] for i in f["variant_ids"]) for f in ok)
    txt = f"{pretty(os.path.basename(path)[:-6])} {passed} of 48"
    cut = sum(r["status"] == "invalid" for r in rs.values())
    if cut:
        txt += (f" ({cut} answers cut off mid-word as in finding 5; {passed_ok} of the {len(ok)} families with complete "
                f"answers passed)")
    answered = [r for r in rs.values() if r["status"] not in ("transport_error", "invalid", "empty")]
    all_right = all(r["decision"] == r["gold"] for r in answered)
    if failed_calls:
        assert all_right, "rewrite the sufficiency sentence: not every answered item is correct"
        txt += (f" ({failed_calls} of its 192 calls failed three times without any response, most likely because the run started "
                f"with my quota nearly spent; it answered every remaining item correctly, {passed_ok} of {len(ok)} complete families)")
    ks.append(txt)
V["kaggle_suff"] = ("On Kaggle's sufficiency task (48 families, four decisions each) the hosted models passed: "
                    + "; ".join(ks) + ".") if ks else ""

# ---------------------------------------------------------------- round-2 additions
V["q_flip"] = pct(CM["qwen3-32b|direct"]["identity_flip"][0], 1)
V["q_flip_struct"] = pct(CM["qwen3-32b|structured"]["identity_flip"][0], 1)
V["c1_kaggle_hi"] = f"{100 * KC['C1_fairness_minus_direct']['hi']:+.1f}"
V["nano_flip_fair"] = pct(KM["openai_gpt-5.4-nano-2026-03-17|fairness"]["identity_flip"][0], 1)
qk = KC["S_qual_response_fairness_minus_direct"]
V["qual_kaggle"], V["qual_kaggle_ci"] = pp(qk["estimate"]), "interval " + ci(qk["lo"], qk["hi"])
ids = CC["S_identity_vs_same_group_structured"]
V["id_flip_struct"], V["sg_flip_struct"] = pct(ids["identity_flip"], 1), pct(ids["same_group_flip"], 1)
V["id_minus_sg_struct"], V["id_minus_sg_struct_ci"] = pp(ids["difference"]), ci(ids["lo"], ids["hi"])
V["mis_direct"], V["mis_struct"] = pct(CM["mistral-small-3.2-24b|direct"]["robust"][0]), pct(CM["mistral-small-3.2-24b|structured"]["robust"][0])
V["fl"], V["fl_struct"] = pct(KM["google_gemini-3.1-flash-lite-preview|direct"]["robust"][0]), pct(KM["google_gemini-3.1-flash-lite-preview|structured"]["robust"][0])
V["llama_xo"] = pct(CM["llama-3.1-8b-instruct|extract_only"]["robust"][0])
xo = CC["S_extract_only_minus_policy_code_same_extraction"]
assert xo["estimate"] < 0 and xo["hi"] < 0, "rewrite the extract-only sentence: it says 'lower' and resolved"
V["extract_only_diff"], V["extract_only_ci"] = f"{abs(100 * xo['estimate']):.1f}", ci(xo["lo"], xo["hi"])
for tag, task in (("direct", "fairscreen-direct"), ("fair", "fairscreen-fairness-instruction")):
    rr = recs(f"results/records/{task}/openai_gpt-oss-120b.jsonl")
    ctx = Counter(r["variant_id"].split("ctx=")[1] for r in rr if r["status"] == "invalid")
    V[f"trunc_{tag}_changed"], V[f"trunc_{tag}_n"] = str(ctx["changed"]), str(sum(ctx.values()))
pm = KC["C2_transfer"]["per_model"]
V["c2_oss"] = f"{100 * pm['openai_gpt-oss-120b']:+.1f}"
V["c2_rest"] = f"{100 * sum(v for m, v in pm.items() if 'gpt-oss' not in m) / (len(pm) - 1):+.1f}"
trows = ["| Model | Families passed | Families with every answer complete | Passed among those |", "|---|---|---|---|"]
gem_failed = 0
for path in sorted(glob.glob("results/records/fairscreen-evidence-sufficiency/*.jsonl")):
    rs = {r["variant_id"]: r for r in recs(path) if r["rep"] == 0}
    ok = [f for f in fams if all(rs[i]["status"] not in ("transport_error", "invalid", "empty") for i in f["variant_ids"])]
    passed = sum(all(rs[i]["decision"] == rs[i]["gold"] for i in f["variant_ids"]) for f in fams)
    passed_ok = sum(all(rs[i]["decision"] == rs[i]["gold"] for i in f["variant_ids"]) for f in ok)
    name = pretty(os.path.basename(path)[:-6])
    if "gemini-3.7" in name:
        gem_failed = sum(r["status"] == "transport_error" for r in rs.values())
        answered = [r for r in rs.values() if r["status"] not in ("transport_error", "invalid", "empty")]
        assert all(r["decision"] == r["gold"] for r in answered)
    trows.append(f"| {name} | {passed} of 48 | {len(ok)} | {passed_ok} |")
V["kaggle_suff_table"] = "\n".join(trows)
V["gem_failed"] = str(gem_failed)

# ---------------------------------------------------------------- one complete family (the opening example's family)
FID = "core-final-0166"
cells = [f"{FID}/q={q}-id={i}-ctx={c}" for q in ("hi", "lo") for i in ("A", "B") for c in ("clean", "changed")]
ab = {"MEETS": "M", "DOES_NOT_MEET": "DNM", "INSUFFICIENT_EVIDENCE": "IE", None: "none"}


def fam_dec(path, key="decision"):
    return {r["variant_id"]: r.get(key) for r in recs(path) if r["rep"] == 0 and r["family_id"] == FID}


hi_y = VS[cells[0]]["gold_facts"]["years:technical writing"]
lo_y = VS[cells[4]]["gold_facts"]["years:technical writing"]
na, nb = VS[cells[0]]["identity_label"], VS[cells[2]]["identity_label"]
head = ["Screener"] + [f"{y:g} yrs, {n}, {'clean' if c == 'clean' else 'reworded'}"
                       for y in (hi_y, lo_y) for n in ("A", "B") for c in ("clean", "changed")]
frows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head),
         "| Correct answer | " + " | ".join(ab[VS[c]["gold_decision"]] for c in cells) + " |"]
for label, path, key in (("Qwen3-32B, direct", "results/cluster_records/qwen3-32b/direct.jsonl", "decision"),
                         ("Qwen3-32B, extract then decide", "results/cluster_records/qwen3-32b/structured.jsonl", "decision"),
                         ("Qwen3-32B, same extraction, code", "results/cluster_records/qwen3-32b/structured.jsonl", "replay"),
                         ("Gemma 3 27B, direct", "results/cluster_records/gemma-3-27b-it/direct.jsonl", "decision"),
                         ("Llama 3.1 8B, direct", "results/cluster_records/llama-3.1-8b-instruct/direct.jsonl", "decision"),
                         ("gpt-oss-120b, direct", "results/cluster_records/gpt-oss-120b/direct.jsonl", "decision")):
    d = fam_dec(path, key)
    frows.append(f"| {label} | " + " | ".join(ab[d[c]] if d[c] == VS[c]["gold_decision"] else f"**{ab[d[c]]}**" for c in cells) + " |")
V["family_table"] = "\n".join(frows)
V["fam_na"], V["fam_nb"], V["fam_hi"], V["fam_lo"] = na, nb, f"{hi_y:g}", f"{lo_y:g}"

# ---------------------------------------------------------------- exploratory GLM run (not registered)
G = json.load(open("results/exploratory/glm53flash_summary.json"))
V["glm_direct"], V["glm_fair"] = pct(G["direct_robust"][0]), pct(G["fairness_robust"][0])
V["glm_diff"], V["glm_ci"] = pp(G["fairness_minus_direct"]["estimate"]), ci(G["fairness_minus_direct"]["lo"], G["fairness_minus_direct"]["hi"])

# ---------------------------------------------------------------- exploratory thinking-on run (not registered)
T = json.load(open("results/exploratory/qwen3-32b-thinking_summary.json"))
td = T["direct"]
V["th_on"], V["th_off"] = pct(td["thinking_on"][0]), pct(td["thinking_off"][0])
V["th_diff"], V["th_ci"] = pp(td["difference"]["estimate"]), ci(td["difference"]["lo"], td["difference"]["hi"])
V["th_fa_on"], V["th_fa_off"] = pct(td["false_accept"][0]), pct(td["false_accept"][1])
V["th_years_on"], V["th_years_off"] = pct(td["by_type_on"]["min_years"][0]), pct(td["by_type_off"]["min_years"][0])
V["th_id_on"], V["th_id_off"] = pct(td["identity_flip"][0][0], 1), pct(td["identity_flip"][1][0], 1)
import statistics
_on = [json.loads(l)["usage"]["out"] for l in open("results/exploratory/qwen3-32b-thinking/direct.jsonl")]
_off = [json.loads(l)["usage"]["out"] for l in open("results/cluster_records/qwen3-32b/direct.jsonl") if '"suite": "core"' in l]
V["th_tok_on"], V["th_tok_off"] = f"{statistics.median(_on):.0f}", f"{statistics.median(_off):.0f}"
V["th_tok_ratio"] = f"{statistics.median(_on) / statistics.median(_off):.0f}"
V["th_n_fam"] = "192"
_toks = {}
for _m, _path, _rob in [(k.split("|")[0], f"results/records/fairscreen-direct/{k.split('|')[0]}.jsonl", v["robust"][0]) for k, v in KM.items() if k.endswith("|direct")] + \
                       [(k.split("|")[0], f"results/cluster_records/{k.split('|')[0]}/direct.jsonl", v["robust"][0]) for k, v in CM.items() if k.endswith("|direct")]:
    _rs = [json.loads(l) for l in open(_path) if '"suite": "core"' in l]
    _comp = KM.get(f"{_m}|direct", {}).get("robust_completed_items", [[_rob]])[0][0] if _m.startswith(("openai_gpt-oss",)) else _rob
    _toks[(_m, _path)] = (statistics.median([r["usage"]["out"] or 0 for r in _rs]), max(_rob, _comp))
_hi = [t for t, r in _toks.values() if r >= 0.9]
_lo = [t for t, r in _toks.values() if r < 0.9]
assert max(_lo) < min(_hi), "reasoning-length split does not separate the panels; rewrite the sentence"
V["reason_lo"], V["reason_hi"] = f"{min(_hi):.0f}", f"{max(_hi):.0f}"
V["noreason_lo"], V["noreason_hi"] = f"{min(_lo):.0f}", f"{max(_lo):.0f}"
_fd = {json.loads(l)["variant_id"]: json.loads(l)["decision"] for l in open("results/exploratory/qwen3-32b-thinking/direct.jsonl")}
assert all(_fd[c] == VS[c]["gold_decision"] for c in cells), "rewrite: thinking run did not get the opening family fully right"

# ---------------------------------------------------------------- protocol deviations
proto = open("protocol.md").read().split("## Deviations", 1)[1]
V["n_deviations"] = str(sum(1 for l in proto.splitlines() if l.startswith("- ")))
V["n_later_dev"] = ("zero one two three four five six seven eight nine ten eleven twelve").split()[int(V["n_deviations"]) - 1]

text = open("report/article.md").read()
for k, val in V.items():
    text = text.replace(f"[[{k}]]", val)
left = sorted(set(re.findall(r"\[\[([^\]]+)\]\]", text)))
open("report/article_filled.md", "w").write(text)
json.dump(V, open("report/article_values.json", "w"), indent=1)
words = len(re.sub(r"```.*?```", "", text.split("---", 2)[2], flags=re.S).split())
print("filled", len(V), "values; unfilled:", left, "; words:", words)
