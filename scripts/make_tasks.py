"""Generate self-contained Kaggle task files (runtime and runner inlined).

python scripts/make_tasks.py --recipe direct --name fairscreen-direct --data dataset
python scripts/make_tasks.py --recipe direct --name fairscreen-dev-smoke --data embed --split dev --limit-families 6 --suites core
"""
import argparse
import base64
import gzip
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
DESCRIPTIONS = {
    "direct": "Direct screening: the model reads the policy and the application and returns a decision.",
    "fairness": "Fairness instruction: direct screening plus a fixed instruction to disregard personal characteristics.",
    "structured": "Structured model judgment: the model extracts typed facts with evidence quotes, then gives its own decision.",
    "extract_only": "Extraction plus policy code: the model only extracts typed facts; fixed policy code makes the decision.",
}
SHORT = {
    "direct": "FairScreen, direct screening. Score: share of applicant families with all 8 paired decisions correct (qualification x identity x presentation), with 95% bootstrap half-width.",
    "fairness": "FairScreen, fairness instruction. Score: share of applicant families with all 8 paired decisions correct (qualification x identity x presentation), with 95% bootstrap half-width.",
    "structured": "FairScreen, extract facts then model decides. Score: share of applicant families with all 8 paired decisions correct, with 95% bootstrap half-width.",
    "extract_only": "FairScreen, model extracts facts and policy code decides. Score: share of applicant families with all 8 paired decisions correct, with 95% bootstrap half-width.",
}
COMMON_DESC = (" Score: robust policy fidelity, the share of independent applicant families for which all eight paired "
               "decisions (qualification crossed or not x two identity signals x clean or changed presentation) match "
               "the policy-defined gold label; plus or minus a 95 percent stratified bootstrap half-width. "
               "Fictional applicants and policies; Singapore identity signals. github.com/thylinao1/fairscreen")


def strip_module(path):
    src = open(path).read()
    # drop the module docstring and relative imports; the inlined modules share one namespace
    lines = [l for l in src.splitlines() if not l.startswith("from .")]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", choices=["embed", "dataset"], default="dataset")
    ap.add_argument("--split", default="final")
    ap.add_argument("--limit-families", type=int, default=0)
    ap.add_argument("--suites", default="")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="kaggle/tasks")
    ap.add_argument("--core-slice", type=int, default=0, help="keep the first N core families, octets only (no controls, no repeats)")
    a = ap.parse_args()
    runtime = strip_module(os.path.join(ROOT, "src/fairscreen/runtime.py"))
    runner = strip_module(os.path.join(ROOT, "src/fairscreen/runner.py"))
    if a.data == "embed":
        variants = [json.loads(l) for l in open(os.path.join(ROOT, f"data/{a.split}/variants.jsonl"))]
        families = [json.loads(l) for l in open(os.path.join(ROOT, f"data/{a.split}/families.jsonl"))]
        suites = set(a.suites.split(",")) if a.suites else None
        if suites:
            families = [f for f in families if f["suite"] in suites]
        if a.limit_families:
            per = {}
            keep = []
            for f in families:
                per[f["suite"]] = per.get(f["suite"], 0) + 1
                if per[f["suite"]] <= a.limit_families:
                    keep.append(f)
            families = keep
        if a.core_slice:
            families = [dict(f, repeats=0, variant_ids=[i for i in f["variant_ids"] if "control=" not in i])
                        for f in families if f["suite"] == "core" and int(f["family_id"].split("-")[-1]) < a.core_slice]
        ids = {f["family_id"] for f in families}
        keep_ids = {i for f in families for i in f["variant_ids"]}
        variants = [v for v in variants if v["family_id"] in ids and v["variant_id"] in keep_ids]
        slim = ("variant_id", "family_id", "suite", "tags", "policy_text", "application_text", "framing_text",
                "fact_keys", "policy_spec", "gold_decision")
        variants = [{k: v[k] for k in slim} for v in variants]
        blob = base64.b64encode(gzip.compress(json.dumps({"variants": variants, "families": families}).encode())).decode()
        loader = f'_BLOB = "{blob}"\nDATA = json.loads(gzip.decompress(base64.b64decode(_BLOB)))\nVARIANTS, FAMILIES = DATA["variants"], DATA["families"]\n'
    else:
        loader = (
            'def _find(name):\n'
            '    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) + glob.glob(f"data/final/{name}")\n'
            '    if not hits:\n'
            '        raise FileNotFoundError(name)\n'
            '    return hits[0]\n'
            'VARIANTS = [json.loads(l) for l in open(_find("variants.jsonl"))]\n'
            'FAMILIES = [json.loads(l) for l in open(_find("families.jsonl"))]\n')
    desc = DESCRIPTIONS[a.recipe] + COMMON_DESC
    short = SHORT[a.recipe]
    assert len(short) <= 255 and len(a.name) <= 255
    fn = a.name.replace("-", "_")
    task = f'''# %% [markdown]
# # FairScreen: {DESCRIPTIONS[a.recipe].split(":")[0]}
# {desc}
#
# Each model call runs in its own fresh chat named `recipe|variant_id|rep|attempt`, so the run file keeps every
# prompt and response. The runtime below is inlined verbatim from the repository so local recomputation runs the
# same parser, policy replay and scoring code.

# %%
import base64, contextvars, glob, gzip, json, os, time
from concurrent.futures import ThreadPoolExecutor
import kaggle_benchmarks as kbench

# %%
# ---------------- FairScreen runtime (src/fairscreen/runtime.py) ----------------
{runtime}

# ---------------- FairScreen runner (src/fairscreen/runner.py) ----------------
{runner}

# %%
{loader}
RECIPE = "{a.recipe}"
THREADS = {a.threads}
print(f"FairScreen runtime {{RUNTIME_VERSION}}: {{len(VARIANTS)}} variants, {{len(FAMILIES)}} families, recipe {{RECIPE}}")


def make_call(llm):
    def call(prompt, tag):
        try:
            with kbench.chats.new(tag) as chat:
                text = llm.prompt(prompt)
            u = chat.usage
            usage = {{"in": u.input_tokens, "out": u.output_tokens}} if u else {{}}
            return text, None, usage
        except Exception as e:  # infrastructure or provider error; retried by the runner
            return None, repr(e)[:300], {{}}
    return call


# %%
@kbench.task(name="{a.name}", description="{short}")
def {fn}(llm) -> tuple[float, float]:
    calls = plan_calls(VARIANTS, FAMILIES)
    call = make_call(llm)
    with ThreadPoolExecutor(THREADS) as ex:
        futs = [ex.submit(contextvars.copy_context().run, evaluate_variant, v, RECIPE, call, rep) for v, rep in calls]
        records = [f.result() for f in futs]
    summary = summarize(records, FAMILIES)
    if RECIPE == "structured":
        summary["policy_code_on_same_extraction"] = summarize(records, FAMILIES, decision_key="replay")["robust_policy_fidelity"]
    print("FAIRSCREEN_SUMMARY " + json.dumps(summary))
    m = summary["robust_policy_fidelity"]
    return (float(m["mean"]), float((m["hi"] - m["lo"]) / 2))


{fn}.run(kbench.llm)
'''
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.name}.py")
    open(path, "w").write(task)
    print(path, len(task))


if __name__ == "__main__":
    main()
