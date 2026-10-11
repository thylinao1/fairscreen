"""Run a FairScreen split locally against the Kaggle model proxy or SoCLaaS.

python scripts/run_local.py --split dev --model kaggle:openai/gpt-5.4-nano-2026-03-17 --recipes direct,structured
python scripts/run_local.py --split dev --model soclaas:qwen3.6:35b --threads 4 --suites core
Records append to results/raw_local/<split>/<model-slug>/<recipe>.jsonl (resumable).
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fairscreen.runtime import *  # noqa: E402,F401,F403
from fairscreen import runtime as RT  # noqa: E402
import fairscreen.runner as RN  # noqa: E402

for name in ("build_prompt", "parse_response", "parse_facts", "replay", "family_pass", "bootstrap_mean"):
    setattr(RN, name, getattr(RT, name))


def kaggle_call(model):
    import kaggle_benchmarks as kbench
    llm = kbench.llms[model]

    def call(prompt, tag):
        try:
            with kbench.chats.new(tag) as chat:
                text = llm.prompt(prompt)
            u = chat.usage
            usage = {"in": u.input_tokens, "out": u.output_tokens, "cost_nd": (u.input_tokens_cost_nanodollars or 0)
                     + (u.output_tokens_cost_nanodollars or 0), "ms": u.total_backend_latency_ms} if u else {}
            return text, None, usage
        except Exception as e:  # infrastructure or provider error; retried by the runner
            return None, repr(e)[:300], {}
    return call


def soclaas_call(model, lock_rate=0.5):
    from openai import OpenAI
    client = OpenAI(base_url=os.environ["SOCLAAS_BASE_URL"], api_key=os.environ["SOCLAAS_API_KEY"])
    lock = threading.Lock()
    last = [0.0]

    def call(prompt, tag):
        with lock:  # stay under 30 requests per minute
            wait = 2.1 - (time.time() - last[0])
            if wait > 0:
                time.sleep(wait)
            last[0] = time.time()
        try:
            kw = {"reasoning_effort": "low"} if model == os.environ.get("SOCLAAS_GLM_MODEL") else {}
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                               temperature=0, max_tokens=1500, **kw)
            u = r.usage
            return r.choices[0].message.content, None, {"in": u.prompt_tokens, "out": u.completion_tokens}
        except Exception as e:
            return None, repr(e)[:300], {}
    return call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model", required=True)
    ap.add_argument("--recipes", default="direct,fairness,structured")
    ap.add_argument("--suites", default="")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--limit-families", type=int, default=0)
    a = ap.parse_args()
    variants = [json.loads(l) for l in open(f"data/{a.split}/variants.jsonl")]
    families = [json.loads(l) for l in open(f"data/{a.split}/families.jsonl")]
    suites = set(a.suites.split(",")) if a.suites else None
    if a.limit_families:
        keep, seen = set(), {}
        for f in families:
            seen.setdefault(f["suite"], 0)
            if seen[f["suite"]] < a.limit_families:
                keep.add(f["family_id"])
                seen[f["suite"]] += 1
        families = [f for f in families if f["family_id"] in keep]
        variants = [v for v in variants if v["family_id"] in keep]
    backend, model = a.model.split(":", 1)
    call = kaggle_call(model) if backend == "kaggle" else soclaas_call(model)
    slug = model.replace("/", "_").replace(":", "_")
    for recipe in a.recipes.split(","):
        out = f"results/raw_local/{a.split}/{slug}/{recipe}.jsonl"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        done = set()
        if os.path.exists(out):
            for l in open(out):
                r = json.loads(l)
                if r["status"] != "transport_error":
                    done.add((r["variant_id"], r["rep"]))
        calls = [(v, rep) for v, rep in RN.plan_calls(variants, families, suites) if (v["variant_id"], rep) not in done]
        print(f"[{slug}/{recipe}] {len(calls)} calls to make, {len(done)} done", flush=True)
        lock = threading.Lock()
        t0 = time.time()
        with open(out, "a") as fh, ThreadPoolExecutor(a.threads) as ex:
            futs = [ex.submit(RN.evaluate_variant, v, recipe, call, rep) for v, rep in calls]
            for i, f in enumerate(as_completed(futs)):
                rec = f.result()
                with lock:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                if (i + 1) % 100 == 0:
                    print(f"  {i+1}/{len(calls)} {time.time()-t0:.0f}s", flush=True)
        recs = [json.loads(l) for l in open(out)]
        recs = [r for r in recs if r["status"] != "transport_error"] + [r for r in recs if r["status"] == "transport_error"]
        fam_used = [f for f in families if not suites or f["suite"] in suites]
        s = RN.summarize(recs, fam_used)
        if recipe == "structured":
            s["replay"] = RN.summarize(recs, fam_used, decision_key="replay")["robust_policy_fidelity"]
        cost = sum((r.get("usage") or {}).get("cost_nd", 0) or 0 for r in recs) / 1e9
        print(f"[{slug}/{recipe}] {json.dumps(s)} cost=${cost:.4f}", flush=True)


if __name__ == "__main__":
    main()
