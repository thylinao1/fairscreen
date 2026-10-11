#!/usr/bin/env python3
"""FairScreen cluster client.

Sends each prompt as a single user message (no system prompt) to a local vLLM OpenAI-compatible
server and appends one JSON line per prompt to the output file:
    {"id", "model", "text", "finish_reason", "prompt_tokens", "completion_tokens", "reasoning_chars"}

Resumable: ids already present in the output file are skipped. Only successful responses are
written there; request failures go to <out>.errors.jsonl and are retried on the next run.
Reasoning text (gpt-oss) is kept on the cluster in <out minus .jsonl>.reasoning.jsonl.
"""
import argparse
import asyncio
import json
import os
import sys
import time

import aiohttp

MAX_ATTEMPTS = 4
REQUEST_TIMEOUT_S = 1800
PROGRESS_EVERY = 250


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--served-name", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--limit", type=int, default=0, help="first N prompts only; 0 = all")
    p.add_argument("--extra-body", default="{}", help="JSON merged into every request body")
    return p.parse_args()


def load_done(path):
    """Ids already written; drops a partial trailing line left by a killed run."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "rb+") as f:
        data = f.read()
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            f.truncate(cut)
            data = data[:cut]
    for line in data.splitlines():
        try:
            done.add(json.loads(line)["id"])
        except (ValueError, KeyError):
            continue
    return done


def load_prompts(path, limit):
    recs = []
    with open(path) as f:
        for line in f:
            recs.append(json.loads(line))
            if limit and len(recs) >= limit:
                break
    return recs


async def post_chat(session, url, payload):
    last_err = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
            async with session.post(url, json=payload, timeout=timeout) as r:
                body = await r.text()
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}: {body[:300]}")
                return json.loads(body)
        except Exception as e:  # retried, then reported to the error log
            last_err = e
            await asyncio.sleep(5 * (attempt + 1))
    raise last_err


def to_record(rec, resp, model_id):
    choice = resp["choices"][0]
    msg = choice.get("message") or {}
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    usage = resp.get("usage") or {}
    out = {
        "id": rec["id"],
        "model": model_id,
        "text": msg.get("content"),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_chars": len(reasoning) if reasoning else 0,
    }
    return out, reasoning


async def main():
    args = parse_args()
    extra = json.loads(args.extra_body)
    recs = load_prompts(args.prompts, args.limit)
    done = load_done(args.out)
    todo = [r for r in recs if r["id"] not in done]
    total = len(recs)
    print(f"[client] {args.served_name}: total {total}, already done {total - len(todo)}, todo {len(todo)}",
          flush=True)
    if not todo:
        return 0

    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    base = args.out[:-6] if args.out.endswith(".jsonl") else args.out
    queue = asyncio.Queue()
    for r in todo:
        queue.put_nowait(r)
    stats = {"ok": 0, "err": 0}
    t0 = time.time()

    with open(args.out, "a") as fout, open(base + ".reasoning.jsonl", "a") as freas, \
            open(args.out + ".errors.jsonl", "a") as ferr:

        async def worker(session):
            while True:
                try:
                    rec = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                payload = {
                    "model": args.served_name,
                    "messages": [{"role": "user", "content": rec["prompt"]}],
                    "temperature": 0,
                    "max_tokens": args.max_tokens,
                    **extra,
                }
                try:
                    resp = await post_chat(session, url, payload)
                    out, reasoning = to_record(rec, resp, args.model_id)
                except Exception as e:
                    stats["err"] += 1
                    ferr.write(json.dumps({"id": rec["id"], "error": str(e)[:500], "t": time.time()}) + "\n")
                    ferr.flush()
                    continue
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                if reasoning:
                    freas.write(json.dumps({"id": rec["id"], "reasoning": reasoning}, ensure_ascii=False) + "\n")
                    freas.flush()
                stats["ok"] += 1
                if stats["ok"] % PROGRESS_EVERY == 0:
                    el = time.time() - t0
                    rate = stats["ok"] / el * 60
                    left = len(todo) - stats["ok"] - stats["err"]
                    print(f"[client] {time.strftime('%H:%M:%S')} done {stats['ok']}/{len(todo)} "
                          f"err {stats['err']} rate {rate:.0f}/min eta {left / max(rate, 1e-9):.0f} min",
                          flush=True)

        conn = aiohttp.TCPConnector(limit=args.concurrency)
        async with aiohttp.ClientSession(connector=conn) as session:
            await asyncio.gather(*(worker(session) for _ in range(args.concurrency)))

    remaining = len(recs) - len(load_done(args.out) & {r["id"] for r in recs})
    print(f"[client] finished in {time.time() - t0:.0f}s ok {stats['ok']} err {stats['err']} "
          f"remaining {remaining}", flush=True)
    return 0 if remaining == 0 else 3


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
