#!/usr/bin/env python3
"""Summarise a FairScreen output file. Usage: fs_check.py <out.jsonl> [check|nocheck]

Prints line count, unique ids, finish reasons, how many texts contain a parseable JSON object,
reasoning presence and leaked harmony markers. In 'check' mode (smoke test) exits 1 when fewer
than 90 percent of texts parse as a JSON object, any text is empty, or harmony markers leak.
"""
import json
import sys
from collections import Counter

HARMONY_MARKERS = ("<|channel|>", "<|start|>", "<|message|>", "<|end|>")


def json_object_in(text):
    if not text:
        return False
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return False
    try:
        return isinstance(json.loads(text[a:b + 1]), dict)
    except ValueError:
        return False


def main():
    path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "nocheck"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    ids = {r["id"] for r in rows}
    fin = Counter(r["finish_reason"] for r in rows)
    parse = sum(json_object_in(r["text"]) for r in rows)
    empty = sum(not r["text"] for r in rows)
    with_reasoning = sum((r.get("reasoning_chars") or 0) > 0 for r in rows)
    leaks = sum(any(m in (r["text"] or "") for m in HARMONY_MARKERS) for r in rows)
    ctoks = sorted(r["completion_tokens"] or 0 for r in rows)
    med = ctoks[len(ctoks) // 2] if ctoks else 0
    print(json.dumps({"file": path, "lines": len(rows), "unique_ids": len(ids), "finish": dict(fin),
                      "json_object": parse, "empty_text": empty, "with_reasoning": with_reasoning,
                      "harmony_leaks": leaks, "median_completion_tokens": med,
                      "max_completion_tokens": ctoks[-1] if ctoks else 0}))
    if mode == "check":
        bad = (not rows) or parse < 0.9 * len(rows) or empty > 0 or leaks > 0
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
