"""Build the development and confirmatory datasets and their manifest.

Usage: python -m fairscreen.build_dataset [--out data] [--final-core 192]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter

from . import world
from .suites import (CONTRASTS, audit_family, core_family, realistic_family, relevance_family,
                     sufficiency_family)
from .generate import CONTEXT_CHANGES

DATASET_VERSION = "1.0.0"


def build_split(split: str, n_core: int, n_suff: int, n_rel: int, n_real: int, n_audit: int,
                control_every: int = 4, length_every: int = 4, repeat_every: int = 10) -> list:
    occs = world.DEV_OCCUPATIONS if split == "dev" else world.OCCUPATIONS
    fams = []
    for i in range(n_core):
        occ = occs[i % len(occs)]
        contrast = CONTRASTS[(i // len(occs)) % len(CONTRASTS)]
        changed = CONTEXT_CHANGES[(i // (len(occs) * len(CONTRASTS)) + i) % len(CONTEXT_CHANGES)]
        fams.append(core_family(i, split, occ, contrast, changed,
                                with_controls=(i % control_every == 1),
                                with_length_control=(changed in ("org_background", "distractor") and i % length_every == 2),
                                repeats=2 if i % repeat_every == 3 else 0))
    # Diagnostic suites use policies with two requirements of different kinds across all occupations of the split.
    for i in range(n_suff):
        fams.append(sufficiency_family(i, split, occs[i % len(occs)]))
    for i in range(n_rel):
        fams.append(relevance_family(i, split))
    for i in range(n_real):
        fams.append(realistic_family(i, split, occs[i % len(occs)]))
    for i in range(n_audit):
        fams.append(audit_family(i, split, occs[i % len(occs)]))
    return fams


def write(fams: list, path: str) -> dict:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    variants = [v for f in fams for v in f["variants"]]
    with open(path, "w") as fh:
        for v in variants:
            fh.write(json.dumps(v, sort_keys=True) + "\n")
    fam_path = path.replace("variants", "families")
    with open(fam_path, "w") as fh:
        for f in fams:
            meta = {k: v for k, v in f.items() if k != "variants"}
            meta["variant_ids"] = [v["variant_id"] for v in f["variants"]]
            fh.write(json.dumps(meta, sort_keys=True) + "\n")
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    return {"variants": path, "families": fam_path, "n_families": len(fams), "n_variants": len(variants),
            "sha256_variants": sha(path), "sha256_families": sha(fam_path),
            "suites": dict(Counter(f["suite"] for f in fams)),
            "gold": dict(Counter((v["suite"], v["gold_decision"]) for v in variants).most_common()) and
            {f"{s}:{g}": n for (s, g), n in Counter((v["suite"], v["gold_decision"]) for v in variants).items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--dev-core", type=int, default=48)
    ap.add_argument("--final-core", type=int, default=192)
    ap.add_argument("--final-suff", type=int, default=48)
    ap.add_argument("--final-rel", type=int, default=32)
    ap.add_argument("--final-real", type=int, default=48)
    ap.add_argument("--final-audit", type=int, default=24)
    a = ap.parse_args()
    manifest = {"dataset_version": DATASET_VERSION, "splits": {}}
    dev = build_split("dev", a.dev_core, 16, 8, 8, 8)
    manifest["splits"]["dev"] = write(dev, f"{a.out}/dev/variants.jsonl")
    fin = build_split("final", a.final_core, a.final_suff, a.final_rel, a.final_real, a.final_audit)
    manifest["splits"]["final"] = write(fin, f"{a.out}/final/variants.jsonl")
    with open(f"{a.out}/manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
