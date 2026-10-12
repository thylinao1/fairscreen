# FairScreen

A paired counterfactual benchmark for LLM eligibility screening, built for the DEV x Kaggle Benchmarking Challenge.

An LLM reads an eligibility policy and a fictional application and answers `MEETS`, `DOES_NOT_MEET` or
`INSUFFICIENT_EVIDENCE`. In each core family, a qualification edit crosses a stated requirement or not. An identity
edit changes Singapore names by ethnicity and gender, or an ethnic self-help-group volunteering marker with a neutral
applicant ID. A presentation edit changes employer background, section order, wording and distractor details. A
correct screener must change its decision when the requirement is crossed, keep it when only the identity changes, and
do both in the changed presentation. The primary score, robust policy fidelity, is the share of families with all
eight decisions correct. A screener that gives everyone the same answer scores zero.

## Results in brief

Open-weight panel (NUS cluster, 192 core families, six models) and Kaggle panel (48 core families, five models):

- Capable models are at or near the ceiling on the core suite (Gemini 3.7 Flash and Gemma 4 31B 100 percent on
  Kaggle, gpt-oss-120b 98 percent and gpt-oss-20b 96 percent on the cluster). Models below the ceiling fail in
  systematic, model-specific ways: too strict, too lenient, or using "insufficient evidence" to avoid a decision.
- Identity swaps changed about as many decisions as swaps between names of the same group (cluster control
  families: 2.6 versus 2.2 percent).
- A fixed fairness instruction had no measurable effect (cluster C1: -0.3 points, 95 percent interval -1.2
  to +0.6).
- Applying the policy with fixed code to the model's own extracted facts raised robust policy fidelity by 20.5
  points on the cluster panel (interval +18.1 to +22.8). The gain is for models whose decisions contradict their own
  extraction. Applying the policy with fixed code can make missing-evidence cases worse when the model records a partial
  value instead of "unknown".
- Exploratory (re-run after the main analysis): Qwen3-32B scored 98 percent direct with thinking on and 39 percent with
  thinking off. Thinking off (the panel run) is a listed deviation from the registered default reasoning. Thinking on
  used about 40 times the output tokens (median 476 against 12 per direct decision;
  results/exploratory/qwen3-32b-thinking/direct.jsonl and results/cluster_records/qwen3-32b/direct.jsonl).
- Exploratory (not registered): GLM 5.3 Flash via NUS SoCLaaS scored 98 percent direct and 93 percent with
  the fairness instruction (results/exploratory/).
- On the same families, gpt-oss-120b scored 63 percent (30 of 48) on Kaggle and 98 percent on the cluster. The
  Kaggle-hosted answers were cut off mid-word.

`report/fill_article.py` generates the write-up, `report/article_filled.md`, from `report/article.md` and takes
every number from result files. Tables: `report/tables.md`. Figures: `results/figures/`.

## Layout

| Path | What it holds |
|---|---|
| `protocol.md` | Registered questions, scores, contrasts, margins, execution rules, deviations |
| `configs/panel.json` | Registered model panels and the stopping rule |
| `src/fairscreen/` | Policy language and oracle A, world (occupations, names, markers), records, renderers, suites, dataset builder, runtime (prompts, fixed parser, oracle B replay, scoring), runner |
| `data/` | Development and confirmatory datasets (v1.1.0) with manifest and checksums; `DATASET_CARD.md` |
| `tests/` | Validity tests: oracle agreement, edit invariants, suite contracts, leakage, parser, baselines |
| `validation/` | Blind review items, answer key, the two Claude Sonnet readers' decisions, defect register |
| `kaggle/tasks/` | The four public Kaggle task files and a private development task (runtime inlined, data embedded) |
| `cluster/` | vLLM job scripts, client, run log |
| `results/kaggle_runs/` | Raw Kaggle run files (every prompt and response) |
| `results/cluster_raw/` | Raw cluster outputs (gzipped JSONL) |
| `analysis/` | Record builders, main analysis, independent recomputation, sensitivity checks, rule-parser and shortcut baselines, cross-platform check, figures |
| `related-work.md` | Closest prior work and what is new here |

## Reproduce

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install kaggle-benchmarks pandas scikit-learn matplotlib pytest
PYTHONPATH=src python -m fairscreen.build_dataset          # regenerates data/ (checksums in data/manifest.json)
PYTHONPATH=src pytest tests                                 # 95 tests: validity, analysis guards, strict parser
python analysis/rule_baseline.py --split final              # text-only parser agrees with gold on 2528/2528
python analysis/kaggle_records.py results/kaggle_runs       # rebuild records from raw Kaggle run files
python analysis/cluster_records.py                          # rebuild records from raw cluster outputs
python analysis/analyze.py --records results/records --split final --out results/final_kaggle
python analysis/analyze.py --local results/cluster_records --split final --out results/final_cluster
python analysis/recompute_check.py results/cluster_records final results/final_cluster/main_table.csv
python analysis/sensitivity_checks.py
python report/fill_article.py
```

No step calls a model.

The exact package versions used are in `requirements-lock.txt`.

### Notes on intervals

Family-level percentile bootstrap intervals collapse to a single point when every family passes or none does, so the
write-up reports exact (Clopper-Pearson style, one-sided 2.5 percent) bounds in those cases. Bootstrap p-values come
from 2,000 resamples, so a saved p of 0 means below 1 in 2,000.

## Licence

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`data/LICENSE-DATA.md`). All applicants, employers and policies are fictional.
