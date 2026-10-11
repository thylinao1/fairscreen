# Cluster run log (NUS SoC Slurm, vLLM)

Status: IN PROGRESS (started 2026-10-12 00:48 SGT). This file is updated as jobs finish.

## Input
- `~/fairscreen/prompts_final_v11.jsonl` on the cluster (dataset v1.1, 11,328 lines, 4 recipes x 2,832:
  direct, fairness, structured, extract_only; longest prompt 5,004 characters). The earlier
  `prompts_final.jsonl` was never run (it was withdrawn before any job was submitted).
- Each prompt sent as one user message, no system prompt, through the model's own chat template.

## Settings (all models)
- vLLM 0.28.0 (`~/miniconda3/envs/bcf`, torch 2.13.0+cu130), `vllm serve` on 127.0.0.1, driven by
  `cluster/fs_client.py` (aiohttp, concurrency 64, retries 4, resumable by id).
- temperature 0, `--generation-config vllm` (the model's generation_config.json sampling defaults
  are NOT applied, so only temperature 0 and max_tokens set sampling), `--seed 0`,
  `--max-model-len 8192`, `--max-num-seqs 128`, gpu_memory_utilization min(0.90, free/total - 0.04).
- Default batched kernels (VLLM_BATCH_INVARIANT not set): greedy outputs are not guaranteed
  identical across batch compositions, so exact repeats can differ (see protocol: repeat disagreement).
- max_tokens: 4096 for gpt-oss-120b and gpt-oss-20b, 1024 for the others.
  Note: `configs/panel.json` cluster_panel.settings says "max 4096 new tokens" for all models;
  finish_reason=length counts below show whether the 1024 cap bound any output.
- gpt-oss: default reasoning (vLLM harmony path; effort medium by default; vLLM's harmony system
  message carries the current date). `text` = final channel only; `reasoning_chars` = length of the
  analysis channel. Reasoning text kept on the cluster only, in `~/fairscreen/out_v11/<slug>.reasoning.jsonl`.
- Qwen3-32B: `chat_template_kwargs={"enable_thinking": false}` on every request.
- gemma-3-27b-it: `--limit-mm-per-prompt {"image":0}` (text only).
- Mistral-Small-3.2-24B-Instruct-2506: `--tokenizer-mode mistral --config-format mistral
  --load-format mistral --limit-mm-per-prompt {"image":0}` (the model card's serving mode;
  mistral_common chat template; no system prompt added).

## Deviation from the brief: weights location
The brief said gpt-oss-120b and gpt-oss-20b were cached in `~/.cache/huggingface`; they are not
(only `config.json`, 59 KB each). All six models need downloading (about 263 GB), and home was at
283 GB of a ~500 GB hidden quota. So `HF_HOME` stays at the default (token) but `HF_HUB_CACHE` points
to node-local `/tmp/$SLURM_JOB_ID/hub` (GPU nodes have 3 to 13.9 TB local NVMe), deleted at job end.
Nothing was added to home except outputs and logs.

## Jobs
(filled in below)
