# Cluster run log (NUS SoC Slurm, vLLM)

Runs started 2026-10-12 00:48 SGT. All six models finished, and their outputs are in `results/cluster_raw/`.

## Input
- `~/fairscreen/prompts_final_v11.jsonl` on the cluster (dataset v1.1, 11,328 lines, 4 recipes x 2,832:
  direct, fairness, structured, extract_only; longest prompt 5,004 characters). The earlier
  `prompts_final.jsonl` was withdrawn before any job was submitted and never run.
- Each prompt is sent as one user message, no system prompt, through the model's own chat template.

## Settings (all models)
- vLLM 0.28.0 (`~/miniconda3/envs/bcf`, torch 2.13.0+cu130), `vllm serve` on 127.0.0.1, driven by
  `cluster/fs_client.py` (aiohttp, concurrency 64, retries 4, resumable by id).
- temperature 0, `--generation-config vllm` (the sampling defaults in the model's generation_config.json
  are not applied, so only temperature 0 and max_tokens set sampling), `--seed 0`,
  `--max-model-len 8192`, `--max-num-seqs 128`, gpu_memory_utilization min(0.90, free/total - 0.04).
- Default batched kernels (VLLM_BATCH_INVARIANT not set): greedy outputs can change with the batch
  composition, so exact repeats can differ (see protocol: repeat disagreement).
- max_tokens: 4096 for gpt-oss-120b and gpt-oss-20b, 1024 for the others.
  `configs/panel.json` cluster_panel.settings says "max 4096 new tokens" for all models. The
  finish_reason=length counts in `results/sensitivity_checks.json` (`length_stops`) show whether the 1024 cap
  cut any output.
- gpt-oss: default reasoning through vLLM's harmony path. Harmony is the gpt-oss response format: it writes the
  reasoning to an analysis channel and the answer to a final channel. Reasoning effort is medium by default, and
  vLLM's harmony system message carries the current date. `text` = final channel only; `reasoning_chars` = length
  of the analysis channel. Reasoning text kept on the cluster only, in `~/fairscreen/out_v11/<slug>.reasoning.jsonl`.
- Qwen3-32B: `chat_template_kwargs={"enable_thinking": false}` on every request.
- gemma-3-27b-it: `--limit-mm-per-prompt {"image":0}` (text only).
- Mistral-Small-3.2-24B-Instruct-2506: `--tokenizer-mode mistral --config-format mistral
  --load-format mistral --limit-mm-per-prompt {"image":0}` (the model card's serving mode;
  mistral_common chat template; no system prompt added).

## Deviation: weights location
The setup assumed that gpt-oss-120b and gpt-oss-20b were cached in `~/.cache/huggingface`. For both models that
cache holds only `config.json` (59 KB each), not the weights. All six models need downloading (about 263 GB), and
home was at 283 GB of a ~500 GB hidden quota (the cluster's quota command does not report it).
So `HF_HOME` stays at the default, which holds the Hugging Face access token. `HF_HUB_CACHE` points
to node-local `/tmp/$SLURM_JOB_ID/hub` on the GPU node's local NVMe disk (3 to 13.9 TB per node).
The directory `/tmp/$SLURM_JOB_ID/hub` is deleted at job end.
Nothing was added to home except outputs and logs.

## Jobs (state at about 01:07 SGT, 12 Oct, before any full run finished)
Outputs: `~/fairscreen/out_v11/<slug>.jsonl` on the cluster. Logs: `~/fairscreen/logs/<jobname>-<jobid>.out`
and `<slug>-<jobid>.vllm.log`. Each job: partition gpu, 8 CPUs, 03:00:00, exclude xgpj0.

| job | name | model (slug) | GPU requested | mem | state |
|---|---|---|---|---|---|
| 934074 | fs-smoke | gpt-oss-20b + llama-3.1-8b-instruct, first 20 prompts | h100-47 | 128G | running on xgpi14 (H100 NVL MIG 3g.47gb); gpt-oss-20b PASSED; Llama step was loading |
| 934084 | fs-gptoss120 | gpt-oss-120b | h200-141 | 192G | pending (Priority) |
| 934085 | fs-qwen32 | qwen3-32b | h100-96 | 192G | running on xgpi4 |
| 934086 | fs-gemma27 | gemma-3-27b-it | a100-80 | 128G | running on xgph0 |
| 934087 | fs-gptoss20 | gpt-oss-20b | h100-47 | 128G | running on xgpi16 |
| 934088 | fs-mistral24 | mistral-small-3.2-24b | a100-80 | 128G | pending, afterany:934086 |
| 934089 | fs-llama8 | llama-3.1-8b-instruct | h100-47 | 128G | pending, afterany:934087 |

The gpt-oss-20b smoke test ran the first 20 prompts, all from the direct recipe, into out_v11_smoke/.
20/20 responses ended normally (finish_reason stop), and 20/20 texts are a JSON object (e.g.
`{"decision":"MEETS"}`). For 20/20 the reasoning was stored separately (916 to 1,397 chars). The answer text
held 0 harmony markers, the control tokens of the harmony format. The median response was 482 completion tokens.
The 20 prompts took 13 s, and the server was ready in 286 s including the 13 GB download.
The llama-3.1-8b-instruct smoke test (same 20 prompts, 01:09 SGT) gave 20/20 stop and 20/20 JSON object, with 15
completion tokens each. Its server was ready in 330 s including the 15 GB download. Both code paths (harmony and
plain chat) pass.

## Resubmitting and collecting outputs
- A job that hits the 3 h limit can be resubmitted with the same command, because the client skips done ids.
- `cluster/fetch_done.sh <slug>` checks for 11,328 unique ids on the cluster, then gzips the output and copies it
  to `results/cluster_raw/<slug>.jsonl.gz`.
