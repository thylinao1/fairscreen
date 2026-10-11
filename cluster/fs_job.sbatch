#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --exclude=xgpj0
#SBATCH --output=/home/e/e1506804/fairscreen/logs/%x-%j.out
# FairScreen cluster run: serve one model at a time with vLLM and run the prompt file through it.
# usage: sbatch --job-name=fs-<x> --gpus=<type>:1 --mem=<n>G fs_job.sbatch <outdir> <limit|0> <check|nocheck> <slug> [<slug> ...]
set -o pipefail
OUTDIR=$1; LIMIT=$2; MODE=$3; shift 3
ROOT=$HOME/fairscreen
PROMPTS=$ROOT/prompts_final_v11.jsonl
SCR=/tmp/$SLURM_JOB_ID
mkdir -p "$SCR" "$OUTDIR" "$ROOT/logs"
SERVER_PID=""

cleanup() {
  if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null; sleep 10; kill -9 "$SERVER_PID" 2>/dev/null; fi
  rm -rf "$SCR"
}
trap cleanup EXIT

echo "job $SLURM_JOB_ID node $(hostname) start $(date -Is) outdir $OUTDIR limit $LIMIT mode $MODE models $*"
nvidia-smi -L || { echo "FATAL: nvidia-smi failed"; exit 21; }
source ~/miniconda3/bin/activate bcf
export VLLM_CACHE_ROOT=$SCR/vllm TRITON_CACHE_DIR=$SCR/triton XDG_CACHE_HOME=$SCR/xdg
# HF_HOME stays at the default (token lives there); weights go to node-local disk because the
# home quota (~500 GB, 283 GB used) cannot hold all six models.
export HF_HUB_CACHE=$SCR/hub
df -h /tmp | tail -1

UTIL=$(python - <<'EOF'
import sys, torch
if not torch.cuda.is_available():
    sys.exit(1)
f, t = torch.cuda.mem_get_info()
print(f"{min(0.90, f / t - 0.04):.3f}")
print(f"cuda ok {torch.__version__} {torch.cuda.get_device_name(0)} free {f/2**30:.1f}/{t/2**30:.1f} GiB", file=sys.stderr)
EOF
) || { echo "FATAL: no CUDA device visible"; exit 21; }
echo "gpu_memory_utilization $UTIL; vllm $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"

model_cfg() {
  EXTRA=(); BODY='{}'
  case $1 in
    gpt-oss-120b) HF=openai/gpt-oss-120b; MAXTOK=4096 ;;
    gpt-oss-20b) HF=openai/gpt-oss-20b; MAXTOK=4096 ;;
    qwen3-32b) HF=Qwen/Qwen3-32B; MAXTOK=1024; BODY='{"chat_template_kwargs":{"enable_thinking":false}}' ;;
    gemma-3-27b-it) HF=google/gemma-3-27b-it; MAXTOK=1024; EXTRA=(--limit-mm-per-prompt '{"image":0}') ;;
    mistral-small-3.2-24b) HF=mistralai/Mistral-Small-3.2-24B-Instruct-2506; MAXTOK=1024
      EXTRA=(--tokenizer-mode mistral --config-format mistral --load-format mistral --limit-mm-per-prompt '{"image":0}') ;;
    llama-3.1-8b-instruct) HF=meta-llama/Llama-3.1-8B-Instruct; MAXTOK=1024 ;;
    *) echo "unknown slug $1"; return 1 ;;
  esac
}

run_one() {
  local slug=$1
  model_cfg "$slug" || return 1
  local port=$((20000 + SLURM_JOB_ID % 20000))
  local vlog=$ROOT/logs/$slug-$SLURM_JOB_ID.vllm.log
  echo "[$(date -Is)] $slug: serving $HF port $port maxtok $MAXTOK extra ${EXTRA[*]} body $BODY"
  vllm serve "$HF" --served-model-name "$slug" --host 127.0.0.1 --port "$port" \
    --max-model-len 8192 --gpu-memory-utilization "$UTIL" --generation-config vllm --seed 0 \
    --max-num-seqs 128 --disable-uvicorn-access-log "${EXTRA[@]}" > "$vlog" 2>&1 &
  SERVER_PID=$!
  local t0; t0=$(date +%s)
  until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[$(date -Is)] $slug: FAIL server exited during startup"; tail -30 "$vlog"; SERVER_PID=""; return 2
    fi
    if (( $(date +%s) - t0 > 3600 )); then
      echo "[$(date -Is)] $slug: FAIL server not ready after 3600 s"; tail -30 "$vlog"; return 2
    fi
    sleep 15
  done
  echo "[$(date -Is)] $slug: server ready after $(( $(date +%s) - t0 )) s; weights $(du -sh "$HF_HUB_CACHE" | cut -f1)"
  python "$ROOT/cluster/fs_client.py" --prompts "$PROMPTS" --out "$OUTDIR/$slug.jsonl" \
    --served-name "$slug" --model-id "$HF" --max-tokens "$MAXTOK" --concurrency 64 --port "$port" \
    --limit "$LIMIT" --extra-body "$BODY"
  local rc=$?
  python "$ROOT/cluster/fs_check.py" "$OUTDIR/$slug.jsonl" "$MODE"
  local rc2=$?
  kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""
  rm -rf "$HF_HUB_CACHE"
  echo "[$(date -Is)] $slug: client rc $rc check rc $rc2"
  if [ "$rc" -ne 0 ]; then return "$rc"; fi
  return "$rc2"
}

RC=0
for slug in "$@"; do
  run_one "$slug" || RC=$?
done
echo "job end $(date -Is) rc $RC"
exit "$RC"
