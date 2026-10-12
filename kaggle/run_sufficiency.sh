#!/bin/zsh
# Push the evidence-sufficiency task, then run it on the Kaggle panel models already run on the core tasks.
cd "$(dirname "$0")"
source ../.venv/bin/activate
LOG=../results/kaggle_logs; MIN_REMAINING=0.35
remaining() { kaggle b quota 2>/dev/null | awk '/^Daily/ {gsub("\\$","",$3); print $3}'; }
echo "$(date -u +%H:%M:%S) push sufficiency remaining=$(remaining)" | tee -a $LOG/driver_suff.log
kaggle b t push fairscreen-evidence-sufficiency -f tasks/fairscreen-evidence-sufficiency.py --wait > $LOG/push_suff.log 2>&1; echo "push exit $?" >> $LOG/driver_suff.log
for m in gpt-5.4-nano-2026-03-17 gemini-3.1-flash-lite-preview gemma-4-31b-it gpt-oss-120b; do
  sleep 20; r=$(remaining); echo "$(date -u +%H:%M:%S) model $m remaining=$r" | tee -a $LOG/driver_suff.log
  if (( $(echo "$r < $MIN_REMAINING" | bc -l) )); then echo "STOP" | tee -a $LOG/driver_suff.log; break; fi
  kaggle b t run fairscreen-evidence-sufficiency -m $m --wait > $LOG/run_suff_$m.log 2>&1; echo "run $m exit $?" >> $LOG/driver_suff.log
done
echo "$(date -u +%H:%M:%S) DONE remaining=$(remaining)" | tee -a $LOG/driver_suff.log
