#!/bin/zsh
# Push the three FairScreen-48 tasks, then run the registered Kaggle panel cheapest first.
# Stops before a model when the settled Model Proxy quota remaining is below MIN_REMAINING dollars.
cd "$(dirname "$0")"
source ../.venv/bin/activate
LOG=../results/kaggle_logs; mkdir -p $LOG
MIN_REMAINING=1.5
TASKS=(fairscreen-direct fairscreen-fairness-instruction fairscreen-structured)
MODELS=(gpt-5.4-nano-2026-03-17 gemini-3.1-flash-lite-preview gpt-oss-120b gemma-4-31b-it qwen3-235b-a22b-instruct-2507 gpt-5.4-mini-2026-03-17 claude-haiku-4-5-20251001)
remaining() { kaggle b quota 2>/dev/null | awk '/^Daily/ {gsub("\\$","",$3); print $3}'; }
for t in $TASKS; do
  echo "$(date -u +%H:%M:%S) push $t remaining=$(remaining)" | tee -a $LOG/driver.log
  kaggle b t push $t -f tasks/$t.py --wait > $LOG/push_$t.log 2>&1; echo "push $t exit $?" >> $LOG/driver.log
done
for m in $MODELS; do
  sleep 30
  r=$(remaining)
  echo "$(date -u +%H:%M:%S) model $m remaining=$r" | tee -a $LOG/driver.log
  if (( $(echo "$r < $MIN_REMAINING" | bc -l) )); then echo "STOP: quota below margin before $m" | tee -a $LOG/driver.log; break; fi
  for t in $TASKS; do
    (kaggle b t run $t -m $m --wait > $LOG/run_${t}_${m}.log 2>&1; echo "run $t $m exit $?" >> $LOG/driver.log) &
  done
  wait
done
echo "$(date -u +%H:%M:%S) DONE remaining=$(remaining)" | tee -a $LOG/driver.log
