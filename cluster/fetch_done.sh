#!/bin/bash
# Copy one finished model's output from the cluster to the Mac, gzipped.
# usage: fetch_done.sh <slug>   (refuses unless the cluster file has all 11,328 unique ids)
set -euo pipefail
SLUG=$1
EXPECTED=11328
DEST=$HOME/Developer/fairscreen/results/cluster_raw
mkdir -p "$DEST"
read -r LINES UNIQ < <(ssh soc "python3 -c \"
import json
ids=[json.loads(l)['id'] for l in open('/home/e/e1506804/fairscreen/out_v11/$SLUG.jsonl') if l.strip()]
print(len(ids), len(set(ids)))
\"")
echo "$SLUG: lines $LINES unique $UNIQ"
if [ "$UNIQ" -ne "$EXPECTED" ]; then echo "incomplete, not copying"; exit 3; fi
ssh soc "cd ~/fairscreen/out_v11 && gzip -9 -c $SLUG.jsonl > $SLUG.jsonl.gz && md5sum $SLUG.jsonl.gz"
scp -q "soc:fairscreen/out_v11/$SLUG.jsonl.gz" "$DEST/$SLUG.jsonl.gz"
md5 -q "$DEST/$SLUG.jsonl.gz"
echo "local lines: $(gzip -dc "$DEST/$SLUG.jsonl.gz" | wc -l | tr -d ' ')  size: $(du -h "$DEST/$SLUG.jsonl.gz" | cut -f1)"
