#!/bin/bash
# run_base_tasks.sh — record one trajectory per base task

set -euo pipefail

REPO=/root/autodl-tmp/MineAgent
CONFIG_BASE=$REPO/config
LOG=$REPO/run_base_tasks.log

cd "$REPO"

echo "[$(date)] Starting base task runs" | tee -a "$LOG"

for yaml in "$CONFIG_BASE"/base/{kill,smelt}/*.yaml; do
    fname=$(basename "$yaml" .yaml)
    [[ "$fname" == "base" ]] && continue

    category=$(basename "$(dirname "$yaml")")
    env_config="base/$category/$fname"

    echo "" | tee -a "$LOG"
    echo "[$(date)] === $env_config ===" | tee -a "$LOG"

    conda run -n minestudio python3 "$REPO/scripts/data_process/record.py" \
        --env-config  "$env_config" \
        --config-base "$CONFIG_BASE" \
        --max-frames  1000 \
        --base-url    http://localhost:8000/v1 \
        --save-frames \
        2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "[$(date)] All base tasks finished." | tee -a "$LOG"
