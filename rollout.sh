#!/bin/bash
# rollout.sh — quick smoke-test every task YAML under config/
# Usage: ./rollout.sh [--config-dir base|complex|both]

set -uo pipefail

REPO=/root/autodl-tmp/MineAgent
CONFIG_BASE=$REPO/config
LOG=$REPO/rollout.log

base_url=http://localhost:8000/v1
workers=4
max_frames=10          # small number for quick env-init check
temperature=0.9
history_num=2
action_chunk_len=1
instruction_type="normal"
model_local_path="$REPO/models/jarvis_vla_qwen2_vl_7b_sft"

# Parse --config-dir argument (default: both)
config_dir="both"
for arg in "$@"; do
    case $arg in
        --config-dir=*) config_dir="${arg#*=}" ;;
        --config-dir)   shift; config_dir="$1" ;;
    esac
done

cd "$REPO"
export PYTHONPATH=/root/autodl-tmp/JarvisVLA:${PYTHONPATH:-}
export OMP_NUM_THREADS=1

echo "[$(date)] rollout smoke-test  config_dir=$config_dir  max_frames=$max_frames" | tee "$LOG"

# Collect task configs: build/{category}/{stem}
tasks=()
for dir in base complex; do
    [[ "$config_dir" != "both" && "$config_dir" != "$dir" ]] && continue
    for yaml in "$CONFIG_BASE/$dir"/craft/*.yaml "$CONFIG_BASE/$dir"/kill/*.yaml "$CONFIG_BASE/$dir"/mine/*.yaml "$CONFIG_BASE/$dir"/smelt/*.yaml; do
        [[ -f "$yaml" ]] || continue
        stem=$(basename "$yaml" .yaml)
        [[ "$stem" == "base" ]] && continue
        category=$(basename "$(dirname "$yaml")")
        tasks+=("$dir/$category/$stem")
    done
done

total=${#tasks[@]}
passed=0
failed=0
fail_list=()

echo "Found $total tasks." | tee -a "$LOG"
echo "" | tee -a "$LOG"

for task in "${tasks[@]}"; do
    echo "[$(date)] === $task ===" | tee -a "$LOG"

    conda run -n minestudio python "$REPO/scripts/eval/evaluate.py" \
        --workers      $workers \
        --split-number $workers \
        --env-config   "$task" \
        --max-frames   $max_frames \
        --temperature  $temperature \
        --checkpoints  "$model_local_path" \
        --video-main-fold "logs/" \
        --base-url     "$base_url" \
        --history-num  $history_num \
        --instruction-type $instruction_type \
        --action-chunk-len $action_chunk_len \
        --config-base  "$CONFIG_BASE" \
        2>&1 | tee -a "$LOG"

    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        echo "  PASS: $task" | tee -a "$LOG"
        ((passed++))
    else
        echo "  FAIL: $task" | tee -a "$LOG"
        ((failed++))
        fail_list+=("$task")
    fi
    echo "" | tee -a "$LOG"
done

echo "========================================" | tee -a "$LOG"
echo "Results: $passed/$total passed" | tee -a "$LOG"
if [[ ${#fail_list[@]} -gt 0 ]]; then
    echo "Failed tasks:" | tee -a "$LOG"
    for t in "${fail_list[@]}"; do
        echo "  FAIL  $t" | tee -a "$LOG"
    done
fi
echo "[$(date)] Done." | tee -a "$LOG"
