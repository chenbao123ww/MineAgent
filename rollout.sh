#!/bin/bash
# rollout.sh — evaluate every task YAML under config/
# Usage: ./rollout.sh [--config-dir base|complex|both|test] [--model <name>]

set -uo pipefail

REPO=/root/autodl-tmp/MineAgent
CONFIG_BASE=$REPO/config
LOG=$REPO/rollout.log

base_url=http://localhost:8000/v1
workers=13
split_number=1
temperature=0.9
history_num=2
action_chunk_len=1
model_local_path="$REPO/models/"

# Parse arguments
config_dir="both"
categories="craft kill mine smelt"
for arg in "$@"; do
    case $arg in
        --config-dir=*) config_dir="${arg#*=}" ;;
        --config-dir)   shift; config_dir="$1" ;;
        --model=*)      model_local_path="$REPO/models/${arg#*=}" ;;
        --model)        shift; model_local_path="$REPO/models/$1" ;;
        --categories=*) categories="${arg#*=}" ;;
        --categories)   shift; categories="$1" ;;
    esac
done

cd "$REPO"
export PYTHONPATH=/root/autodl-tmp/JarvisVLA:${PYTHONPATH:-}
export OMP_NUM_THREADS=1

echo "[$(date)] rollout eval  config_dir=$config_dir  model=$(basename $model_local_path)" | tee "$LOG"

# Collect task configs
tasks=()
if [[ "$config_dir" == "test" ]]; then
    for yaml in $(for cat in $categories; do echo "$CONFIG_BASE/test/$cat/*.yaml"; done); do
        [[ -f "$yaml" ]] || continue
        stem=$(basename "$yaml" .yaml)
        [[ "$stem" == "base" ]] && continue
        category=$(basename "$(dirname "$yaml")")
        tasks+=("test/$category/$stem")
    done
else
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
fi

total=${#tasks[@]}
passed=0
failed=0
fail_list=()

echo "Found $total tasks." | tee -a "$LOG"
echo "" | tee -a "$LOG"

for task in "${tasks[@]}"; do
    echo "[$(date)] === $task ===" | tee -a "$LOG"

    category=$(echo "$task" | awk -F'/' '{print $(NF-1)}')
    if [[ "$category" == "craft" || "$category" == "smelt" ]]; then
        cur_instruction_type="recipe"
        cur_max_frames=400ha
    elif [[ "$category" == "mine" ]]; then
        cur_instruction_type="normal"
        cur_max_frames=500
    else
        cur_instruction_type="normal"
        cur_max_frames=300
    fi
    if [[ "$category" == "mine" ]]; then
        cur_temperature=0.6
    else
        cur_temperature=$temperature
    fi

    conda run -n minestudio python "$REPO/scripts/eval/evaluate.py" \
        --workers      $workers \
        --split-number $split_number \
        --env-config   "$task" \
        --max-frames   $cur_max_frames \
        --temperature  $cur_temperature \
        --checkpoints  "$model_local_path" \
        --video-main-fold "$REPO/logs/" \
        --base-url     "$base_url" \
        --history-num  $history_num \
        --instruction-type $cur_instruction_type \
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
echo "Results: $passed/$total tasks launched" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ---- success rate summary ----
python3 - "$REPO/logs" "$workers" <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, os, sys, glob

logs_dir = sys.argv[1]
n_episodes = int(sys.argv[2])

cat_map = {}   # category -> list of (task, rate)
task_rows = []

for end_json in sorted(glob.glob(os.path.join(logs_dir, "*", "end.json"))):
    folder = os.path.basename(os.path.dirname(end_json))
    # folder: {model}-{task_name}  e.g. jarvis_vla_qwen2_vl_7b_sft-craft_wooden_pickaxe
    task_name = folder.split("-", 1)[-1] if "-" in folder else folder
    # infer category from task name prefix
    category = task_name.split("_")[0] if "_" in task_name else "unknown"

    try:
        results = json.load(open(end_json))
        successes = sum(1 for r in results if r[0])
        total = len(results)
        rate = successes / total if total else 0.0
    except Exception as e:
        rate = float("nan")
        successes, total = 0, 0

    task_rows.append((category, task_name, successes, total, rate))
    cat_map.setdefault(category, []).append(rate)

if not task_rows:
    print("No end.json files found — tasks may still be running.")
    sys.exit(0)

# per-task table
print("=" * 55)
print(f"{'Category':<10} {'Task':<35} {'Success Rate'}")
print("-" * 55)
for category, task_name, suc, tot, rate in task_rows:
    print(f"{category:<10} {task_name:<35} {suc}/{tot} ({rate*100:.1f}%)")

# per-category summary
print("=" * 55)
print("Category averages:")
all_rates = []
for cat in ["craft", "kill", "mine", "smelt"]:
    rates = cat_map.get(cat, [])
    if rates:
        avg = sum(rates) / len(rates)
        all_rates.extend(rates)
        print(f"  {cat:<8} {avg*100:.1f}%")
    else:
        print(f"  {cat:<8} N/A")
if all_rates:
    overall = sum(all_rates) / len(all_rates)
    print(f"  {'overall':<8} {overall*100:.1f}%")
print("=" * 55)
PYEOF

echo "[$(date)] Done." | tee -a "$LOG"
