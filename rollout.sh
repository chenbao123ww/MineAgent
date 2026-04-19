#!/bin/bash

base_url=http://localhost:8000/v1
workers=5
max_frames=300
temperature=0.9
history_num=2
action_chunk_len=1
instruction_type="normal"
model_local_path="/root/autodl-tmp/MineAgent/models/jarvis_vla_qwen2_vl_7b_sft"

tasks=(
    "kill/kill_zombie"
    "kill/kill_skeleton"
    "mine/chop_oak_log"
    "mine/mine_iron_ore"
    "mine/mine_stone"
)

echo "Running evaluation with model $model_local_path..."

# Run from MineAgent root so hydra finds ./config/
cd /root/autodl-tmp/MineAgent
export PYTHONPATH=/root/autodl-tmp/JarvisVLA:$PYTHONPATH
export OMP_NUM_THREADS=1

for task in "${tasks[@]}"; do
    env_config=$task

    num_iterations=$(($workers / 5 + 1))
    for ((i = 0; i < num_iterations; i++)); do
        python /root/autodl-tmp/JarvisVLA/jarvisvla/evaluate/evaluate.py \
            --workers $workers \
            --env-config $env_config \
            --max-frames $max_frames \
            --temperature $temperature \
            --checkpoints $model_local_path \
            --video-main-fold "logs/" \
            --base-url "$base_url" \
            --history-num $history_num \
            --instruction-type $instruction_type \
            --action-chunk-len $action_chunk_len \
            --config-base /root/autodl-tmp/MineAgent/config
        if [[ $? -eq 0 ]]; then
            echo "[$task] 第 $i 次迭代执行成功，退出循环。"
            break
        fi
        if [[ $i -lt $((num_iterations - 1)) ]]; then
            echo "等待 10 秒..."
            sleep 10
        fi
    done
done
