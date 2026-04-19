#!/bin/bash

cuda_visible_devices=0
card_num=1
model_name_or_path="/root/autodl-tmp/MineAgent/models/jarvis_vla_qwen2_vl_7b_sft"

CUDA_VISIBLE_DEVICES=$cuda_visible_devices OMP_NUM_THREADS=4 \
    vllm serve $model_name_or_path \
    --port 8000 \
    --max-model-len 8448 \
    --max-num-seqs 10 \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size $card_num \
    --trust-remote-code \
    --served_model_name "jarvisvla" \
    --limit-mm-per-prompt '{"image": 5}'
