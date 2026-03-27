#!/bin/bash
# Single-node training for Klein-base-4B with Flow-GRPO
# Usage: bash scripts/single_node/grpo_klein.sh

accelerate launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes=4 \
    --main_process_port 29503 \
    scripts/train_klein.py \
    --config config/grpo.py:pickscore_klein_base_4b
