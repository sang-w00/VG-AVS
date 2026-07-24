#!/usr/bin/env bash
# set -euo pipefail

# Check Python environment
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
DATA_JSONL=${DATA_JSONL:-/home/andy2884/workspace/VG-AVS/data/avs_existence_train_final_0302_005.jsonl}
IMG_ROOT=${IMG_ROOT:-/path/to/dataset}

cd src/open-r1-multimodal

# Configuration
BASE_MODEL=${BASE_MODEL:-/home/andy2884/workspace/VG-AVS/src/open-r1-multimodal/output/sft-multistep-cot-3b-0302-005-20epoch}
# BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}
GRPO_RUN_NAME=${GRPO_RUN_NAME:-grpo-multistep-3b-0302-005}
MAX_ROLLOUT_STEPS=${MAX_ROLLOUT_STEPS:-4}
SAVE_ROLLOUT_VIS=${SAVE_ROLLOUT_VIS:-true}
ROLLOUT_VIS_INTERVAL=${ROLLOUT_VIS_INTERVAL:-1}
ROLLOUT_VIS_NUM_SAMPLES=${ROLLOUT_VIS_NUM_SAMPLES:-1}

# Torch/distributed settings
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

if [[ -z "${DATA_JSONL:-}" ]]; then
  echo "ERROR: Please set DATA_JSONL to your dataset JSONL file." >&2
  exit 1
fi

if [[ -z "${IMG_ROOT:-}" ]]; then
  echo "ERROR: Please set IMG_ROOT to your image root directory." >&2
  exit 1
fi

GRPO_OUTPUT_DIR="output/${GRPO_RUN_NAME}"

export WANDB_PROJECT=${WANDB_PROJECT:-vgavs}
REPORT_TO=${REPORT_TO:-wandb}

echo ""
echo "========================================="
echo "Multi-Step GRPO: Procthor Active VQA"
echo "========================================="
echo "Model: ${BASE_MODEL}"
echo "Max rollout steps: ${MAX_ROLLOUT_STEPS}"
echo "Rollout visualization: ${SAVE_ROLLOUT_VIS} (interval=${ROLLOUT_VIS_INTERVAL}, samples=${ROLLOUT_VIS_NUM_SAMPLES})"
echo "Verifier: ${VERIFIER_MODEL_PATH:-qwen2.5vl:7b} (max_new_tokens=${VERIFIER_MAX_NEW_TOKENS:-16}, minibatch=${VERIFIER_MINIBATCH:-4})"
echo "Output: ${GRPO_OUTPUT_DIR}"
echo ""

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export DEBUG_MODE=${DEBUG_MODE:-1}
export LOG_PATH="./debug_log_${GRPO_RUN_NAME}.txt"
export VERIFIER_MAX_NEW_TOKENS=${VERIFIER_MAX_NEW_TOKENS:-16}
export VERIFIER_MINIBATCH=${VERIFIER_MINIBATCH:-2}
export VERIFIER_MODEL_PATH=${VERIFIER_MODEL_PATH:-qwen2.5vl:7b}

TORCHRUN="$(dirname $(which python))/torchrun"
echo "Using torchrun: ${TORCHRUN}"

${TORCHRUN} --nproc_per_node="8" \
  --nnodes="1" \
  --node_rank="0" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="12348" \
  src/open_r1/grpo_multistep.py \
  --deepspeed local_scripts/zero2.json \
  --output_dir ${GRPO_OUTPUT_DIR} \
  --model_name_or_path ${BASE_MODEL} \
  --data_file_paths "${DATA_JSONL}" \
  ${IMG_ROOT:+--image_folders "${IMG_ROOT}"} \
  --dataset_name None \
  --reward_funcs multistep_verifier multistep_format \
  --grpo_reward_weights 1 0.3 \
  --max_rollout_steps ${MAX_ROLLOUT_STEPS} \
  --save_rollout_visualizations ${SAVE_ROLLOUT_VIS} \
  --rollout_vis_interval ${ROLLOUT_VIS_INTERVAL} \
  --rollout_vis_num_samples ${ROLLOUT_VIS_NUM_SAMPLES} \
  --rollout_vis_dir ${GRPO_OUTPUT_DIR}/rollout_visualizations \
  --num_generations 16 \
  --beta 0.04 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --logging_steps 1 \
  --freeze_vision_modules true \
  --bf16 \
  --torch_dtype bfloat16 \
  --data_seed 42 \
  --report_to ${REPORT_TO} \
  --gradient_checkpointing true \
  --attn_implementation flash_attention_2 \
  --num_train_epochs 5 \
  --run_name ${GRPO_RUN_NAME} \
  --save_strategy epoch \
  --save_only_model false \
  --learning_rate 1e-6 \
  --num_iterations 1 \
  --max_completion_length 512 \

echo ""
echo "========================================="
echo "Multi-Step GRPO Training Complete!"
echo "========================================="
echo ""
