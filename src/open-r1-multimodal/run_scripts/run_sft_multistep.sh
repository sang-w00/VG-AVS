#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC=${NPROC:-8}
MASTER_PORT=${MASTER_PORT:-12337}
# ===== dataset ====== #
PROJECT_ROOT=${PROJECT_ROOT:-/path/to/your/project}
export PYTHONPATH=${PYTHONPATH:-}:${PROJECT_ROOT}/src/open-r1-multimodal/src
DATA_JSONL=${DATA_JSONL:-/home/andy2884/workspace/VG-AVS/data/avs_existence_train_final_0228.jsonl}
IMG_ROOT=${IMG_ROOT:-/path/to/dataset}
cd src/open-r1-multimodal
RUN_NAME=${RUN_NAME:-sft-multistep-cot-3b-0228-20epoch-undecidable}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}
# MODEL=${MODEL:-/home/andy2884/workspace/VG-AVS/src/open-r1-multimodal/output/sft-multistep-cot-3b-0228-20epoch-undecidable}
# Wandb settings
export WANDB_PROJECT=${WANDB_PROJECT:-vgavs}
REPORT_TO=${REPORT_TO:-wandb}
# Torch/distributed settings
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
# Validation split
VAL_SPLIT_RATIO=${VAL_SPLIT_RATIO:-0}
VAL_SPLIT_SEED=${VAL_SPLIT_SEED:-42}
# Model settings
MAX_PIXELS=${MAX_PIXELS:-12845056}
MIN_PIXELS=${MIN_PIXELS:-3136}
# Training hyperparameters
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-2.0e-5}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-20}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-1}
# LoRA settings
USE_PEFT=${USE_PEFT:-false}
LORA_R=${LORA_R:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
# Check required environment variables
if [[ -z "${DATA_JSONL:-}" ]]; then
  echo "ERROR: Please set DATA_JSONL to your dataset JSONL file." >&2
  exit 1
fi
if [[ -z "${IMG_ROOT:-}" ]]; then
  echo "ERROR: Please set IMG_ROOT to your image root directory." >&2
  exit 1
fi
echo "==========================================="
echo "Multi-Step SFT Training: ProcTHOR Active VQA"
echo "==========================================="
echo "Run name: ${RUN_NAME}"
echo "Model: ${MODEL}"
echo "Data: ${DATA_JSONL}"
echo "Image root: ${IMG_ROOT}"
echo "Validation split: ${VAL_SPLIT_RATIO}"
echo "Output dir: output/${RUN_NAME}"
echo "==========================================="
torchrun --nproc_per_node="${NPROC}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  src/open_r1/sft_multistep.py \
  --model_name_or_path ${MODEL} \
  --data_file_paths "${DATA_JSONL}" \
  --dataset_name None \
  --image_folders "${IMG_ROOT}" \
  --val_split_ratio ${VAL_SPLIT_RATIO} \
  --val_split_seed ${VAL_SPLIT_SEED} \
  --max_pixels ${MAX_PIXELS} \
  --min_pixels ${MIN_PIXELS} \
  --output_dir output/${RUN_NAME} \
  --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --learning_rate ${LEARNING_RATE} \
  --num_train_epochs ${NUM_TRAIN_EPOCHS} \
  --logging_steps ${LOGGING_STEPS} \
  --save_strategy epoch \
  --save_total_limit ${SAVE_TOTAL_LIMIT} \
  --eval_strategy no \
  --bf16 \
  --torch_dtype bfloat16 \
  --gradient_checkpointing true \
  --attn_implementation flash_attention_2 \
  --report_to ${REPORT_TO} \
  --run_name ${RUN_NAME} \
  --trust_remote_code true \
  --remove_unused_columns false \

echo ""
echo "==========================================="
echo "Multi-Step SFT Training completed!"
echo "Model saved to: output/${RUN_NAME}"
echo "==========================================="
