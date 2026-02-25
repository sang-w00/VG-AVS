#!/usr/bin/env bash
set -euo pipefail

cd src/open-r1-multimodal

# ===== Project Root =====
# Change this path to your project root directory
PROJECT_ROOT=${PROJECT_ROOT:-/home/andy2884/workspace/VG-AVS}

MODEL_PATH=${MODEL_PATH:-/home/andy2884/workspace/VG-AVS/model/grpo-procthor-sft20-refine-internvl-0125-3epoch}

IMG_ROOT=${IMG_ROOT:-/home/andy2884/workspace/VG-AVS/data}
NUM_SAMPLES=${NUM_SAMPLES:--1} # use all samples

# Verifier settings
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
VERIFIER_MODEL=${VERIFIER_MODEL:-gemini-2.5-flash}
VERIFIER_DEVICE=${VERIFIER_DEVICE:-cuda:9}
GPU_DEVICE=${GPU_DEVICE:-9}
STUDENT_DEVICE=${STUDENT_DEVICE:-cuda:9}

# API Keys
export GEMINI_API_KEY="AIzaSyD0CIn8kCOzJniUvCMjOoJkVLtCHxgy81k"
#export OPENAI_API_KEY="<OPENAI_API_KEY>"  # Uncomment and set when using GPT

echo "Testing ProcTHOR action prediction model..."
echo "Model: ${MODEL_PATH}"
echo "Image root: ${IMG_ROOT}"
echo "GPU device for rendering: ${GPU_DEVICE}"
echo ""

# existence
EXISTENCE_TEST_JSONL=${PROJECT_ROOT}/data/avs_procthor_existence.jsonl
COUNTING_TEST_JSONL=${PROJECT_ROOT}/data/avs_procthor_counting.jsonl
STATE_TEST_JSONL=${PROJECT_ROOT}/data/avs_procthor_state.jsonl

MODEL_NAME=$(basename ${MODEL_PATH})
OUTPUT_DIR=./output/${MODEL_NAME}/existence_eval/$(date +%Y%m%d_%H%M%S)

export PYTHONPATH=${PYTHONPATH:-}:${PROJECT_ROOT}/src/open-r1-multimodal/src

# python src/open_r1/test_procthor_action_accuracy.py \
#   --model_path "${MODEL_PATH}" \
#   --test_jsonl "${EXISTENCE_TEST_JSONL}" \
#   --output_dir "${OUTPUT_DIR}" \
#   --num_samples ${NUM_SAMPLES} \
#   --verifier_model ${VERIFIER_MODEL} \
#   --max_new_tokens 256 \
#   --verifier_max_tokens 48 \
#   --device ${STUDENT_DEVICE} \
#   --verifier_device ${VERIFIER_DEVICE} \
#   --gpu_device ${GPU_DEVICE} \
#   --use_gemini_verifier \
#   --image_root "${IMG_ROOT}" \
#   --use_refine


# counting

OUTPUT_DIR=./output/${MODEL_NAME}/counting_eval/$(date +%Y%m%d_%H%M%S)
python src/open_r1/test_procthor_action_accuracy.py \
  --model_path "${MODEL_PATH}" \
  --test_jsonl "${COUNTING_TEST_JSONL}" \
  --image_root "${IMG_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_samples ${NUM_SAMPLES} \
  --verifier_model ${VERIFIER_MODEL} \
  --max_new_tokens 256 \
  --verifier_max_tokens 48 \
  --device ${STUDENT_DEVICE} \
  --verifier_device ${VERIFIER_DEVICE} \
  --gpu_device ${GPU_DEVICE} \
  --use_gemini_verifier \
  --custom_house_path ${PROJECT_ROOT}/data/val_count_1106.jsonl.gz \
  --use_refine



# STATE TEST

# OUTPUT_DIR=./output/${MODEL_NAME}/state_eval/$(date +%Y%m%d_%H%M%S)
# python src/open_r1/test_procthor_action_accuracy.py \
#   --model_path "${MODEL_PATH}" \
#   --test_jsonl "${STATE_TEST_JSONL}" \
#   --image_root "${IMG_ROOT}" \
#   --output_dir "${OUTPUT_DIR}" \
#   --num_samples ${NUM_SAMPLES} \
#   --verifier_model ${VERIFIER_MODEL} \
#   --max_new_tokens 256 \
#   --verifier_max_tokens 48 \
#   --device ${STUDENT_DEVICE} \
#   --verifier_device ${VERIFIER_DEVICE} \
#   --gpu_device ${GPU_DEVICE} \
#   --use_gemini_verifier \
#   --use_refine 



echo ""
echo "Testing complete! Check results at: ${OUTPUT_DIR}"