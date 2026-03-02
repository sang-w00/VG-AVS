#!/usr/bin/env bash


# ===== Project Root =====
# Change this path to your project root directory
PROJECT_ROOT=${PROJECT_ROOT:-/home/daehyeonchoi/code/eccv2026/AVS_eccv}

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

IMG_ROOT=${IMG_ROOT:-${PROJECT_ROOT}/data}
NUM_SAMPLES=${NUM_SAMPLES:--1} # use all samples

# Verifier settings
#CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
VERIFIER_MODEL=${VERIFIER_MODEL:-gemini-2.5-flash}
VERIFIER_DEVICE=${VERIFIER_DEVICE:-cuda:0}
GPU_DEVICE=${GPU_DEVICE:-0}
STUDENT_DEVICE=${STUDENT_DEVICE:-cuda:0}

# API Keys
export GEMINI_API_KEY=""
#export OPENAI_API_KEY="<OPENAI_API_KEY>"  # Uncomment and set when using GPT

echo "Testing ProcTHOR action prediction model..."
echo "Model: ${MODEL_PATH}"
echo "Image root: ${IMG_ROOT}"
echo "GPU device for rendering: ${GPU_DEVICE}"
echo ""

# existence
EXISTENCE_TEST_JSONL=${PROJECT_ROOT}/data/avs_existence_eval.jsonl
COUNTING_TEST_JSONL=${PROJECT_ROOT}/data/avs_counting_eval.jsonl
STATE_TEST_JSONL=${PROJECT_ROOT}/data/avs_state_eval.jsonl

MODEL_NAME=gemini25_pro
OUTPUT_DIR=${PROJECT_ROOT}/src/open-r1-multimodal/output/visualization/${MODEL_NAME}/existence_eval/$(date +%Y%m%d_%H%M%S)

export PYTHONPATH=${PYTHONPATH:-}:${PROJECT_ROOT}/src/open-r1-multimodal/src

python ${PROJECT_ROOT}/src/open-r1-multimodal/src/open_r1/test_procthor_multistep_action_accuracy.py \
  --model_path "${MODEL_PATH}" \
  --test_jsonl "${EXISTENCE_TEST_JSONL}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_samples ${NUM_SAMPLES} \
  --verifier_model ${VERIFIER_MODEL} \
  --max_new_tokens 256 \
  --verifier_max_tokens 48 \
  --device ${STUDENT_DEVICE} \
  --verifier_device ${VERIFIER_DEVICE} \
  --gpu_device ${GPU_DEVICE} \
  --use_gemini_verifier \
  --image_root "${IMG_ROOT}" \
  --max_rollout_steps 4 \
  --use_gemini_action \


# counting

OUTPUT_DIR=${PROJECT_ROOT}/src/open-r1-multimodal/output/visualization/${MODEL_NAME}/counting_eval/$(date +%Y%m%d_%H%M%S)

python ${PROJECT_ROOT}/src/open-r1-multimodal/src/open_r1/test_procthor_multistep_action_accuracy.py \
  --model_path "${MODEL_PATH}" \
  --test_jsonl "${COUNTING_TEST_JSONL}" \
  --image_root "${IMG_ROOT}" \
  --max_rollout_steps 4 \
  --output_dir "${OUTPUT_DIR}" \
  --num_samples ${NUM_SAMPLES} \
  --verifier_model ${VERIFIER_MODEL} \
  --max_new_tokens 256 \
  --verifier_max_tokens 48 \
  --device ${STUDENT_DEVICE} \
  --verifier_device ${VERIFIER_DEVICE} \
  --gpu_device ${GPU_DEVICE} \
  --use_gemini_verifier \
  --custom_house_path ${PROJECT_ROOT}/data/counting_house_data.json \
  --use_gemini_action \




# STATE TEST

OUTPUT_DIR=${PROJECT_ROOT}/src/open-r1-multimodal/output/visualization/${MODEL_NAME}/state_eval/$(date +%Y%m%d_%H%M%S)

python ${PROJECT_ROOT}/src/open-r1-multimodal/src/open_r1/test_procthor_multistep_action_accuracy.py \
  --model_path "${MODEL_PATH}" \
  --test_jsonl "${STATE_TEST_JSONL}" \
  --image_root "${IMG_ROOT}" \
  --max_rollout_steps 4 \
  --output_dir "${OUTPUT_DIR}" \
  --num_samples ${NUM_SAMPLES} \
  --verifier_model ${VERIFIER_MODEL} \
  --max_new_tokens 256 \
  --verifier_max_tokens 48 \
  --device ${STUDENT_DEVICE} \
  --verifier_device ${VERIFIER_DEVICE} \
  --gpu_device ${GPU_DEVICE} \
  --use_gemini_verifier \
  --use_gemini_action \

echo ""
echo "Testing complete! Check results at: ${OUTPUT_DIR}"