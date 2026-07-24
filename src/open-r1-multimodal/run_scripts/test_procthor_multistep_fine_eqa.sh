#!/usr/bin/env bash

# ===== Project Root =====
# Change this path to your project root directory
PROJECT_ROOT=${PROJECT_ROOT:-/home/andy2884/workspace/VG-AVS}

# The Fine-EQA method uses an arbitrary foundation VLM (defaulting to the same model structure if possible, 
# or a specific vision-language model for action evaluation)
# MODEL_PATH=${MODEL_PATH:-${PROJECT_ROOT}/src/open-r1-multimodal/output/sft-multistep-cot-3b-0228-20epoch}

IMG_ROOT=${IMG_ROOT:-${PROJECT_ROOT}/data}
NUM_SAMPLES=${NUM_SAMPLES:--1} # use all samples

# Verifier settings
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
export CUDA_VISIBLE_DEVICES
VERIFIER_MODEL=${VERIFIER_MODEL:-gemini-2.5-flash}
VERIFIER_DEVICE=${VERIFIER_DEVICE:-cuda:0}
# Default rendering GPU to the first visible physical GPU index.
GPU_DEVICE=${GPU_DEVICE:-${CUDA_VISIBLE_DEVICES%%,*}}
STUDENT_DEVICE=${STUDENT_DEVICE:-cuda:0}


echo "Testing ProcTHOR Fine-EQA multi-step action prediction model..."
# echo "Model: ${MODEL_PATH}"
echo "Image root: ${IMG_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "GPU device for rendering: ${GPU_DEVICE}"
echo ""

# existence
EXISTENCE_TEST_JSONL=${PROJECT_ROOT}/data/avs_existence_eval.jsonl
COUNTING_TEST_JSONL=${PROJECT_ROOT}/data/avs_counting_eval.jsonl
STATE_TEST_JSONL=${PROJECT_ROOT}/data/avs_state_eval.jsonl

# MODEL_NAME=$(basename ${MODEL_PATH})
OUTPUT_DIR=${PROJECT_ROOT}/src/open-r1-multimodal/output/visualization/fine_eqa_${MODEL_NAME}/existence_eval/$(date +%Y%m%d_%H%M%S)

export PYTHONPATH=${PYTHONPATH:-}:${PROJECT_ROOT}/src/open-r1-multimodal/src

# python ${PROJECT_ROOT}/src/open-r1-multimodal/src/open_r1/test_procthor_multistep_fine_eqa.py \
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
#   --use_prismatic \
#   --image_root "${IMG_ROOT}" \
#   --max_rollout_steps 4 \


# counting

OUTPUT_DIR=./output/fine_eqa_${MODEL_NAME}/counting_eval/$(date +%Y%m%d_%H%M%S)
python src/open-r1-multimodal/src/open_r1/test_procthor_multistep_fine_eqa.py \
  --use_prismatic \
  --test_jsonl "${COUNTING_TEST_JSONL}" \
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
  --custom_house_path /home/daehyeonchoi/code/eccv2026/AVS_eccv/data/counting_house_data.json \




# STATE TEST

# OUTPUT_DIR=./output/fine_eqa_${MODEL_NAME}/state_eval/$(date +%Y%m%d_%H%M%S)
# python src/open-r1-multimodal/src/open_r1/test_procthor_multistep_fine_eqa.py \
#   --use_prismatic \
#   --test_jsonl "${STATE_TEST_JSONL}" \
#   --max_rollout_steps 4 \
#   --output_dir "${OUTPUT_DIR}" \
#   --num_samples ${NUM_SAMPLES} \
#   --verifier_model ${VERIFIER_MODEL} \
#   --max_new_tokens 256 \
#   --verifier_max_tokens 48 \
#   --device ${STUDENT_DEVICE} \
#   --verifier_device ${VERIFIER_DEVICE} \
#   --gpu_device ${GPU_DEVICE} \
#   --use_gemini_verifier \



echo ""
echo "Testing complete! Check results at: ${OUTPUT_DIR}"
