#!/usr/bin/env bash


# ===== Project Root =====
# Change this path to your project root directory
PROJECT_ROOT=${PROJECT_ROOT:-/home/andy2884/workspace/VG-AVS}

MODEL_PATH=${MODEL_PATH:-/home/andy2884/workspace/VG-AVS/src/open-r1-multimodal/output/grpo-pure-multistep-7b-0302-005-cam-extrinsic}

IMG_ROOT=${IMG_ROOT:-${PROJECT_ROOT}/data}
NUM_SAMPLES=${NUM_SAMPLES:--1} # use all samples

# Verifier settings
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES
VERIFIER_MODEL=${VERIFIER_MODEL:-gemini-2.5-flash}
VERIFIER_DEVICE=${VERIFIER_DEVICE:-cuda:0}
# AI2-THOR expects a logical GPU index inside CUDA_VISIBLE_DEVICES.
GPU_DEVICE=${GPU_DEVICE:-0}
AITHOR_GPU_DEVICE=${AITHOR_GPU_DEVICE:-${GPU_DEVICE}}
export AITHOR_GPU_DEVICE
STUDENT_DEVICE=${STUDENT_DEVICE:-cuda:0}


echo "Testing ProcTHOR action prediction model..."
echo "Model: ${MODEL_PATH}"
echo "Image root: ${IMG_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "AITHOR_GPU_DEVICE: ${AITHOR_GPU_DEVICE}"
echo "GPU device for rendering: ${GPU_DEVICE}"
echo ""

# existence
EXISTENCE_TEST_JSONL=${PROJECT_ROOT}/data/avs_existence_eval_final.jsonl
COUNTING_TEST_JSONL=/home/daehyeonchoi/code/eccv2026/AVS_eccv/data/avs_counting_eval_final.jsonl
STATE_TEST_JSONL=/home/daehyeonchoi/code/eccv2026/AVS_eccv/data/avs_state_eval_final.jsonl

MODEL_NAME=$(basename ${MODEL_PATH})


# OUTPUT_DIR=${PROJECT_ROOT}/src/open-r1-multimodal/output/visualization/${MODEL_NAME}/existence_eval/$(date +%Y%m%d_%H%M%S)
# python ${PROJECT_ROOT}/src/open-r1-multimodal/src/open_r1/test_procthor_multistep_action_accuracy.py \
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
#   --max_rollout_steps 4 \


# counting

# OUTPUT_DIR=${PROJECT_ROOT}/src/open-r1-multimodal/output/visualization/${MODEL_NAME}/counting_eval/$(date +%Y%m%d_%H%M%S)
# python ${PROJECT_ROOT}/src/open-r1-multimodal/src/open_r1/test_procthor_multistep_action_accuracy.py \
#   --model_path "${MODEL_PATH}" \
#   --test_jsonl "${COUNTING_TEST_JSONL}" \
#   --image_root "${IMG_ROOT}" \
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
#   --custom_house_path /home/daehyeonchoi/code/eccv2026/AVS_eccv/data/counting_house_data.json \




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



echo ""
echo "Testing complete! Check results at: ${OUTPUT_DIR}"
