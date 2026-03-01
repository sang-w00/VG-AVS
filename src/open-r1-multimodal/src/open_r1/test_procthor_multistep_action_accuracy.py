import argparse
import json
import os
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from open_r1.utils.gemini_utils import (
    _API_STATS,
    _resolve_gemini_model_id,
    _resolve_gpt_model_id,
    _is_gemini_backend,
    _is_gpt_backend,
    run_gemini_action_prediction,
    run_gpt_action_prediction,
    run_gemini_verifier,
    parse_mcq_letter_from_question,
    normalize_verifier_answer,
)
from open_r1.utils.model_load import resolve_model_path
from open_r1.utils.procthor_utils import build_additional_view, get_procthor_controller
from open_r1.utils.prompt_templates import (
    MULTISTEP_ACTION_PROMPT_TEMPLATE,
    MULTISTEP_FORMAT_PROMPT,
)
from open_r1.grpo_multistep import _parse_multistep_output
from open_r1.utils.visualization import create_multistep_rollout_visualization
from open_r1.utils.rewards import _verifier_answer
from open_r1.utils.string_utils import (
    action_string_to_list,
    action_string_to_list,
    extract_thinking_and_action_string,
)
from open_r1.utils.visualization import create_multistep_rollout_visualization
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Test ProcTHOR action prediction with visualization"
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to trained model checkpoint or Gemini model alias"
    )
    parser.add_argument("--test_jsonl", required=True, help="Path to test JSONL file")
    parser.add_argument(
        "--image_root",
        default=None,
        help="Root directory for images (if paths in JSONL are relative)",
    )
    parser.add_argument(
        "--output_dir",
        default="results/procthor_action_test",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--num_samples", type=int, default=-1, help="Number of samples to test"
    )
    parser.add_argument(
        "--verifier_model", default="qwen2.5vl:7b", help="Verifier model path or alias"
    )
    parser.add_argument(
        "--max_rollout_steps", type=int, default=3, help="Max steps for multi-step agent"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Max tokens for student model"
    )
    parser.add_argument(
        "--verifier_max_tokens", type=int, default=48, help="Max tokens for verifier"
    )
    parser.add_argument("--device", default="cuda:0", help="Device for student model")
    parser.add_argument(
        "--verifier_device", default="cpu", help="Device for verifier model"
    )
    parser.add_argument(
        "--gpu_device", type=int, default=0, help="GPU device for AI2-THOR rendering"
    )
    parser.add_argument(
        "--input_view_for_verifier",
        action="store_true",
        help="Skip action prediction, use input view directly for verifier",
    )
    parser.add_argument(
        "--gt_view_for_verifier",
        action="store_true",
        help="Skip action, use GT view directly for verifier",
    )
    parser.add_argument(
        "--gt_view_for_action",
        action="store_true",
        help="Use GT view as input to action model",
    )
    parser.add_argument(
        "--custom_house_path", default=None, help="Path to custom house path"
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=None,
        help="Max pixels for image processor (must match training config)",
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=None,
        help="Min pixels for image processor (must match training config)",
    )
    # GEMINI
    parser.add_argument(
        "--use_gemini_verifier",
        action="store_true",
        help="Use Gemini API for verifier (instead of local Qwen model)",
    )
    parser.add_argument(
        "--use_gemini_action",
        action="store_true",
        help="Use Gemini API for action prediction (instead of local model)",
    )
    parser.add_argument(
        "--use_gpt_action",
        action="store_true",
        help="Use GPT API for action prediction (instead of local model)",
    )


    parser.add_argument(
        "--use_sft",
        action="store_true",
        help="Use SFT format prompt (no reasoning tags)",
    )
    parser.add_argument(
        "--use_refine",
        action="store_true",
        help="Use SFT GRPO format prompt with refined reasoning (initial guess + refine)",
    )

    args = parser.parse_args()

    # Check mutually exclusive modes
    if args.input_view_for_verifier and args.gt_view_for_verifier:
        print("Error: --input_view_for_verifier and --gt_view_for_verifier are mutually exclusive")
        return

    if args.input_view_for_verifier:
        print(
            "\n[MODE] Input view for verifier: Skipping action prediction, using input view directly"
        )
    elif args.gt_view_for_verifier:
        print(
            "\n[MODE] GT view for verifier: Using GT view directly for verifier (skip action)"
        )
    elif args.gt_view_for_action:
        print(
            "\n[MODE] GT view for action: Using GT view as input to action model"
        )
    else:
        print(
            "\n[MODE] Action prediction mode: Using action model to predict and generate views"
        )
    


    # Load student model (skip if input_view_for_verifier or gt_view_for_verifier mode)
    student_model = None
    student_processor = None
    use_gemini_action = False
    use_gpt_action = False
    gemini_action_model_id = None
    gpt_action_model_id = None

    if not args.input_view_for_verifier and not args.gt_view_for_verifier:
        # Check if Gemini should be used for action prediction
        if args.use_gemini_action:
            use_gemini_action = True
            gemini_action_model_id = _resolve_gemini_model_id("gemini-2.5-pro")
            print(f"✓ Using Gemini API for action prediction: {gemini_action_model_id}")
            print(
                f"  API Key: {'Set' if os.environ.get('GEMINI_API_KEY') else 'NOT SET (will fail)'}"
            )
        elif args.use_gpt_action:
            use_gpt_action = True
            gpt_action_model_id = _resolve_gpt_model_id("gpt-5")
            print(f"✓ Using GPT API for action prediction: {gpt_action_model_id}")
            print(
                f"  API Key: {'Set' if os.environ.get('OPENAI_API_KEY') else 'NOT SET (will fail)'}"
            )
        else:
            print(f"Loading student model from {args.model_path}...")

            # Check if this is a LoRA checkpoint
            is_lora_checkpoint = False
            adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                is_lora_checkpoint = True
                print(f"✓ Detected LoRA checkpoint at: {args.model_path}")

            # Determine base model path
            if is_lora_checkpoint:
                # Read adapter_config.json to get base model name
                with open(adapter_config_path, "r") as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get(
                    "base_model_name_or_path", "Qwen/Qwen2.5-VL-7B-Instruct"
                )
                print(f"Base model: {base_model_name}")
                 # Detect model type from base model
                print(f"Loading base model: {base_model_name}")
                student_model = AutoModelForVision2Seq.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    ignore_mismatched_sizes=True,
                )
                # Load LoRA adapter
                print(f"Loading LoRA adapter from: {args.model_path}")
                student_model = PeftModel.from_pretrained(
                    student_model, args.model_path
                )
                print("✓ LoRA checkpoint loaded successfully")
            else:
                # Load full model checkpoint
                print("Loading full model checkpoint...")
                student_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.bfloat16
                )

            student_processor = AutoProcessor.from_pretrained(
                args.model_path, trust_remote_code=True
            )
            # Set max_pixels and min_pixels if provided (must match training config)
            if args.max_pixels is not None:
                student_processor.image_processor.max_pixels = args.max_pixels
            if args.min_pixels is not None:
                student_processor.image_processor.min_pixels = args.min_pixels

            for p in student_model.parameters():
                p.requires_grad_(False)
            student_model.to(args.device)
            student_model.eval()


    # Load verifier backend
    print(f"Loading verifier model/backend...")
    use_gemini_verifier = args.use_gemini_verifier
    verifier_model = None
    verifier_processor = None
    gemini_model_id = None
    if use_gemini_verifier:
        gemini_model_id = _resolve_gemini_model_id("gemini-2.5-flash")
        print(f"Using Gemini backend for verifier: {gemini_model_id}")
    else:
        verifier_path = resolve_model_path(args.verifier_model)
        verifier_processor = AutoProcessor.from_pretrained(
            verifier_path, trust_remote_code=True
        )
        verifier_model = AutoModelForVision2Seq.from_pretrained(
            verifier_path,
            torch_dtype=(
                torch.float32 if args.verifier_device == "cpu" else torch.bfloat16
            ),
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
        verifier_model.to(args.verifier_device)
        verifier_model.eval()
        for p in verifier_model.parameters():
            p.requires_grad_(False)

    # Initialize AI2-THOR controller (needed for action prediction and gt_action modes)
    # Initialize AI2-THOR controller (needed for action prediction and gt_action modes)
    controller = None
    if not args.input_view_for_verifier:
        print("Initializing AI2-THOR controller...")
        controller = get_procthor_controller(headless=True)
    else:
        print("Skipping AI2-THOR controller initialization (input_view_for_verifier mode)")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    test_data = []
    with open(args.test_jsonl, "r") as f:
        for line in f:
            test_data.append(json.loads(line.strip()))

    if args.num_samples > 0:
        test_data = test_data[: args.num_samples]

    print(f"Loaded {len(test_data)} test samples")

    # Test each sample
    results = []
    for idx, item in tqdm(enumerate(test_data)):
        print(f"\n{'='*80}")
        print(f"Sample {idx+1}/{len(test_data)}")
        print(f"{'='*80}")

        required_fields = [
            "scene_id",
        ]
        for field in required_fields:
            if field not in item:
                raise ValueError(f"Field {field} is required in the example")

        # gt action
        steps = item.get("steps", [])
        
        # Get image path - use anchor view (steps[0].view_image) to match training/GRPO
        # steps[0] is the anchor view, which is the actual starting point
        if steps:
            input_img_path = steps[0].get("view_image", item.get("question_image", ""))
        else:
            input_img_path = item.get("question_image", "")
        if not os.path.isabs(input_img_path) and args.image_root:
            input_img_path = os.path.join(args.image_root, input_img_path)
        assert os.path.exists(input_img_path), f"Input image not found: {input_img_path}"

        vqa_question = item.get("mcq_question")
        if vqa_question is None:
            vqa_question = item['question']

        if item['question_type'] == 'existence':
            vqa_question += "\\nAnswer with the option's letter from the given choices directly."
        elif item['question_type'] == 'counting':
            vqa_question += "\\nAnswer with a single number only."
        elif item["question_type"] == "state":
            vqa_question += "\\nAnswer with the option's letter from the given choices directly."
        elif item["question_type"] == "OCR":
            vqa_question += "\\nAnswer with a single number only."

        action_question = (
            MULTISTEP_ACTION_PROMPT_TEMPLATE.format(question=vqa_question)
            + MULTISTEP_FORMAT_PROMPT
        )
        
        gt_img_path = item.get("gt_image", steps[-1].get("view_image") if steps else "")
        if not os.path.isabs(gt_img_path) and args.image_root:
            gt_img_path = os.path.join(args.image_root, gt_img_path)
        assert os.path.exists(gt_img_path), f"GT image not found: {gt_img_path}"

        if steps:
            first_step = steps[0]
            gt_actions = [float(x) for x in first_step.get("action", [0, 0, 0])] if "action" in first_step and first_step["action"] else [0, 0, 0]
            
            position_list = first_step.get("position", {"x": 0, "y": 0, "z": 0})
            if isinstance(position_list, dict):
                position_list = [position_list["x"], position_list["y"], position_list["z"]]
            rotation_scalar = first_step.get("rotation", 0.0)
            
            gt_position_list = steps[-1].get("position", {"x": 0, "y": 0, "z": 0})
            if isinstance(gt_position_list, dict):
                gt_position_list = [gt_position_list["x"], gt_position_list["y"], gt_position_list["z"]]
            gt_rotation_scalar = steps[-1].get("rotation", 0.0)
        else:
            # Fallback for single-step data format
            gt_actions = [float(x) for x in item.get("gt_action", [0,0,0])]
            position_list = item.get("question_position", [0,0,0])
            rotation_scalar = item.get("question_rotation", 0.0)
            gt_position_list = item.get("gt_position", [0,0,0])
            gt_rotation_scalar = item.get("gt_rotation", 0.0)

        if args.gt_view_for_action:
            # overwrite input image and position
            print("[GT view for action mode] Using GT view as input to action model")
            input_img_path = gt_img_path
            render_position = {
                "x": float(gt_position_list[0]),
                "y": float(gt_position_list[1]),
                "z": float(gt_position_list[2]),
            }
            render_rotation = {
                "x": 0.0,
                "y": float(gt_rotation_scalar),
                "z": 0.0,
            }
            gt_actions = [0, 0, 0]
        else:
            render_position = {
                "x": float(position_list[0]),
                "y": float(position_list[1]),
                "z": float(position_list[2]),
            }
            render_rotation = {
                "x": 0.0,
                "y": float(rotation_scalar),
                "z": 0.0,
            }
        gt_position = {
            "x": float(gt_position_list[0]),
            "y": float(gt_position_list[1]),
            "z": float(gt_position_list[2]),
        }
        gt_rotation = {
            "x": 0.0,
            "y": float(gt_rotation_scalar),
            "z": 0.0,
        }

        scene_index = int(re.search(r'house_(\d+)', item["scene_id"]).group(1))
        data_split = item.get("split", "train")
        scene_path = None

        render_metadata = {
            "position": render_position,
            "rotation": render_rotation,
            "gt_position": gt_position,
            "gt_rotation": gt_rotation,
            "scene_index": scene_index,
            "trans_scale": 100.0,
            "check_obj_existence": False,
            "pixel_threshold": 200,
            "use_fallback": True,
            "num_grids": 8,
            "data_split": data_split, # Now, we don't use train hard coded when calling get_procthor_house().
            "custom_house_path": args.custom_house_path,
            "scene_path": scene_path,
        }

        if item.get("state", None):
            render_metadata["state"] = item["state"]
            render_metadata["object_id"] = item["object_id"]

        # Test inference
        try:
            input_img = Image.open(input_img_path).convert("RGB")
            gt_img = Image.open(gt_img_path).convert("RGB")
            gen_img = None
            predicted_actions = [0, 0, 0]
            thinking_process = "N/A"
            actions_text = "N/A"
            student_output = "N/A"

            if args.input_view_for_verifier:
                # Input view for verifier mode: Skip action prediction, use input view directly
                print("[Input view for verifier mode] Using input view directly for verifier")
                thinking_process = "N/A (input_view_for_verifier mode)"
                actions_text = "N/A (input_view_for_verifier mode)"
                student_output = "N/A (input_view_for_verifier mode)"

                # Run verifier directly on input image
                verifier_images = [input_img]
                if use_gemini_verifier:
                    verifier_full_output, verifier_answer = run_gemini_verifier(
                        verifier_images, vqa_question, gemini_model_id
                    )
                else:
                    verifier_full_output, verifier_answer = _verifier_answer(
                        verifier_images,
                        vqa_question,
                        verifier_model=verifier_model,
                        verifier_processor=verifier_processor,
                        max_new_tokens=args.verifier_max_tokens,
                    )
                # Normalize by question type
                verifier_answer = normalize_verifier_answer(
                    vqa_question, verifier_full_output, verifier_answer, item.get("question_type")
                )
                print(f"Verifier Output: {verifier_full_output[:200]}...")
            elif args.gt_view_for_verifier:
                # GT view for verifier mode: Use GT view directly for verifier
                print("[GT view for verifier mode] Using GT view directly for verifier")
                thinking_process = "N/A (gt_view_for_verifier mode)"
                actions_text = "N/A (gt_view_for_verifier mode)"
                student_output = "N/A (gt_view_for_verifier mode)"

                print(f"Using gt_action: {gt_actions}")

                # Run verifier on generated view
                verifier_images = [gt_img]

                if use_gemini_verifier:
                    verifier_full_output, verifier_answer = run_gemini_verifier(
                        verifier_images, vqa_question, gemini_model_id
                    )
                else:
                    verifier_full_output, verifier_answer = _verifier_answer(
                        verifier_images,
                        vqa_question,
                        verifier_model=verifier_model,
                        verifier_processor=verifier_processor,
                        max_new_tokens=args.verifier_max_tokens,
                    )
                verifier_answer = normalize_verifier_answer(
                    vqa_question, verifier_full_output, verifier_answer, item.get("question_type")
                )
                print(f"Verifier Output: {verifier_full_output[:200]}...")
            else:
                # Action mode: Predict actions and generate new view
                pil_img = Image.open(input_img_path).convert("RGB")
                
                # Ensure image has reasonable dimensions to avoid tensor issues
                def process_img(img):
                    w, h = img.size
                    if w < 28 or h < 28:
                        if w < h:
                            new_w = 28
                            new_h = int(h * (28 / w))
                        else:
                            new_h = 28
                            new_w = int(w * (28 / h))
                        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    return img
                
                pil_img = process_img(pil_img)

                chat = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text", "text": action_question},
                        ],
                    }
                ]
                
                current_images = [pil_img]
                
                thinking_process = "N/A"
                actions_text = "N/A"
                student_output = "N/A"
                predicted_actions = [0, 0, 0]
                gen_img = None
                
                rollout_steps = []
                
                for step in range(args.max_rollout_steps):
                    print(f"--- Rollout Step {step+1}/{args.max_rollout_steps} ---")
                    
                    if use_gemini_action or use_gpt_action:
                        api_messages = []
                        image_turn_idx = 0
                        for turn in chat:
                            role = turn.get("role")
                            if role == "assistant":
                                assistant_texts = []
                                for c in turn.get("content", []):
                                    if c.get("type") == "text" and c.get("text"):
                                        assistant_texts.append(str(c.get("text")))
                                if assistant_texts:
                                    api_messages.append(
                                        {"role": "assistant", "text": "\n".join(assistant_texts)}
                                    )
                                continue

                            if role != "user":
                                continue

                            user_texts = []
                            has_image = False
                            for c in turn.get("content", []):
                                if c.get("type") == "text" and c.get("text"):
                                    user_texts.append(str(c.get("text")))
                                elif c.get("type") == "image":
                                    has_image = True

                            user_message = {"role": "user"}
                            if user_texts:
                                user_message["text"] = "\n".join(user_texts)
                            if has_image and image_turn_idx < len(current_images):
                                user_message["image"] = current_images[image_turn_idx]
                                image_turn_idx += 1
                            if "text" in user_message or "image" in user_message:
                                api_messages.append(user_message)

                        if use_gemini_action:
                            print(
                                f"[Gemini Action] Calling Gemini API with model {gemini_action_model_id}..."
                            )
                            step_output = run_gemini_action_prediction(
                                image=None,
                                prompt=None,
                                model_id=gemini_action_model_id,
                                messages=api_messages,
                            )
                        else:
                            print(
                                f"[GPT Action] Calling GPT API with model {gpt_action_model_id}..."
                            )
                            step_output = run_gpt_action_prediction(
                                image=None,
                                prompt=None,
                                model_id=gpt_action_model_id,
                                messages=api_messages,
                            )

                        if step_output.startswith("ERROR:"):
                            print(f"Action prediction failed: {step_output}")
                            break
                    else:
                        chat_text = student_processor.apply_chat_template(
                            chat, tokenize=False, add_generation_prompt=True
                        )
                        
                        inputs = student_processor(
                            text=[chat_text],
                            images=current_images,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=4096,
                        )
                        inputs = {
                            k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                            for k, v in inputs.items()
                        }
                        
                        student_model.eval()
                        if hasattr(student_model, "reset_cache"):
                            student_model.reset_cache()
                            
                        gen = student_model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            pad_token_id=student_processor.tokenizer.eos_token_id,
                            use_cache=True,
                            output_attentions=False,
                            output_hidden_states=False,
                            return_dict_in_generate=False,
                        )
                        
                        prompt_length = inputs["input_ids"].size(1)
                        # For AutoModel generation we might get full sequence back
                        # if the model is Qwen2VL
                        if isinstance(student_model, Qwen2_5_VLForConditionalGeneration):
                            completion_ids = gen[:, prompt_length:]
                        else:
                            completion_ids = gen
                            
                        try:
                            step_output = student_processor.batch_decode(
                                completion_ids, skip_special_tokens=True
                            )[0]
                        except:
                            step_output = student_processor.batch_decode(
                                gen, skip_special_tokens=True
                            )[0]
                        
                    print(f"Step {step+1} Output:\\n{step_output}\\n")
                    student_output = step_output if step_output else student_output
                    
                    chat.append({"role": "assistant", "content": [{"type": "text", "text": step_output}]})
                    
                    parsed = _parse_multistep_output(step_output)
                    thinking_process = parsed.get("thinking", "")
                    actions_text = parsed.get("action_text", "")
                    predicted_actions = parsed.get("action")
                    
                    step_record = {
                        "step_index": step + 1,
                        "thinking": thinking_process,
                        "action_text": actions_text or "<invalid>",
                        "raw_completion": step_output,
                        "is_stop": parsed.get("is_stop", False),
                        "is_unknown": parsed.get("is_unknown", False),
                        "view_image": None
                    }
                    rollout_steps.append(step_record)
                    
                    if parsed.get("is_stop"):
                        print("Model emitted <stop>.")
                        break
                    
                    effective_action = predicted_actions
                    if parsed.get("is_unknown"):
                        effective_action = [90, 0, 0] # fallback
                        
                    if effective_action is None:
                        print("Invalid action. Stopping rollout.")
                        break
                        
                    print(f"Generating view for action {effective_action}...")
                    gen_img, result_metadata = build_additional_view(
                        controller, effective_action, render_metadata
                    )
                    
                    if gen_img:
                        print("✓ View generated")
                        step_record["view_image"] = gen_img
                        current_images.append(process_img(gen_img))
                        chat.append({"role": "user", "content": [{"type": "image", "text": None}]})
                        
                        if result_metadata and result_metadata.get("actual_position"):
                            actual_pos = result_metadata["actual_position"]
                            rot1, rot2 = effective_action[0], effective_action[2]
                            rot1_norm = rot1 if rot1 >= 0 else rot1 + 360
                            rot2_norm = rot2 if rot2 >= 0 else rot2 + 360
                            new_yaw = (render_metadata["rotation"].get("y", 0) + rot1_norm + rot2_norm) % 360
                            
                            render_metadata = dict(render_metadata)
                            render_metadata["position"] = actual_pos
                            render_metadata["rotation"] = {"x": 0.0, "y": new_yaw, "z": 0.0}
                    else:
                        print("✗ Failed to generate view. Stopping rollout.")
                        break

                # Run verifier (use latest generated image if available, else input)
                final_img = current_images[-1]

                verifier_images = [final_img]
                
                if use_gemini_verifier:
                    verifier_full_output, verifier_answer = run_gemini_verifier(
                        verifier_images, vqa_question, gemini_model_id
                    )
                else:
                    verifier_full_output, verifier_answer = _verifier_answer(
                        verifier_images,
                        vqa_question,
                        verifier_model=verifier_model,
                        verifier_processor=verifier_processor,
                        max_new_tokens=args.verifier_max_tokens,
                    )
                verifier_answer = normalize_verifier_answer(
                    vqa_question, verifier_full_output, verifier_answer, item.get("question_type")
                )
                print(f"Verifier Output: {verifier_full_output[:200]}...")

            # Check accuracy
            gt_answer = item.get("mcq_answer", item.get("answer"))
            if gt_answer is None and item.get("question_type") == "existence":
                gt_answer = "A" if "yes" in str(item.get("answer", "")).lower() else "B"
            else:
                gt_answer = str(gt_answer)
                
            is_correct = verifier_answer.strip().lower() == gt_answer.strip().lower()
            print(f"Accuracy: {'✓ CORRECT' if is_correct else '✗ WRONG'}")

            # Save generated images for the rollout
            gen_img_dir = os.path.join(args.output_dir, "generated_images", f"sample_{idx:04d}")
            os.makedirs(gen_img_dir, exist_ok=True)
            for s_idx, step_rec in enumerate(rollout_steps):
                if step_rec.get("view_image"):
                    step_img_path = os.path.join(gen_img_dir, f"step_{s_idx+1}.png")
                    step_rec["view_image"].save(step_img_path)
            
            # Save thinking process and raw output separately
            thinking_dir = os.path.join(args.output_dir, "thinking_process")
            raw_output_dir = os.path.join(args.output_dir, "raw_output")
            os.makedirs(thinking_dir, exist_ok=True)
            os.makedirs(raw_output_dir, exist_ok=True)
            
            thinking_path = os.path.join(thinking_dir, f"sample_{idx:04d}_thinking.txt")
            raw_output_path = os.path.join(raw_output_dir, f"sample_{idx:04d}_raw_output.txt")
            
            with open(thinking_path, 'w', encoding='utf-8') as f:
                for s in rollout_steps:
                    f.write(f"--- Step {s['step_index']} ---\n{s['thinking']}\n\n")
            with open(raw_output_path, 'w', encoding='utf-8') as f:
                for s in rollout_steps:
                    f.write(f"--- Step {s['step_index']} ---\n{s['raw_completion']}\n\n")

            # Create visualization
            vis_path = os.path.join(args.output_dir, f"sample_{idx:04d}.png")
            create_multistep_rollout_visualization(
                input_images=current_images,
                trajectory_steps=rollout_steps,
                vqa_question=vqa_question,
                verifier_answer=verifier_answer,
                gt_answer=gt_answer,
                verifier_reward=1.0 if is_correct else 0.0,
                output_path=vis_path,
                target_visibility=item.get("target_object_visibility_level", "unknown"),
                turn_type="multi-turn" if len(steps) > 2 else "single-turn",
                gt_image=gt_img_path,
            )

            # Record result
            results.append(
                {
                    "sample_id": idx,
                    "input_image": input_img_path,
                    "num_steps": len(rollout_steps),
                    "final_predicted_action": predicted_actions,
                    "verifier_answer": verifier_answer,
                    "gt_answer": gt_answer,
                    "correct": is_correct,
                    "has_generated_view": len(current_images) > 1,
                }
            )

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            traceback.print_exc()
            print(f"Error processing sample {idx}: {e}")
            traceback.print_exc()

    # Save results JSON
    results_json_path = os.path.join(args.output_dir, "results.json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Calculate summary statistics
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    with_view = sum(1 for r in results if r["has_generated_view"])
    accuracy = (correct / total * 100) if total > 0 else 0.0

    # Save summary JSON
    mode_name = (
        "input_view_for_verifier"
        if args.input_view_for_verifier
        else ("gt_view_for_verifier" if args.gt_view_for_verifier else "action_prediction")
    )
    summary = {
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "samples_with_generated_view": with_view,
        "mode": mode_name,
        "model_path": (
            args.model_path if not (args.input_view_for_verifier or args.gt_view_for_verifier) else "N/A"
        ),
        "action_model_type": (
            "gemini_api"
            if use_gemini_action
            else (
                "local_model" if not (args.input_view_for_verifier or args.gt_view_for_verifier) else "N/A"
            )
        ),
        "verifier_model": args.verifier_model,
        "verifier_model_type": "gemini_api" if use_gemini_verifier else "local_model",
        "test_jsonl": args.test_jsonl,
    }
    summary_json_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Mode: {summary['mode']}")
    if use_gemini_action:
        print(f"Action Model: Gemini API ({gemini_action_model_id})")
    elif not (args.input_view_for_verifier or args.gt_view_for_verifier):
        print(f"Action Model: Local model ({args.model_path})")
    print(
        f"Verifier Model: {'Gemini API (' + gemini_model_id + ')' if use_gemini_verifier else args.verifier_model}"
    )
    print(f"Total samples: {total}")
    if not args.input_view_for_verifier:
        print(f"Samples with generated view: {with_view}/{total}")
    if args.gt_view_for_verifier:
        print(f"Using GT view directly for verifier (skip action)")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {results_json_path}")
    print(f"Summary saved to: {summary_json_path}")
    print(f"Visualizations saved to: {args.output_dir}/")
    # API error summary
    if use_gemini_verifier or use_gemini_action:
        print("-")
        print("API status (Gemini):")
        print(f"  Retry events: {_API_STATS['gemini_retry_events']}")
        print(f"  Failed requests: {_API_STATS['gemini_failed_requests']}")
        if _API_STATS["gemini_failed_requests"] > 0:
            print("  Failures:")
            for d in _API_STATS["gemini_retry_details"][-5:]:
                print(f"    - {d}")


if __name__ == "__main__":
    main()
