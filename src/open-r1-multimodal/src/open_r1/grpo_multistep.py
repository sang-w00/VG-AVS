# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multi-step GRPO trainer for ProcTHOR Active VQA task.
Uses multi-step rollouts (max 5 steps) where the model outputs:
  <think> reasoning </think>
  <head> X </head> <fwd> Y </fwd> <view> Z </view>
Or:
  <think> reasoning </think>
  <stop>
Or:
  <think> reasoning </think>
  <unknown>

After rollout, VLM verifier computes reward on the final view.
"""

import json
import os
import pathlib
import re
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Any, Union

import PIL
from PIL import Image
import torch
import numpy as np
from datasets import Dataset
from open_r1.qwen2_5vl_monkey_patch import (
    monkey_patch_qwen2_5vl_flash_attn,
    monkey_patch_torch_load,
)
from open_r1.trainer import GRPOConfig, VLMGRPOTrainer
from open_r1.utils.model_load import get_vlm_module, initialize_verifier
from open_r1.utils.prompt_templates import (
    MULTISTEP_ACTION_PROMPT_TEMPLATE,
    SINGLE_TURN_MULTISTEP_ACTION_PROMPT_TEMPLATE,
    MULTISTEP_FORMAT_PROMPT,
)
from open_r1.utils.rewards import (
    default_accuracy_reward,
    _add_question_type_prompt,
)
from open_r1.utils.procthor_utils import (
    build_additional_view,
    get_procthor_controller,
)
from open_r1.utils.visualization import create_multistep_rollout_visualization
from transformers.utils import logging
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.models import unwrap_model_for_generation

monkey_patch_qwen2_5vl_flash_attn()
monkey_patch_torch_load()

logger = logging.get_logger(__name__)

DEBUG_MODE = str(os.getenv("DEBUG_MODE", "0")) == "1"
MAX_ROLLOUT_STEPS = 5
UNKNOWN_FALLBACK_ACTION = [90, 0, 0]
VISION_TOKEN_STRINGS = [
    "<|image_pad|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|video_pad|>",
    "<image>",
]


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """Script arguments for multi-step GRPO training."""

    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    val_split_seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed for train/validation split."},
    )
    save_validation_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save validation split."},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["multistep_verifier", "multistep_format"],
        metadata={"help": "List of reward functions."},
    )
    grpo_reward_weights: list[float] = field(
        default_factory=lambda: [1.0, 0.3],
        metadata={"help": "Weights for each reward function"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    use_fallback: bool = field(
        default=True,
        metadata={"help": "Use fallback position when predicted action is infeasible"},
    )
    fallback_num_grids: int = field(
        default=8,
        metadata={"help": "Number of grid points for fallback"},
    )
    customized_scene_path: str = field(
        default=None,
        metadata={"help": "Path to customized scene"},
    )
    vlm_lr: Optional[float] = field(
        default=1e-6,
        metadata={"help": "Learning rate for VLM backbone"},
    )
    max_rollout_steps: int = field(
        default=5,
        metadata={"help": "Maximum number of rollout steps (default: 5)"},
    )
    save_rollout_visualizations: bool = field(
        default=True,
        metadata={"help": "Save rollout visualization image during training"},
    )
    rollout_vis_interval: int = field(
        default=1,
        metadata={"help": "Save one rollout visualization every N training steps"},
    )
    rollout_vis_num_samples: int = field(
        default=1,
        metadata={"help": "How many rollouts to visualize per save step"},
    )
    rollout_vis_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory for rollout visualization images"},
    )
    single_turn_vision: bool = field(
        default=False,
        metadata={"help": "If True, predict action in a single turn using only the current view image and prompt."},
    )


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


def _get_local_rank():
    return int(os.getenv("LOCAL_RANK", "0"))


def _parse_multistep_output(text: str) -> dict:
    """Parse model output to extract thinking, action params, or stop/unknown decisions.
    
    Returns:
        dict with keys:
            - 'thinking': str (content of <think> tag)
            - 'is_stop': bool (True if <stop> tag found)
            - 'is_unknown': bool (True if <unknown> tag found)
            - 'action': list[int] or None ([head, fwd, view] if move action exists)
            - 'action_text': str (normalized action/decision text for history)
    """
    think_matches = re.findall(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
    thinking = think_matches[-1].strip() if think_matches else ""
    for tok in VISION_TOKEN_STRINGS:
        thinking = thinking.replace(tok, "")
    thinking = thinking.strip()

    if re.search(r'<unknown\s*>', text, re.IGNORECASE):
        return {
            'thinking': thinking,
            'is_stop': False,
            'is_unknown': True,
            'action': None,
            'action_text': '<unknown>',
        }

    if re.search(r'<stop\s*>', text, re.IGNORECASE):
        return {
            'thinking': thinking,
            'is_stop': True,
            'is_unknown': False,
            'action': None,
            'action_text': '<stop>',
        }

    # Parse last complete move tags
    head_matches = re.findall(r'<head>\s*(-?\d+)\s*</head>', text, re.IGNORECASE)
    fwd_matches = re.findall(r'<fwd>\s*(-?\d+)\s*</fwd>', text, re.IGNORECASE)
    view_matches = re.findall(r'<view>\s*(-?\d+)\s*</view>', text, re.IGNORECASE)

    if head_matches and fwd_matches and view_matches:
        action = [int(head_matches[-1]), int(fwd_matches[-1]), int(view_matches[-1])]
        action_text = (
            f"<head> {action[0]} </head> "
            f"<fwd> {action[1]} </fwd> "
            f"<view> {action[2]} </view>"
        )
        return {
            'thinking': thinking,
            'is_stop': False,
            'is_unknown': False,
            'action': action,
            'action_text': action_text,
        }

    return {
        'thinking': thinking,
        'is_stop': False,
        'is_unknown': False,
        'action': None,
        'action_text': '',
    }


def _build_history_text(history: list) -> str:
    """Build history text from list of step entries."""
    if not history:
        return ""
    
    parts = []
    for i, entry in enumerate(history):
        step_text = f"\n[Step {i+1}]\n"
        step_text += f"<think> {entry['thinking']} </think>\n"
        step_text += f"{entry['action_text']}\n"
        parts.append(step_text)
    
    return "\nNavigation History:" + "".join(parts)


def _extract_answer_text(solution: str) -> str:
    """Extract answer text from <answer> tag if present."""
    if solution is None:
        return ""
    match = re.search(r"<answer>(.*?)</answer>", str(solution), re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else str(solution).strip()


def _score_multistep_format_text(content: str) -> float:
    """Score one model output chunk for multi-step format quality."""
    stripped = content.strip()

    has_think = re.search(r'<think>.*?</think>', stripped, re.DOTALL | re.IGNORECASE)
    if not has_think:
        return 0.0

    if re.search(r'<unknown\s*>', stripped, re.IGNORECASE):
        return 1.0

    if re.search(r'<stop\s*>', stripped, re.IGNORECASE):
        return 1.0

    # Check for move params
    head_m = re.search(r'<head>\s*(-?\d+)\s*</head>', stripped, re.IGNORECASE)
    fwd_m = re.search(r'<fwd>\s*(-?\d+)\s*</fwd>', stripped, re.IGNORECASE)
    view_m = re.search(r'<view>\s*(-?\d+)\s*</view>', stripped, re.IGNORECASE)

    if head_m and fwd_m and view_m:
        try:
            h, f, v = int(head_m.group(1)), int(fwd_m.group(1)), int(view_m.group(1))
            h_ok = -90 <= h <= 90
            f_ok = f >= 0
            v_ok = -90 <= v <= 90
            return 1.0 if (h_ok and f_ok and v_ok) else 0.75
        except ValueError:
            return 0.5

    has_partial_decision_tag = re.search(
        r'<head>|<fwd>|<view>|<stop>|<unknown>',
        stripped,
        re.IGNORECASE,
    )
    return 0.5 if has_partial_decision_tag else 0.25


def multistep_format_reward(completions, **kwargs):
    """Format reward for multi-step output.
    
    Checks that output follows:
      <think>...</think> followed by one of:
        1) <head><fwd><view>
        2) <stop>
        3) <unknown>
    
    Scoring:
    - 1.0: <think> + valid move tags in range OR <stop> OR <unknown>
    - 0.75: <think> + move tags present with values out of range
    - 0.5: <think> + at least one decision tag present but not parseable as valid move/stop/unknown
    - 0.25: <think> present, decision tags missing
    - 0.0: Missing <think>

    If `step_texts_per_sample` is provided in kwargs, score each rollout step independently
    and return the mean score over steps for each sample.
    """
    contents = [c[0]["content"] for c in completions]
    step_texts_per_sample = kwargs.get("step_texts_per_sample", None)

    # Preferred path: score each step independently, then aggregate per-sample.
    if isinstance(step_texts_per_sample, list) and len(step_texts_per_sample) > 0:
        rewards = []
        for i, step_texts in enumerate(step_texts_per_sample):
            if isinstance(step_texts, list) and len(step_texts) > 0:
                step_scores = [
                    _score_multistep_format_text(step_text)
                    for step_text in step_texts
                    if isinstance(step_text, str)
                ]
                if step_scores:
                    rewards.append(float(np.min(step_scores)))
                    continue

            # Fallback to trajectory-level score if step list is empty/invalid.
            fallback_content = contents[i] if i < len(contents) else ""
            rewards.append(_score_multistep_format_text(fallback_content))

        # Keep return length aligned with completions length.
        if len(rewards) < len(contents):
            for i in range(len(rewards), len(contents)):
                rewards.append(_score_multistep_format_text(contents[i]))
        return rewards

    # Backward-compatible fallback if step-level inputs are not provided.
    return [_score_multistep_format_text(content) for content in contents]


def multistep_verifier_reward(completions, **kwargs):
    """VLM verifier reward for multi-step rollouts.
    
    This function is called AFTER the rollout is complete.
    It uses the final rendered view from the rollout to compute the reward.
    The final views and metadata are passed via kwargs.
    """
    contents = [c[0]["content"] for c in completions]
    rewards = [0.0 for _ in contents]

    final_images = kwargs.get("final_images", [])
    questions = kwargs.get("vqa_question", [])
    solutions = kwargs.get("solution", [])
    question_types = kwargs.get("question_type", [])
    verifier_details_out = kwargs.get("verifier_details_out", None)
    verifier_max_new_tokens = int(os.getenv("VERIFIER_MAX_NEW_TOKENS", "16"))
    verifier_minibatch = max(1, int(os.getenv("VERIFIER_MINIBATCH", "8")))
    
    # Try to grab the exact device from completions or kwargs to keep verifier on same GPU as actor
    device = kwargs.get("device", completions[0][0]["content"].device if hasattr(completions[0][0]["content"], "device") else None)
    verifier_model, verifier_processor = initialize_verifier(device=device)
    t0 = time.time()

    pending_items = []
    for i, _ in enumerate(contents):
        final_img = final_images[i] if i < len(final_images) else None
        question = questions[i] if i < len(questions) else ""
        sol = solutions[i] if i < len(solutions) else ""
        qt = question_types[i] if i < len(question_types) else None
        detail = {
            "verifier_answer": None,
            "verifier_raw": None,
            "reward": 0.0,
            "question": question,
            "solution": sol,
        }
        if isinstance(verifier_details_out, list) and i < len(verifier_details_out):
            verifier_details_out[i] = detail
        if final_img is not None:
            pending_items.append((i, final_img, question, qt, sol))

    for s in range(0, len(pending_items), verifier_minibatch):
        chunk = pending_items[s : s + verifier_minibatch]
        try:
            pil_images = []
            chat_texts = []
            for _, final_img, question, qt, _ in chunk:
                if isinstance(final_img, str):
                    pil = Image.open(final_img).convert("RGB")
                elif isinstance(final_img, PIL.Image.Image):
                    pil = final_img.convert("RGB")
                else:
                    pil = None
                if pil is None:
                    pil_images.append(None)
                    chat_texts.append(None)
                    continue
                q_with_prompt = _add_question_type_prompt(question, qt)
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil},
                        {"type": "text", "text": q_with_prompt},
                    ],
                }]
                chat_text = verifier_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                pil_images.append(pil)
                chat_texts.append(chat_text)

            valid_indices = [idx for idx, (im, txt) in enumerate(zip(pil_images, chat_texts)) if im is not None and txt is not None]
            if not valid_indices:
                continue

            batch_images = [pil_images[idx] for idx in valid_indices]
            batch_texts = [chat_texts[idx] for idx in valid_indices]
            inputs = verifier_processor(
                text=batch_texts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
            ).to(verifier_model.device)
            gen = verifier_model.generate(
                **inputs,
                max_new_tokens=verifier_max_new_tokens,
                do_sample=False,
            )
            outputs = verifier_processor.batch_decode(gen, skip_special_tokens=True)

            for local_pos, out_text in zip(valid_indices, outputs):
                i, _, question, _, sol = chunk[local_pos]
                ans_blocks = re.findall(r"<answer>(.*?)</answer>", out_text, re.DOTALL | re.IGNORECASE)
                verifier_ans = (
                    ans_blocks[-1].strip().rstrip(".").strip()
                    if ans_blocks
                    else out_text.strip().split("\n")[-1].strip().rstrip(".").strip()
                )
                reward = default_accuracy_reward(verifier_ans, sol) if verifier_ans is not None else 0.0
                rewards[i] = reward

                if isinstance(verifier_details_out, list) and i < len(verifier_details_out):
                    verifier_details_out[i]["verifier_raw"] = out_text
                    verifier_details_out[i]["verifier_answer"] = verifier_ans
                    verifier_details_out[i]["reward"] = reward

                if DEBUG_MODE and _get_local_rank() == 0:
                    print(
                        f"[multistep_verifier] q='{question[:60]}' verifier='{verifier_ans}' "
                        f"gt='{sol[:60]}' r={reward:.3f}"
                    )
        except Exception as e:
            if DEBUG_MODE and _get_local_rank() == 0:
                print(f"[multistep_verifier] batch error: {e}")
            for i, _, _, _, _ in chunk:
                rewards[i] = 0.0
                if isinstance(verifier_details_out, list) and i < len(verifier_details_out):
                    verifier_details_out[i]["error"] = str(e)

        if DEBUG_MODE and _get_local_rank() == 0:
            elapsed = time.time() - t0
            done = min(s + len(chunk), len(pending_items))
            print(f"[multistep_verifier] progress {done}/{len(pending_items)} elapsed={elapsed:.1f}s")
            
    return rewards


reward_funcs_registry = {
    "multistep_verifier": multistep_verifier_reward,
    "multistep_format": multistep_format_reward,
}


class MultiStepGRPOTrainer(VLMGRPOTrainer):
    """GRPO Trainer with multi-step rollout support.
    
    Overrides _generate_and_score_completions to perform multi-step rollouts:
    1. Start from query position with query view
    2. Generate thinking + move/stop/unknown decision
    3. If move or unknown: render new view, update history, repeat
    4. If stop or max steps: compute VLM verifier reward
    """
    
    def __init__(self, max_rollout_steps=5, script_args=None, **kwargs):
        super().__init__(**kwargs)
        self.max_rollout_steps = max_rollout_steps
        self.script_args = script_args

        self.save_rollout_visualizations = bool(
            getattr(script_args, "save_rollout_visualizations", False)
        )
        self.rollout_vis_interval = max(
            1, int(getattr(script_args, "rollout_vis_interval", 1) or 1)
        )
        self.rollout_vis_num_samples = max(
            1, int(getattr(script_args, "rollout_vis_num_samples", 1) or 1)
        )
        self._last_rollout_vis_global_step: Optional[int] = None
        default_vis_dir = os.path.join(self.args.output_dir, "rollout_visualizations")
        custom_vis_dir = getattr(script_args, "rollout_vis_dir", None)
        self.rollout_vis_dir = custom_vis_dir or default_vis_dir

        if self.save_rollout_visualizations and self.accelerator.process_index == 0:
            os.makedirs(self.rollout_vis_dir, exist_ok=True)

    def _should_save_rollout_visualizations(self) -> bool:
        """Save rollout visualization only on process 0 and configured interval."""
        if not self.save_rollout_visualizations:
            return False
        if self.accelerator.process_index != 0:
            return False
        current_global_step = int(self.state.global_step)
        if self._last_rollout_vis_global_step == current_global_step:
            return False
        return (current_global_step % self.rollout_vis_interval) == 0

    def _save_rollout_visualizations(self, rollout_records: list[dict], verifier_details: list[dict]) -> None:
        """Save one or more rollout summary images for the current training step."""
        if not self._should_save_rollout_visualizations():
            return
        if not rollout_records:
            return

        os.makedirs(self.rollout_vis_dir, exist_ok=True)
        max_samples = min(self.rollout_vis_num_samples, len(rollout_records))
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        rollout_call_idx = int(getattr(self, "_step", 0))

        for sample_idx in range(max_samples):
            record = rollout_records[sample_idx]
            details = {}
            if isinstance(verifier_details, list) and sample_idx < len(verifier_details):
                details = verifier_details[sample_idx] or {}

            output_name = (
                f"rollout_step_{int(self.state.global_step):07d}"
                f"_call_{rollout_call_idx:07d}"
                f"_sample_{sample_idx:02d}_{now_str}.png"
            )
            output_path = os.path.join(self.rollout_vis_dir, output_name)

            try:
                create_multistep_rollout_visualization(
                    input_images=record.get("input_images", []),
                    trajectory_steps=record.get("steps", []),
                    vqa_question=record.get("question", ""),
                    verifier_answer=details.get("verifier_answer"),
                    gt_answer=_extract_answer_text(record.get("solution", "")),
                    verifier_reward=details.get("reward"),
                    output_path=output_path,
                    target_visibility=record.get("target_visibility", "unknown"),
                    turn_type=record.get("turn_type", "unknown"),
                )
                if DEBUG_MODE and _get_local_rank() == 0:
                    print(f"[rollout_vis] saved: {output_path}")
            except Exception as e:
                if DEBUG_MODE and _get_local_rank() == 0:
                    print(f"[rollout_vis] failed to save {output_path}: {e}")
        self._last_rollout_vis_global_step = int(self.state.global_step)
    
    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]], model
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Multi-step rollout: generate actions, render views, accumulate history."""
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        
        # === MULTI-STEP ROLLOUT ===
        # For each sample, perform multi-step rollout
        all_completions_text = []  # Final concatenated text per sample (for reward funcs)
        all_final_images = []  # Final rendered view per sample
        all_step_counts = []  # Number of steps taken per sample
        all_step_texts_per_sample = []  # Step texts per sample for step-wise format reward
        all_rollout_records = []  # Trajectory records for visualization
        
        # New collection for step-wise logp computation (Hstar style, full trajectory)
        trajectory_chat_list = []
        trajectory_images_list = []
        
        # We need the controller for rendering new views
        controller = get_procthor_controller(headless=True)
        
        for sample_idx, x in enumerate(inputs):
            # Initialize per-sample state
            question = x.get("vqa_question", "")
            if self.script_args.single_turn_vision:
                base_prompt = SINGLE_TURN_MULTISTEP_ACTION_PROMPT_TEMPLATE.format(question=question) + MULTISTEP_FORMAT_PROMPT
            else:
                base_prompt = MULTISTEP_ACTION_PROMPT_TEMPLATE.format(question=question) + MULTISTEP_FORMAT_PROMPT
            
            # Load initial images
            image_paths = x.get("image_path", [])
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            
            current_images = []
            for p in image_paths:
                try:
                    img = PIL.Image.open(p).convert('RGB')
                    w, h = img.size
                    if w < 28 or h < 28:
                        if w < h:
                            new_w, new_h = 28, int(h * (28 / w))
                        else:
                            new_h, new_w = 28, int(w * (28 / h))
                        img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                    current_images.append(img)
                except Exception as e:
                    if DEBUG_MODE and _get_local_rank() == 0:
                        print(f"[rollout] Failed to load image {p}: {e}")
            
            # Current render metadata (position, rotation, scene info)
            render_metadata = x.get("render_metadata", {})
            current_position = render_metadata.get("position", {})
            current_rotation = render_metadata.get("rotation", {})
            
            # Set up the chat for multi-turn rollout
            if self.script_args.single_turn_vision:
                # In single turn vision, we only give the *last* current image and the prompt.
                chat = []
                chat.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "text": None},
                        {"type": "text", "text": base_prompt}
                    ]
                })
                # all_chat_images is just the single image we care about
                all_chat_images = [current_images[-1]] if current_images else []
            else:
                chat = []
                image_content = [{"type": "image", "text": None} for _ in current_images]
                chat.append({
                    "role": "user",
                    "content": [
                        *image_content,
                        {"type": "text", "text": base_prompt}
                    ]
                })
                all_chat_images = list(current_images)

            full_completion_text = ""
            step_texts_this_sample = []
            final_image = current_images[-1] if current_images else None
            step_count = 0
            
            sample_flat_chats = []
            sample_flat_images = []
            
            rollout_record = {
                "question": question,
                "solution": x.get("solution", ""),
                "target_visibility": x.get("target_visibility", "unknown"),
                "turn_type": x.get("turn_type", "unknown"),
                "input_images": list(current_images),
                "steps": [],
            }
            
            for step in range(self.max_rollout_steps):
                step_count = step + 1
                
                # Prepare input for generation from the current conversational chat state
                step_input = {
                    "prompt": chat,
                    "image_path": None,  # Images passed directly
                }
                
                prompts_text = self.vlm_module.prepare_prompt(self.processing_class, [step_input])
                flat_prompt_text = prompts_text[0]
                
                prompt_inputs, _ = self.vlm_module.prepare_model_inputs(
                    self.processing_class,
                    prompts_text,
                    all_chat_images,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                )
                prompt_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                               for k, v in prompt_inputs.items()}
                
                # Generate
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    with torch.no_grad():
                        gen_output = unwrapped_model.generate(
                            **{k: v for k, v in prompt_inputs.items() 
                               if k not in self.vlm_module.get_non_generate_params()},
                            generation_config=self.generation_config,
                        )
                
                # Decode
                prompt_length = prompt_inputs["input_ids"].size(1)
                if not self.vlm_module.is_embeds_input():
                    completion_ids = gen_output[:, prompt_length:]
                else:
                    completion_ids = gen_output
                
                step_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)[0]
                full_completion_text += step_text
                step_texts_this_sample.append(step_text)
                
                # Append assistant's response to the chat history
                chat.append({"role": "assistant", "content": [{"type": "text", "text": step_text}]})
                
                if self.script_args.single_turn_vision:
                    # Save the complete single turn chat for this step
                    sample_flat_chats.append(list(chat))
                    sample_flat_images.append(list(all_chat_images))
                
                # Parse output
                parsed = _parse_multistep_output(step_text)
                step_record = {
                    "step_index": step + 1,
                    "thinking": parsed.get("thinking", ""),
                    "action_text": parsed.get("action_text", "") or "<invalid>",
                    "is_stop": parsed.get("is_stop", False),
                    "is_unknown": parsed.get("is_unknown", False),
                    "view_image": None,
                    "raw_completion": step_text,
                }
                rollout_record["steps"].append(step_record)
                
                effective_action = parsed["action"]
                if parsed["is_unknown"]:
                    # Unknown is treated as an exploration step: rotate 90 degrees.
                    effective_action = UNKNOWN_FALLBACK_ACTION.copy()

                if DEBUG_MODE and _get_local_rank() == 0:
                    print(
                        f"[rollout] sample={sample_idx} step={step} "
                        f"is_stop={parsed['is_stop']} is_unknown={parsed['is_unknown']} "
                        f"action={parsed['action']} effective_action={effective_action}"
                    )
                
                # STOP is terminal
                if parsed['is_stop']:
                    step_record["termination_reason"] = "model_stop"
                    break
                
                # Check for valid action
                if effective_action is None:
                    # Failed to parse action, treat as stop
                    if DEBUG_MODE and _get_local_rank() == 0:
                        print(f"[rollout] sample={sample_idx} step={step} failed to parse action, stopping")
                    step_record["termination_reason"] = "invalid_or_missing_action"
                    break
                
                # Render new view using the action
                try:
                    new_view, result_metadata = build_additional_view(
                        controller,
                        effective_action,
                        render_metadata=render_metadata,
                    )
                except Exception as e:
                    if DEBUG_MODE and _get_local_rank() == 0:
                        print(f"[rollout] sample={sample_idx} step={step} render failed: {e}")
                    new_view = None
                    result_metadata = {}
                
                if new_view is None:
                    # Rendering failed, stop
                    if DEBUG_MODE and _get_local_rank() == 0:
                        print(f"[rollout] sample={sample_idx} step={step} render returned None, stopping")
                    step_record["termination_reason"] = "render_failed"
                    break
                
                final_image = new_view
                step_record["view_image"] = new_view
                
                # Update render_metadata with new position/rotation for next step
                if result_metadata and result_metadata.get("actual_position"):
                    actual_pos = result_metadata["actual_position"]
                    # Compute new rotation after action
                    rot1 = effective_action[0]
                    rot2 = effective_action[2]
                    rot1_norm = rot1 if rot1 >= 0 else rot1 + 360
                    rot2_norm = rot2 if rot2 >= 0 else rot2 + 360
                    new_yaw = (current_rotation.get("y", 0) + rot1_norm + rot2_norm) % 360
                    
                    render_metadata = dict(render_metadata)  # Copy
                    render_metadata["position"] = actual_pos
                    render_metadata["rotation"] = {"x": 0.0, "y": new_yaw, "z": 0.0}
                    current_position = actual_pos
                    current_rotation = render_metadata["rotation"]
                
                # Update history for the next step
                step_record["termination_reason"] = "continue"
                
                if self.script_args.single_turn_vision:
                    all_chat_images = [new_view]
                    chat = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "text": None},
                                {"type": "text", "text": base_prompt}
                            ]
                        }
                    ]
                else:
                    all_chat_images.append(new_view)
                    chat.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "text": None}
                        ]
                    })
            
            all_completions_text.append(full_completion_text)
            all_final_images.append(final_image)
            all_step_counts.append(step_count)
            all_step_texts_per_sample.append(step_texts_this_sample)
            all_rollout_records.append(rollout_record)
            
            if self.script_args.single_turn_vision:
                trajectory_chat_list.append(sample_flat_chats)
                trajectory_images_list.append(sample_flat_images)
            else:
                trajectory_chat_list.append(chat)
                trajectory_images_list.append(all_chat_images)
        
        if DEBUG_MODE and _get_local_rank() == 0:
            avg_steps = np.mean(all_step_counts)
            print(f"[rollout] Average steps: {avg_steps:.1f}, min: {min(all_step_counts)}, max: {max(all_step_counts)}")
        
        # Format completions for reward functions
        completions = [[{"role": "assistant", "content": ct}] for ct in all_completions_text]
        
        # Compute rewards
        rewards_per_func = torch.zeros(len(inputs), len(self.reward_funcs), device=device)
        verifier_details_out = [{} for _ in range(len(inputs))]
        reward_base_kwargs = {
            "final_images": all_final_images,
            "vqa_question": [x.get("vqa_question", "") for x in inputs],
            "solution": [x.get("solution", "") for x in inputs],
            "question_type": [x.get("question_type", None) for x in inputs],
            "step_counts": all_step_counts,
            "step_texts_per_sample": all_step_texts_per_sample,
            "training_args": self.args,
            "verifier_details_out": verifier_details_out,
        }
        
        for i, reward_func in enumerate(self.reward_funcs):
            if callable(reward_func) and not isinstance(reward_func, torch.nn.Module):
                reward_kwargs = dict(reward_base_kwargs)
                reward_kwargs["device"] = device
                output = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output, dtype=torch.float32, device=device)
        
        # Gather and compute advantages
        rewards_per_func = self.accelerator.gather(rewards_per_func)
        reward_w = torch.as_tensor(self.reward_weights, device=device, dtype=torch.float32)
        rewards = (rewards_per_func * reward_w).sum(dim=1)
        
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4) # Shape: (B,)
        
        process_slice = slice(
            self.accelerator.process_index * len(inputs),
            (self.accelerator.process_index + 1) * len(inputs),
        )
        advantages = advantages[process_slice] # This advantage corresponds to the trajectory
        
        # ================= NEW TRAJECTORY LOGP COMPUTATION (HSTAR STYLE) =================
        # Instead of flattening the steps, we tokenize the ENTIRE multi-turn trajectory per sample!
        if self.script_args.single_turn_vision:
            flat_chat_list = []
            flat_images_list = []
            flat_advantages = []
            for i in range(len(inputs)):
                sample_chats = trajectory_chat_list[i]
                sample_imgs = trajectory_images_list[i]
                for sc, si in zip(sample_chats, sample_imgs):
                    flat_chat_list.append(sc)
                    flat_images_list.append(si)
                    flat_advantages.append(advantages[i])
                    
            traj_input_dicts = [{"prompt": c, "image_path": None} for c in flat_chat_list]
            traj_images_to_use = flat_images_list
            traj_advantages = torch.tensor(flat_advantages, dtype=advantages.dtype, device=device)
        else:
            traj_input_dicts = [{"prompt": chat, "image_path": None} for chat in trajectory_chat_list]
            traj_images_to_use = trajectory_images_list
            traj_advantages = advantages
            
        traj_prompts_text = self.vlm_module.prepare_prompt(self.processing_class, traj_input_dicts)
        
        traj_inputs, _ = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            traj_prompts_text,
            traj_images_to_use,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        traj_inputs = super(VLMGRPOTrainer, self)._prepare_inputs(traj_inputs)
        traj_input_ids = traj_inputs["input_ids"]
        traj_attention_mask = traj_inputs["attention_mask"]

        # Search for assistant turns to build completion_mask
        start_marker = self.processing_class.tokenizer.encode("<|im_start|>assistant\n")
        end_marker   = self.processing_class.tokenizer.encode("<|im_end|>\n")
        fallback_start = self.processing_class.tokenizer.encode("<|im_start|>")
        fallback_end = self.processing_class.tokenizer.encode("<|im_end|>")

        traj_completion_mask = torch.zeros_like(traj_attention_mask)
        for b in range(traj_input_ids.size(0)):
            ids = traj_input_ids[b].tolist()
            mask = [0]*len(ids)
            in_assistant = False
            
            i = 0
            while i < len(ids):
                if not in_assistant:
                    if ids[i:i+len(start_marker)] == start_marker:
                        in_assistant = True
                        i += len(start_marker)
                    else:
                        i += 1
                else:
                    if ids[i:i+len(end_marker)] == end_marker:
                        in_assistant = False
                        mask[i] = 1  # mask the <|im_end|> token
                        i += len(end_marker)
                    elif ids[i:i+len(fallback_end)] == fallback_end:
                        in_assistant = False
                        mask[i] = 1 # mask the <|im_end|> token
                        i += len(fallback_end)
                    else:
                        mask[i] = 1
                        i += 1
            
            traj_completion_mask[b] = torch.tensor(mask, dtype=traj_completion_mask.dtype, device=device)
        
        traj_completion_mask = traj_completion_mask * traj_attention_mask
        
        # Get multimodal inputs
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        traj_multimodal_inputs = {k: traj_inputs[k] if k in traj_inputs else None for k in multimodal_keywords}
        
        # Compute trajectory logps
        with torch.no_grad():
            if self.num_iterations > 1:
                traj_old_per_token_logps = self._get_per_token_logps(
                    model, traj_input_ids, traj_attention_mask, **traj_multimodal_inputs
                )
            else:
                traj_old_per_token_logps = None
            
            if self.beta == 0.0:
                traj_ref_per_token_logps = None
            elif self.ref_model is not None:
                traj_ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, traj_input_ids, traj_attention_mask, **traj_multimodal_inputs
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    traj_ref_per_token_logps = self._get_per_token_logps(
                        model, traj_input_ids, traj_attention_mask, **traj_multimodal_inputs
                    )

        # Log metrics
        completion_length = self.accelerator.gather_for_metrics(traj_completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        self._metrics["avg_rollout_steps"].append(np.mean(all_step_counts))
        
        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, torch.nn.Module):
                name = reward_func.config._name_or_path.split("/")[-1]
            else:
                name = reward_func.__name__
            self._metrics[f"rewards/{name}"].append(reward_per_func[i].item())
        
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        self._save_rollout_visualizations(all_rollout_records, verifier_details_out)
        
        return {
            "input_ids": traj_input_ids,
            "attention_mask": traj_attention_mask,
            "completion_mask": traj_completion_mask,
            "old_per_token_logps": traj_old_per_token_logps,
            "ref_per_token_logps": traj_ref_per_token_logps,
            "advantages": traj_advantages,
            "multimodal_inputs": traj_multimodal_inputs,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
    
        # Check if we need to generate new completions or use buffered ones
        # MultiStepGRPOTrainer uses the same buffering system 
        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs, model)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1

        # Get the prepared inputs (now full trajectories)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"]
        multimodal_inputs = inputs["multimodal_inputs"]
        
        # Get the current policy's log probabilities on the trajectory
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, **multimodal_inputs)
        
        # Logps are shifted by 1 in TRL's _get_per_token_logps, so they have length seq_len - 1.
        # We align the completion mask directly with logps by dropping the first element.
        completion_mask = completion_mask[:, 1:]

        # Get the advantages from inputs (pre-scaled by step weights)
        advantages = inputs["advantages"]

        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        # Compute the policy ratio and clipped version
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Add KL penalty if beta > 0
        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl

            # Log KL divergence
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # Compute final loss over tokens, then sum over samples.
        loss_per_sequence = (per_token_loss * completion_mask).sum(dim=1) / (completion_mask.sum(dim=1) + 1e-8)
        loss = loss_per_sequence.mean()

        # Log clip ratio
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / (completion_mask.sum() + 1e-8)
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        return loss

def main(script_args, training_args, model_args):
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    if DEBUG_MODE and local_rank == 0:
        print("Using VLM module:", vlm_module_cls.__name__)
    
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    reward_weights = script_args.grpo_reward_weights
    assert len(reward_funcs) == len(reward_weights), "Length of reward_funcs and reward_weights must match"
    
    # ====== DATA PREPROCESSING ======
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    def make_conversation_from_jsonl(example, image_folder: str):
        required_fields = ["question", "answer", "steps"]
        for f in required_fields:
            if f not in example:
                raise ValueError(f"Field {f} is required in the example")
        
        steps = example["steps"]
        if not steps:
            raise ValueError("Steps list is empty")
            
        first_step = steps[0]
        last_step = steps[-1]
        
        # Resolve image paths
        img_path = first_step.get("view_image")
        if not img_path:
            raise ValueError("First step missing view_image")
            
        if os.path.isabs(img_path):
            image_paths = [img_path]
        else:
            image_paths = [os.path.join(image_folder, img_path)]
        
        # Build action question
        if script_args.single_turn_vision:
            action_question = (
                SINGLE_TURN_MULTISTEP_ACTION_PROMPT_TEMPLATE.format(question=example["question"])
                + MULTISTEP_FORMAT_PROMPT
            )
        else:
            action_question = (
                MULTISTEP_ACTION_PROMPT_TEMPLATE.format(question=example["question"])
                + MULTISTEP_FORMAT_PROMPT
            )
        
        # Camera pose
        position_dict = first_step.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
        rotation_scalar = first_step.get("rotation", 0.0)
        render_position = {
            "x": float(position_dict["x"]),
            "y": float(position_dict["y"]),
            "z": float(position_dict["z"]),
        }
        render_rotation = {"x": 0.0, "y": float(rotation_scalar), "z": 0.0}
        
        gt_position_dict = last_step.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
        gt_rotation_scalar = last_step.get("rotation", 0.0)
        gt_position = {
            "x": float(gt_position_dict["x"]),
            "y": float(gt_position_dict["y"]),
            "z": float(gt_position_dict["z"]),
        }
        gt_rotation = {"x": 0.0, "y": float(gt_rotation_scalar), "z": 0.0}
        
        # Derive scene_index
        scene_id = example.get("scene_id", example.get("house_id", "house_00000"))
        scene_match = re.search(r"house_(\d+)", scene_id)
        scene_index = int(scene_match.group(1)) if scene_match else 0
        
        render_metadata = {
            "position": render_position,
            "rotation": render_rotation,
            "gt_position": gt_position,
            "gt_rotation": gt_rotation,
            "scene_index": scene_index,
            "trans_scale": 100.0,
            "check_obj_existence": False,
            "pixel_threshold": 200,
            "use_fallback": script_args.use_fallback,
            "num_grids": script_args.fallback_num_grids,
        }
        
        if "question_type" in example:
            render_metadata["question_type"] = example["question_type"]
        
        if script_args.customized_scene_path is not None:
            custom_house_path = {
                "train": script_args.customized_scene_path,
                "val": None,
                "test": None,
            }
        else:
            custom_house_path = None
        render_metadata["custom_house_path"] = custom_house_path
        
        assert all(os.path.exists(p) for p in image_paths), f"Image paths do not exist: {image_paths}"
        
        question_type = example.get("question_type", None)
        target_visibility = example.get("target_object_visibility_level", "unknown")
        
        num_steps = len(steps)
        if num_steps <= 2:
            turn_type = "single-turn"
        else:
            turn_type = "multi-turn"
        
        return {
            "image_path": image_paths,
            "action_question": action_question,
            "vqa_question": example["question"],
            "gt_action": first_step.get("action", [0, 0, 0]) or [0, 0, 0],
            "render_metadata": render_metadata,
            "solution": f"<answer> {example['answer']} </answer>",
            "question_type": question_type,
            "target_visibility": target_visibility,
            "turn_type": turn_type,
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        *({"type": "image", "text": None} for _ in range(len(image_paths))),
                        {"type": "text", "text": action_question},
                    ],
                }
            ],
        }
    
    all_data = []
    for data_file, image_folder in zip(data_files, image_folders):
        with open(data_file, "r") as f:
            for line in f:
                item = json.loads(line)
                try:
                    transformed = make_conversation_from_jsonl(item, image_folder)
                    all_data.append(transformed)
                except Exception as e:
                    if DEBUG_MODE and local_rank == 0:
                        print(f"Warning: skip item: {e}")
    
    print(f"Loaded {len(all_data)} training examples")
    dataset = Dataset.from_list(all_data)
    
    # Split dataset
    splits = {"train": dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio, seed=script_args.val_split_seed
        )
        splits["train"] = train_val_split["train"]
        splits["validation"] = train_val_split["test"]
    
    # Initialize trainer
    trainer = MultiStepGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=splits["train"],
        eval_dataset=(
            splits.get("validation") if training_args.eval_strategy != "no" else None
        ),
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        max_rollout_steps=script_args.max_rollout_steps,
        script_args=script_args,
    )
    
    # Train
    checkpoint_dirs = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoint_dirs:
        def get_checkpoint_number(path):
            try:
                return int(path.name.split("-")[1])
            except (IndexError, ValueError):
                return 0
        latest_checkpoint = max(checkpoint_dirs, key=get_checkpoint_number)
        print(f"Resuming from checkpoint from {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("Training from scratch")
        trainer.train()
    
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
