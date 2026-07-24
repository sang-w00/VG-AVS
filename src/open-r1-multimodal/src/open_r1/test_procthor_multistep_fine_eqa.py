import argparse
import json
import os
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

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
from open_r1.utils.visualization import create_multistep_rollout_visualization
from open_r1.utils.rewards import _verifier_answer
from open_r1.utils.prompt_templates import (
    MULTISTEP_ACTION_PROMPT_TEMPLATE,
    MULTISTEP_FORMAT_PROMPT,
)
from open_r1.grpo_multistep import _parse_multistep_output
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm



import traceback
from collections import deque
import numpy as np
from PIL import ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from open_r1.utils.gemini_utils import parse_yes_no

CATEGORY_ORDER = ["easy", "medium", "multistep easy", "multistep medium", "undecidable"]


def _normalize_visibility_level(value: Any) -> str:
    if not isinstance(value, str):
        return "undecidable"

    level = value.strip().lower()
    if level in {"undecidable", "unknown"}:
        return "undecidable"
    if level in {"medium"}:
        return "medium"
    if level in {"easy", "visible"}:
        return "easy"
    return "undecidable"


def _categorize_sample(target_visibility_level: Any, num_steps: int) -> str:
    normalized = _normalize_visibility_level(target_visibility_level)
    if normalized == "undecidable":
        return "undecidable"

    is_multistep = int(num_steps) >= 3
    if normalized == "medium":
        return "multistep medium" if is_multistep else "medium"
    return "multistep easy" if is_multistep else "easy"


def _compute_category_stats(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for category in CATEGORY_ORDER:
        cat_results = [r for r in results if r.get("eval_category") == category]
        sample_count = len(cat_results)
        correct_count = sum(1 for r in cat_results if r.get("correct"))
        acc = (correct_count / sample_count * 100.0) if sample_count > 0 else 0.0

        vis_values = [
            float(r["target_object_visibility_gen_over_gt"])
            for r in cat_results
            if isinstance(r.get("target_object_visibility_gen_over_gt"), (int, float))
        ]
        avg_visibility = (sum(vis_values) / len(vis_values)) if vis_values else None

        stats[category] = {
            "sample_count": sample_count,
            "correct": correct_count,
            "accuracy": acc,
            "avg_visibility": avg_visibility,
            "visibility_count": len(vis_values),
        }
    return stats


@dataclass
class FineEQAConfig:
    """Configuration for Fine-EQA method (GOE with FBE semantic map)"""
    # Semantic map parameters (from FBE)
    gsv_T: float = 0.5  # Temperature for GSV computation
    gsv_F: float = 3.0  # Factor for GSV normalization
    smooth_sigma: float = 5.0  # Gaussian smoothing for semantic maps
    semantic_integration_weight: float = 1.0  # Weight for semantic value integration
    semantic_radius: float = 1.0  # Radius for semantic value update (meters)
    
    # Visual prompt parameters (for farthest point sampling)
    num_prompt_points: int = 3  # Number of sampled points for VLM
    min_num_prompt_points: int = 2  # Minimum points required
    circle_radius: int = 18  # Circle radius for visual markers
    
    # Goal-Oriented Exploration parameters
    region_confidence_threshold: float = 0.0  # Keep only non-unknown region (closer to EXPRESS-Bench behavior)
    region_neighborhood_radius: float = 1.0  # Radius for region update (meters)
    visited_point_decay: float = 0.3  # Decay factor for visited points
    exploration_radius: float = 0.5  # Radius to mark as explored (meters)
    max_exp: int = 3  # GOE exploration cycle count (matches EXPRESS-Bench default)
    goe_num_regions: int = 2  # EXPRESS-Bench GOE state machine handles top-2 relevant regions
    region_merge_para: float = 3.0  # Merge radius multiplier for nearby representative points
    
    # Action prediction parameters
    max_forward_distance: float = 300.0  # Max forward movement (cm), aligned with EXPRESS-Bench 3m bound
    rotation_step: float = 30.0  # Rotation step size (degrees)
    
    # Region definitions
    regions_dict: Dict[str, str] = None
    
    # GPT API settings
    gpt_model: str = "gpt-5-mini"  # GPT model for region/termination checks
    
    def __post_init__(self):
        if self.regions_dict is None:
            self.regions_dict = {
                'A': "bathroom", 'B': "bedroom", 'C': "dining room", 
                'D': "garage", 'E': "kitchen", 'F': "laundry room", 
                "G": "living room", "H": "office", "I": "rec room", 
                "J": "study", "K": "hallway", "L": "entryway", 
                "M": "laboratory", "N": "workout room", "O": "warehouse", 
                "P": "lounge", "Q": "balcony", "R": "staircase", 
                "S": "cloakroom", "T": "unknown"
            }


class SimpleSemanticMap:
    """Semantic map for GOE with FBE-style semantic value computation"""
    
    def __init__(self, map_size: Tuple[int, int] = (200, 200), resolution: float = 0.1):
        self.map_size = map_size
        self.resolution = resolution
        
        # Initialize maps as described in paper
        # Msem: Global semantic map (constructed from vl and vg)
        self.semantic_value_map = np.zeros(map_size, dtype=np.float32)
        self.semantic_weight_map = np.zeros(map_size, dtype=np.float32)  # Weight for cumulative averaging
        
        # Mreg: Functional region semantic map (stores region IDs)
        self.region_map = -np.ones(map_size, dtype=np.int32)
        
        # Mmasked: Masked semantic map for current priority region
        self.masked_semantic_map = np.zeros(map_size, dtype=np.float32)
        
        # Tracking maps
        self.visited_map = np.zeros(map_size, dtype=np.bool_)
        self.exploration_count_map = np.zeros(map_size, dtype=np.int32)
        
        # GOE state
        self.prioritized_regions = []  # Ordered by priority (highest first)
        self.current_priority_region = None
        self.region_points = {}  # Maps region_id -> list of representative points
        self.explored_points = []  # Track explored positions
        
        # For semantic integration (stores current candidates)
        self.candidates = []  # Current sampled points for semantic update
        
    def world_to_grid(self, x: float, z: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x / self.resolution) + self.map_size[0] // 2)
        grid_z = int((z / self.resolution) + self.map_size[1] // 2)
        grid_x = np.clip(grid_x, 0, self.map_size[0] - 1)
        grid_z = np.clip(grid_z, 0, self.map_size[1] - 1)
        return grid_x, grid_z
    
    def grid_to_world(self, grid_x: int, grid_z: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        x = (grid_x - self.map_size[0] // 2) * self.resolution
        z = (grid_z - self.map_size[1] // 2) * self.resolution
        return x, z
    
    def points_in_circle(self, center_x: int, center_z: int, radius: int) -> List[Tuple[int, int]]:
        """Get all grid points within a circle"""
        points = []
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx**2 + dz**2 <= radius**2:
                    gx, gz = center_x + dx, center_z + dz
                    if 0 <= gx < self.map_size[0] and 0 <= gz < self.map_size[1]:
                        points.append((gx, gz))
        return points
    
    def integrate_semantic_values(self, sampled_points: List[Tuple[float, float]], 
                                  semantic_values: np.ndarray, radius: float = 1.0,
                                  obs_weight: float = 1.0):
        """
        Integrate semantic values into Msem using weighted fusion.
        Equation: Msem ← (psample, vl, vg)
        
        Args:
            sampled_points: List of (x, z) world coordinates
            semantic_values: Array of semantic values (SV = vl * vg) for each point
            radius: Integration radius in meters
            obs_weight: Observation weight for cumulative averaging
        """
        assert len(sampled_points) == len(semantic_values), \
            f"Mismatch: {len(sampled_points)} points vs {len(semantic_values)} values"
        
        self.candidates = sampled_points  # Store for current iteration
        
        radius_grid = int(radius / self.resolution)
        
        for p_ind, (x, z) in enumerate(sampled_points):
            grid_x, grid_z = self.world_to_grid(x, z)
            sv = semantic_values[p_ind]
            
            # Get points in circle around this sample
            pts = self.points_in_circle(grid_x, grid_z, radius_grid)
            
            for gx, gz in pts:
                # Cumulative moving average
                w_old = self.semantic_weight_map[gx, gz]
                self.semantic_weight_map[gx, gz] += obs_weight
                self.semantic_value_map[gx, gz] = (
                    w_old * self.semantic_value_map[gx, gz] + obs_weight * sv
                ) / self.semantic_weight_map[gx, gz]
    
    def smooth_semantic_map(self, sigma: float = 5.0):
        """Apply Gaussian smoothing to Msem"""
        if sigma and sigma > 0:
            self.semantic_value_map = gaussian_filter(self.semantic_value_map, sigma=sigma)
    
    def update_region_map(self, x: float, z: float, region_id: int, radius: float = 1.0):
        """
        Update functional region semantic map Mreg around representative point q.
        Equation (9): Mreg(N(q)) = IDReg
        """
        grid_x, grid_z = self.world_to_grid(x, z)
        radius_grid = int(radius / self.resolution)
        
        # Update neighborhood N(q)
        for dx in range(-radius_grid, radius_grid + 1):
            for dz in range(-radius_grid, radius_grid + 1):
                gx, gz = grid_x + dx, grid_z + dz
                if 0 <= gx < self.map_size[0] and 0 <= gz < self.map_size[1]:
                    dist = np.sqrt(dx**2 + dz**2)
                    if dist <= radius_grid:
                        self.region_map[gx, gz] = region_id
        
        # Store representative point for this region
        if region_id not in self.region_points:
            self.region_points[region_id] = []
        self.region_points[region_id].append((x, z))
    
    def _bresenham_line(self, x1: int, z1: int, x2: int, z2: int) -> List[Tuple[int, int]]:
        """Integer grid line between two points."""
        points = []
        dx = abs(x2 - x1)
        dz = abs(z2 - z1)
        sx = 1 if x1 < x2 else -1
        sz = 1 if z1 < z2 else -1
        err = dx - dz
        while True:
            points.append((x1, z1))
            if x1 == x2 and z1 == z2:
                break
            e2 = 2 * err
            if e2 > -dz:
                err -= dz
                x1 += sx
            if e2 < dx:
                err += dx
                z1 += sz
        return points

    def merge_adjacent_regions(self, region_id: int, merge_para: float = 3.0, radius: float = 1.0):
        """
        Merge region points with spatial proximity, similar to EXPRESS-Bench modify_map().
        """
        if region_id not in self.region_points or len(self.region_points[region_id]) < 2:
            return

        points = self.region_points[region_id]
        radius_grid = max(int(radius / self.resolution), 1)

        for i, (x1, z1) in enumerate(points):
            for j, (x2, z2) in enumerate(points[i+1:], start=i+1):
                gx1, gz1 = self.world_to_grid(x1, z1)
                gx2, gz2 = self.world_to_grid(x2, z2)
                d_grid = float(np.hypot(gx2 - gx1, gz2 - gz1))
                if d_grid <= merge_para * radius_grid:
                    for gx, gz in self._bresenham_line(gx1, gz1, gx2, gz2):
                        if 0 <= gx < self.map_size[0] and 0 <= gz < self.map_size[1]:
                            self.region_map[gx, gz] = region_id
                            for cx, cz in self.points_in_circle(gx, gz, radius_grid):
                                self.region_map[cx, cz] = region_id

    def get_masked_semantic_map(self, priority_region_id: int, visited_decay: float = 0.3, 
                                 smooth_sigma: float = 5.0, explored_zero_radius: float = 0.5) -> np.ndarray:
        """
        Apply masking operations to focus on high-priority regions.
        Equation (10): Mmasked = φ(Msem, Mreg, r)
        
        Args:
            priority_region_id: The region ID with current highest priority
            visited_decay: Decay factor for previously visited points
            smooth_sigma: Gaussian smoothing parameter
        
        Returns:
            Mmasked: Masked semantic map for the priority region
        """
        # Create mask for priority region
        region_mask = (self.region_map == priority_region_id).astype(np.float32)
        
        # Apply mask to global semantic map
        masked_map = self.semantic_value_map * region_mask
        
        # Zero-out previously explored neighborhoods (closer to EXPRESS-Bench behavior)
        radius_grid = max(int(explored_zero_radius / self.resolution), 0)
        for grid_x, grid_z in self.explored_points:
            if 0 <= grid_x < self.map_size[0] and 0 <= grid_z < self.map_size[1]:
                for gx, gz in self.points_in_circle(grid_x, grid_z, radius_grid):
                    masked_map[gx, gz] = 0.0
        
        # Apply Gaussian smoothing
        if smooth_sigma > 0:
            masked_map = gaussian_filter(masked_map, sigma=smooth_sigma)
        
        self.masked_semantic_map = masked_map
        return masked_map
    
    def get_highest_value_position(self) -> Optional[Tuple[float, float]]:
        """
        Select position χ = argmax(x,y)(Mmasked) with highest semantic value.
        Returns world coordinates.
        """
        if self.masked_semantic_map.sum() == 0:
            return None
        
        # Find position with maximum value
        max_idx = np.argmax(self.masked_semantic_map)
        grid_x, grid_z = np.unravel_index(max_idx, self.masked_semantic_map.shape)
        
        # Convert to world coordinates
        world_x, world_z = self.grid_to_world(grid_x, grid_z)
        
        # Mark as explored
        self.explored_points.append((grid_x, grid_z))
        
        return world_x, world_z
    
    def mark_visited(self, x: float, z: float, radius: float = 0.3):
        """Mark a location as visited"""
        grid_x, grid_z = self.world_to_grid(x, z)
        radius_grid = int(radius / self.resolution)
        
        for dx in range(-radius_grid, radius_grid + 1):
            for dz in range(-radius_grid, radius_grid + 1):
                gx, gz = grid_x + dx, grid_z + dz
                if 0 <= gx < self.map_size[0] and 0 <= gz < self.map_size[1]:
                    dist = np.sqrt(dx**2 + dz**2)
                    if dist <= radius_grid:
                        self.visited_map[gx, gz] = True
                        self.exploration_count_map[gx, gz] += 1


class FineEQAActionPredictor:
    """
    Fine-EQA action predictor using Goal-Oriented Exploration (GOE) only.
    
    Implements the GOE strategy from the paper:
    1. Functional Region Semantic Mapping (Mreg)
    2. Task-Relevant Region Prioritization (LLM-based)
    3. Masked Semantic Mapping (Mmasked = φ(Msem, Mreg, r))
    4. Position selection: χ = argmax(x,y)(Mmasked)
    """
    
    def __init__(
        self,
        vlm_model,
        vlm_processor,
        config: FineEQAConfig,
        device: str = "cuda:0",
        controller=None,
        use_prismatic: bool = False,
    ):
        self.vlm_model = vlm_model
        self.vlm_processor = vlm_processor
        self.config = config
        self.device = device
        self.controller = controller
        self.use_prismatic = bool(use_prismatic)

        # Resolve backend defensively to avoid NoneType processor failures.
        if self.use_prismatic and not self._can_use_prismatic_backend() and self._can_use_hf_backend():
            print("[Fine-EQA] use_prismatic=True but model is not prismatic; falling back to HF processor backend.")
            self.use_prismatic = False
        elif not self.use_prismatic and not self._can_use_hf_backend() and self._can_use_prismatic_backend():
            print("[Fine-EQA] HF processor backend unavailable; auto-switching to prismatic backend.")
            self.use_prismatic = True
        elif not self._can_use_hf_backend() and not self._can_use_prismatic_backend():
            print("[Fine-EQA] Warning: no usable VLM backend (HF processor/prismatic). Some scores may fallback.")
        
        # Initialize semantic map with Msem, Mreg, Mmasked
        self.semantic_map = SimpleSemanticMap()
        
        # Region ID mapping (string name -> int ID)
        self.region_name_to_id = {name: idx for idx, name in enumerate(config.regions_dict.values())}
        self.region_id_to_name = {idx: name for name, idx in self.region_name_to_id.items()}
        self.region_key_to_name = dict(config.regions_dict)
        self.region_name_to_key = {v: k for k, v in self.region_key_to_name.items()}
        
        # History tracking
        self.observation_history = deque(maxlen=10)
        self.position_history = deque(maxlen=10)
        
        # GOE state
        self.prioritized_regions = []  # List of (region_name, priority_score)
        self.current_priority_region = None
        self.exp_fbe_state = True  # True-FBE, False-GOE (same convention as EXPRESS-Bench)
        self.exp_list: List[Dict[str, Any]] = []
        self.last_region_key: str = "T"

    def _can_use_hf_backend(self) -> bool:
        return self.vlm_processor is not None and hasattr(self.vlm_processor, "apply_chat_template")

    def _can_use_prismatic_backend(self) -> bool:
        return (
            self.vlm_model is not None
            and hasattr(self.vlm_model, "get_prompt_builder")
            and hasattr(self.vlm_model, "llm_backbone")
        )

    def _use_prismatic_backend(self) -> bool:
        return self.use_prismatic and self._can_use_prismatic_backend()

    def _init_goe_state(self):
        """Initialize GOE state machine using top-k prioritized regions."""
        candidates = []
        for region_name, _ in self.prioritized_regions:
            if region_name != "unknown" and region_name not in candidates:
                candidates.append(region_name)
            if len(candidates) >= max(1, int(self.config.goe_num_regions)):
                break
        self.exp_list = [{"region": r, "count": 0, "dir": 0, "point": []} for r in candidates]
        self.exp_fbe_state = True

    def _reset_exp(self, index: int):
        if 0 <= index < len(self.exp_list):
            self.exp_list[index] = {
                "region": self.exp_list[index]["region"],
                "count": 0,
                "dir": 0,
                "point": [],
            }

    def _has_available_points(self, index: int) -> bool:
        return (
            0 <= index < len(self.exp_list)
            and self.exp_list[index]["count"] == 0
            and len(self.exp_list[index]["point"]) > 0
        )

    def _is_exploration_complete(self, index: int) -> bool:
        return (
            0 <= index < len(self.exp_list)
            and self.exp_list[index]["count"] >= int(self.config.max_exp)
            and self.exp_list[index]["dir"] == 3
        )

    def _advance_exploration(self, index: int):
        if not (0 <= index < len(self.exp_list)):
            return
        if self.exp_list[index]["dir"] == 3:
            self.exp_list[index]["count"] += 1
            self.exp_list[index]["dir"] = 0
        else:
            self.exp_list[index]["dir"] += 1

    def _transition_to_goe(self, index: int):
        if not (0 <= index < len(self.exp_list)):
            return
        self.exp_fbe_state = False
        self.exp_list[index]["count"] = 1

    def _update_goe_state(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Update exploration status with EXPRESS-Bench-like state transitions.
        Returns:
            (is_fbe_state, exp_list)
        """
        if not self.exp_list:
            return True, self.exp_list

        if self.exp_fbe_state:
            if self._has_available_points(0):
                self._transition_to_goe(0)
            elif len(self.exp_list) > 1 and self._has_available_points(1):
                self._transition_to_goe(1)
            return self.exp_fbe_state, self.exp_list

        # GOE state
        if len(self.exp_list) == 1:
            if self._is_exploration_complete(0):
                self._reset_exp(0)
                self.exp_fbe_state = True
            else:
                self._advance_exploration(0)
            return self.exp_fbe_state, self.exp_list

        # Two-region logic (same design as EXPRESS-Bench)
        if self.exp_list[0]["count"] > 0:
            if self._is_exploration_complete(0):
                self._reset_exp(0)
                if self._has_available_points(1):
                    self.exp_list[1]["count"] = 1
                else:
                    self.exp_fbe_state = True
            else:
                self._advance_exploration(0)
        else:
            if self._has_available_points(0):
                self._reset_exp(1)
                self.exp_list[0]["count"] = 1
            elif self._is_exploration_complete(1):
                self._reset_exp(1)
                self.exp_fbe_state = True
            else:
                self._advance_exploration(1)
        return self.exp_fbe_state, self.exp_list

    def _active_goe_index(self) -> Optional[int]:
        if not self.exp_list:
            return None
        counts = [entry["count"] for entry in self.exp_list]
        if max(counts) <= 0:
            return None
        return int(np.argmax(counts))

    def _enqueue_region_point(self, region_name: str, world_point: Tuple[float, float], radius: float):
        """Queue representative point for region-guided GOE step selection."""
        if not self.exp_list:
            return
        rel_region = [entry["region"] for entry in self.exp_list]
        if region_name not in rel_region:
            return
        gx, gz = self.semantic_map.world_to_grid(world_point[0], world_point[1])
        radius_vox = max(int(radius / self.semantic_map.resolution), 1)
        is_near_existing = any(
            np.hypot(px - gx, pz - gz) < (radius_vox / 2.0)
            for px, pz in self.semantic_map.explored_points
        )
        if not is_near_existing:
            idx = rel_region.index(region_name)
            self.exp_list[idx]["point"].append((gx, gz))
        
    def identify_task_relevant_regions(self, question: str) -> List[str]:
        """
        Identify task-relevant regions for the question using VLM.
        Adapted from Fine-EQA's region identification.
        """
        # Create prompt for region identification
        regions_list = list(self.config.regions_dict.values())
        regions_str = ", ".join(regions_list)
        
        prompt = f"""To answer the question, determine the most relevant regions.
Question: {question}

Available regions: {regions_str}

Output only the region name(s), separated by commas if multiple. 
If multiple regions, prioritize by relevance."""
        
        try:
            if self._use_prismatic_backend():
                from PIL import Image as _PILImage
                dummy_image = _PILImage.new("RGB", (2, 2), color=(255, 255, 255))
                prompt_builder = self.vlm_model.get_prompt_builder()
                prompt_builder.add_turn(role="human", message=prompt)
                prompt_text = prompt_builder.get_prompt()
                output = self.vlm_model.generate(
                    dummy_image,
                    prompt_text,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=50,
                    min_length=1,
                )
            elif self._can_use_hf_backend():
                messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }]
                
                chat_text = self.vlm_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                inputs = self.vlm_processor(
                    text=[chat_text],
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                
                gen = self.vlm_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.vlm_processor.tokenizer.eos_token_id,
                )
                
                output = self.vlm_processor.batch_decode(gen, skip_special_tokens=True)[0]
            else:
                raise RuntimeError(
                    "No valid backend for region identification: missing HF processor and prismatic model API."
                )
            
            # Parse regions from output
            identified_regions = []
            for region in regions_list:
                if region.lower() in output.lower():
                    identified_regions.append(region)
            
            return identified_regions if identified_regions else ["unknown"]
            
        except Exception as e:
            print(f"Error in region identification: {e}")
            return ["unknown"]
    
    def prioritize_task_relevant_regions(self, question: str) -> List[Tuple[str, float]]:
        """
        Task-Relevant Region Prioritization using EXPRESS-Bench's region prompt flow.
        - Prefer `prompt/region.txt` and preserve [system]/[user] separation.
        - Append `QUESTION ... REGION:` suffix as in original main.py.
        - Parse comma-prioritized regions; tolerate minor punctuation/line-break deviations.
        Returns a list of (region_name, priority_score) with 0 as highest priority.
        """
        regions_list = list(self.config.regions_dict.values())
        regions_set = {r.lower(): r for r in regions_list}

        # Resolve region prompt path (prefer env, then common EXPRESS-Bench paths)
        prompt_candidates = []
        env_prompt = os.environ.get("FINEEQA_REGION_PROMPT")
        if env_prompt:
            prompt_candidates.append(env_prompt)
        prompt_candidates.extend(
            [
                "/home/andy2884/workspace/EXPRESS-Bench/prompt/region.txt",
                "/data/andy2884/workspace/EXPRESS-Bench/prompt/region.txt",
            ]
        )

        prompt_path = None
        prompt_lines: List[str] = []
        for cand in prompt_candidates:
            if cand and os.path.exists(cand):
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        prompt_lines = [ln.rstrip("\n") for ln in f.readlines()]
                    prompt_path = cand
                    break
                except Exception:
                    continue

        system_prompt = "Your task is to determine the regions relevant to the question."
        user_prompt = (
            "To answer the question, it is necessary to determine the most relevant region for the question, "
            "facilitating navigation to the corresponding region in the scene to gather the required information.\n"
            "Additionally, based on common household layouts, if this most relevant region is typically located near "
            "another region, the nearby region should also be included in the output. If no such nearby region exists, "
            "only the most relevant region needs to be output.\n"
            "If there are multiple relevant regions, please prioritize them in descending order of relevance.\n"
            f"All possible regions are shown in the array: {regions_list}."
        )

        # Mirror EXPRESS-Bench prompt_make behavior: line 2 as system, lines 4+ as user body.
        if len(prompt_lines) >= 4 and prompt_lines[0].strip().lower() == "[system]" and prompt_lines[2].strip().lower() == "[user]":
            system_prompt = prompt_lines[1].strip() or system_prompt
            user_prompt_raw = "\n".join(prompt_lines[3:]).strip()
            if user_prompt_raw:
                user_prompt = user_prompt_raw

        ex_prompt = f"QUESTION: {question}\nREGION: "
        print(
            f"[Fine-EQA] Region prompt path={prompt_path}, loaded={bool(prompt_path)}, "
            f"system_len={len(system_prompt)}, user_len={len(user_prompt)}"
        )

        ordered: List[str] = []
        try:
            model_id = _resolve_gpt_model_id(getattr(self.config, "gpt_model", "gpt-5-mini"))
            print(f"[Fine-EQA] Region prioritization model_id={model_id}")
            llm_output = run_gpt_action_prediction(
                image=None,
                prompt=None,
                model_id=model_id,
                messages=[
                    {"role": "system", "text": system_prompt},
                    {"role": "user", "text": f"{user_prompt}\n{ex_prompt}"},
                ],
            )
            if llm_output.strip().startswith("ERROR:"):
                raise RuntimeError(llm_output.strip())
            print(f"[Fine-EQA] Region prioritization raw_output={llm_output[:200].strip()}...")

            # 1) Original-style comma parsing (with minor normalization).
            normalized = llm_output.replace("\n", ",")
            parts = [p.strip() for p in re.split(r"[,;/|]", normalized) if p.strip()]
            for p in parts:
                key = p.lower().strip()
                key = re.sub(r"^[\s\-\d\)\.\:]+", "", key)
                key = re.sub(r"[\s\.\!\?]+$", "", key)
                if key in regions_set:
                    reg = regions_set[key]
                    if reg not in ordered:
                        ordered.append(reg)

            # 2) If strict parse fails, recover by scanning region names in first-appearance order.
            if not ordered:
                text_lower = llm_output.lower()
                hits = []
                for reg in regions_list:
                    idx = text_lower.find(reg.lower())
                    if idx >= 0:
                        hits.append((idx, reg))
                hits.sort(key=lambda x: x[0])
                for _, reg in hits:
                    if reg not in ordered:
                        ordered.append(reg)
            print(f"[Fine-EQA] Region prioritization parsed={ordered}")
        except Exception as e:
            print(f"[Fine-EQA] Region prioritization LLM call failed: {e}; will fallback")
            ordered = []
        
        if not ordered:
            seed_regions = self.identify_task_relevant_regions(question)
            ordered = [r for r in seed_regions if r != "unknown"]
            print(f"[Fine-EQA] Region prioritization fallback seed_regions={ordered}")
        if not ordered:
            ordered = ["unknown"]
            print(f"[Fine-EQA] Region prioritization final fallback=['unknown']")
        return [(reg, float(i)) for i, reg in enumerate(ordered)]
    
    def farthest_point_sampling(self, image: Image.Image, render_metadata: Dict,
                                 num_points: int = 3) -> List[Tuple[float, float, Tuple[int, int]]]:
        """
        Depth-based farthest point sampling using ProcTHOR depth image and FOV.
        - Teleport to current pose
        - Read event.depth_frame
        - Sample candidate columns across width; compute depth at center row
        - Back-project using yaw + per-column angle (FOV=90 by default)
        - Perform greedy FPS in world (x,z) space
        Returns list of (world_x, world_z, (pixel_x, pixel_y))
        """
        # Ensure controller
        from open_r1.utils.procthor_utils import _safe_controller_step, _is_controller_alive, get_procthor_controller
        controller = self.controller
        if controller is None or not _is_controller_alive(controller):
            controller = get_procthor_controller(headless=True)
            self.controller = controller
        print("[Fine-EQA] FPS: controller initialized/ready")
        
        # Reset scene and teleport to current pose
        current_position = render_metadata.get("position")
        current_rotation = render_metadata.get("rotation")
        scene_index = render_metadata.get("scene_index")
        custom_house_path = render_metadata.get("custom_house_path", None)
        split = render_metadata.get("data_split", "train")
        try:
            if scene_index is not None:
                # Reset scene
                from open_r1.utils.procthor_utils import get_procthor_house, _safe_controller_reset
                house = get_procthor_house(custom_house_path=custom_house_path, house_index=scene_index, split=split)
                _safe_controller_reset(controller, scene=house)
            print(f"[Fine-EQA] FPS: Teleport → pos={current_position}, rot={current_rotation}, scene={scene_index}, split={split}")
            # Teleport to pose
            event, success = _safe_controller_step(
                controller,
                action="Teleport",
                position=current_position,
                rotation=current_rotation,
                forceAction=True,
            )
            if not success or event is None:
                print("[Fine-EQA] FPS: Teleport failed")
                return []
        except Exception:
            print("[Fine-EQA] FPS: Exception during teleport/reset")
            return []
        
        # Get depth and resolution
        event = controller.last_event
        frame = getattr(event, "frame", None)
        depth_frame = getattr(event, "depth_frame", None)
        if frame is None or depth_frame is None:
            print("[Fine-EQA] FPS: Missing frame/depth_frame")
            return []
        height, width = depth_frame.shape
        
        # Parameters
        fov_deg = 90.0
        center_row = height // 2
        num_candidates = max(num_points * 4, num_points)  # generate more candidates then FPS
        cols = np.linspace(int(width * 0.1), int(width * 0.9), num_candidates, dtype=int)
        
        # Build candidate world points from depth midline
        yaw = float(current_rotation.get("y", 0.0))
        candidates_world = []
        candidates_px = []
        for col in cols:
            d = float(depth_frame[center_row, col])
            if not np.isfinite(d) or d <= 0.01 or d > 10.0:
                continue
            # angle offset from center using horizontal FOV
            # normalized column in [-0.5, 0.5]
            norm = (col + 0.5) / width - 0.5
            angle_offset = norm * fov_deg  # degrees
            angle_rad = np.radians(yaw + angle_offset)
            # project in x-z plane (y ignored)
            world_x = current_position["x"] + d * np.sin(angle_rad)
            world_z = current_position["z"] + d * np.cos(angle_rad)
            candidates_world.append((world_x, world_z))
            candidates_px.append((col, center_row))
        
        if len(candidates_world) == 0:
            print("[Fine-EQA] FPS: No candidate world points derived from depth midline")
            return []
        
        # Greedy farthest point sampling in world space
        selected = []
        selected_idx = []
        # Start from the farthest point from the robot
        dists = [np.hypot(wx - current_position["x"], wz - current_position["z"]) for wx, wz in candidates_world]
        if len(dists) == 0:
            print("[Fine-EQA] FPS: No valid distances to candidates")
            return []
        first = int(np.argmax(dists))
        selected.append(candidates_world[first])
        selected_idx.append(first)
        
        while len(selected) < min(num_points, len(candidates_world)):
            # for each candidate, compute min distance to selected set
            min_dists = []
            for i, (wx, wz) in enumerate(candidates_world):
                if i in selected_idx:
                    min_dists.append(-1.0)
                    continue
                md = min(np.hypot(wx - sx, wz - sz) for sx, sz in selected)
                min_dists.append(md)
            nxt = int(np.argmax(min_dists))
            if min_dists[nxt] <= 0:
                break
            selected.append(candidates_world[nxt])
            selected_idx.append(nxt)
        
        sampled = []
        for idx in selected_idx:
            wx, wz = candidates_world[idx]
            px, py = candidates_px[idx]
            sampled.append((wx, wz, (int(px), int(py))))
        print(f"[Fine-EQA] FPS: Selected {len(sampled)} sampled points: {sampled}")
        return sampled
    
    def compute_local_semantic_value(self, image: Image.Image, question: str,
                                      sampled_points: List[Tuple[float, float, Tuple[int, int]]]) -> np.ndarray:
        """
        Compute Local Semantic Value (vl) for sampled points.
        VLM evaluates exploration priority based on task relevance.
        
        Returns:
            vl: Array of local semantic values for each sampled point
        """
        if len(sampled_points) < self.config.min_num_prompt_points:
            return np.ones(len(sampled_points)) / max(len(sampled_points), 1)
        
        # Draw markers on image
        img_with_markers = image.copy()
        draw = ImageDraw.Draw(img_with_markers)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        except:
            font = ImageFont.load_default()
        
        draw_letters = ["A", "B", "C", "D"][:len(sampled_points)]
        
        for i, (_, _, (px, py)) in enumerate(sampled_points):
            draw.ellipse(
                (px - self.config.circle_radius, py - self.config.circle_radius,
                 px + self.config.circle_radius, py + self.config.circle_radius),
                fill=(200, 200, 200, 255),
                outline=(0, 0, 0, 255),
                width=3
            )
            draw.text((px, py), draw_letters[i], font=font, fill=(0, 0, 0, 255), anchor="mm")
        
        # Query VLM for direction preference
        prompt = f"\nConsider the question: '{question}', and you will explore the environment for answering it.\nWhich direction (black letters on the image) would you explore then? Answer with a single letter."
        
        try:
            print(f"[Fine-EQA] LSV: sampled_points={len(sampled_points)}, letters={draw_letters}")
            # Use VLM to get probabilities for each direction
            lsv = self._get_token_probabilities(img_with_markers, prompt, draw_letters)
            print(f"[Fine-EQA] LSV raw={lsv}")
            
            # Scale by number of points (as in EXPRESS-Bench)
            lsv *= len(sampled_points) / 3.0
            print(f"[Fine-EQA] LSV scaled={lsv}")
            
            return lsv
            
        except Exception as e:
            print(f"Error computing LSV: {e}")
            return np.ones(len(sampled_points)) / len(sampled_points)
    
    def compute_global_semantic_value(self, image: Image.Image, question: str) -> float:
        """
        Compute Global Semantic Value (vg) - exploration decision confidence.
        VLM assesses whether current view is worth exploring.
        
        Returns:
            vg: Global semantic value (confidence score)
        """
        prompt = f"\nConsider the question: '{question}', and you will explore the environment for answering it. Is there any direction shown in the image worth exploring? Answer with Yes or No."
        
        try:
            # Get probability for "Yes" vs "No"
            probs = self._get_token_probabilities(image, prompt, ["Yes", "No"])
            gsv = probs[0]  # Probability of "Yes"
            print(f"[Fine-EQA] GSV probs(Yes/No)={probs}, gsv_raw_yes={gsv}")
            
            # Apply temperature scaling as in EXPRESS-Bench
            gsv = np.exp(gsv / self.config.gsv_T) / self.config.gsv_F
            print(f"[Fine-EQA] GSV scaled={gsv:.4f} (T={self.config.gsv_T}, F={self.config.gsv_F})")
            
            return float(gsv)
            
        except Exception as e:
            print(f"Error computing GSV: {e}")
            return 0.5
    
    def _get_token_probabilities(self, image: Image.Image, prompt: str, 
                                 tokens: List[str]) -> np.ndarray:
        """
        Get normalized probabilities for candidate tokens.
        Prismatic backend follows EXPRESS-Bench `vlm.get_loss()` path.
        """
        try:
            if self._use_prismatic_backend():
                prompt_builder = self.vlm_model.get_prompt_builder()
                prompt_builder.add_turn(role="human", message=prompt)
                prompt_text = prompt_builder.get_prompt()

                # EXPRESS-Bench style (older Prismatic): model.get_loss(...)
                if hasattr(self.vlm_model, "get_loss"):
                    losses = self.vlm_model.get_loss(
                        image,
                        prompt_text,
                        return_string_probabilities=tokens,
                    )[0]
                    losses = np.array(losses, dtype=np.float32)
                    if losses.ndim != 1 or len(losses) != len(tokens):
                        raise ValueError(
                            f"Unexpected loss shape for tokens: shape={losses.shape}, n_tokens={len(tokens)}"
                        )
                    probs = np.exp(-losses)
                    denom = float(np.sum(probs))
                    if not np.isfinite(denom) or denom <= 0:
                        raise ValueError(f"Invalid probability denominator from losses: {denom}")
                    return probs / denom

                # Newer Prismatic API: avoid generate/generate_batch path and use forward logits directly.
                if hasattr(self.vlm_model, "vision_backbone") and hasattr(self.vlm_model, "llm_backbone"):
                    tokenizer = self.vlm_model.llm_backbone.tokenizer
                    image_transform = self.vlm_model.vision_backbone.image_transform

                    pixel_values = image_transform(image)
                    if isinstance(pixel_values, torch.Tensor):
                        pixel_values = pixel_values[None, ...].to(self.device)
                    elif isinstance(pixel_values, dict):
                        pixel_values = {
                            k: v[None, ...].to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in pixel_values.items()
                        }
                    else:
                        raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

                    input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)

                    with torch.inference_mode():
                        autocast_dtype = getattr(self.vlm_model.llm_backbone, "half_precision_dtype", torch.bfloat16)
                        enable_amp = bool(getattr(self.vlm_model, "enable_mixed_precision_training", True))
                        use_cuda_amp = str(self.device).startswith("cuda")
                        with torch.autocast("cuda", dtype=autocast_dtype, enabled=(use_cuda_amp and enable_amp)):
                            output = self.vlm_model.forward(
                                input_ids=input_ids,
                                pixel_values=pixel_values,
                                return_dict=True,
                            )
                    if not hasattr(output, "logits"):
                        raise ValueError(f"Prismatic forward output has no logits: {type(output)}")
                    last_token_logits = output.logits[0, -1, :]
                    vocab_probs = torch.softmax(last_token_logits, dim=0)

                    token_ids = []
                    for token in tokens:
                        if hasattr(self.vlm_model, "string2idx") and token in self.vlm_model.string2idx:
                            token_ids.append(int(self.vlm_model.string2idx[token]))
                        else:
                            tid = tokenizer.encode(token, add_special_tokens=False)
                            token_ids.append(int(tid[0]) if len(tid) > 0 else None)

                    probs_list = []
                    for tid in token_ids:
                        if tid is not None and 0 <= tid < int(vocab_probs.shape[0]):
                            probs_list.append(float(vocab_probs[tid]))
                        else:
                            probs_list.append(0.0)
                    probs = np.array(probs_list, dtype=np.float32)
                    denom = float(np.sum(probs))
                    if not np.isfinite(denom) or denom <= 0:
                        raise ValueError(f"Invalid probability denominator from Prismatic forward logits: {denom}")
                    return probs / denom

                raise RuntimeError(
                    "Prismatic backend missing both get_loss and forward-logit compatible interfaces."
                )
            elif self._can_use_hf_backend():
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }]
                
                chat_text = self.vlm_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                inputs = self.vlm_processor(
                    text=[chat_text],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                
                # Get logits for each token
                token_probs = []
                with torch.no_grad():
                    for token in tokens:
                        # Generate with constrained output
                        gen_output = self.vlm_model.generate(
                            **inputs,
                            max_new_tokens=5,
                            do_sample=False,
                            output_scores=True,
                            return_dict_in_generate=True,
                            pad_token_id=self.vlm_processor.tokenizer.eos_token_id,
                        )
                        
                        # Get first token scores
                        if hasattr(gen_output, 'scores') and len(gen_output.scores) > 0:
                            first_token_scores = gen_output.scores[0][0]
                            token_ids = self.vlm_processor.tokenizer.encode(token, add_special_tokens=False)
                            if len(token_ids) > 0:
                                token_id = token_ids[0]
                                score = float(torch.softmax(first_token_scores, dim=0)[token_id])
                            else:
                                score = 1.0 / len(tokens)
                        else:
                            score = 1.0 / len(tokens)
                        
                        token_probs.append(score)
                
                # Normalize to probabilities
                token_probs = np.array(token_probs)
                if token_probs.sum() > 0:
                    token_probs = token_probs / token_probs.sum()
                else:
                    token_probs = np.ones(len(tokens)) / len(tokens)
                return token_probs
            else:
                raise RuntimeError(
                    "No valid backend for token probabilities: missing HF processor and prismatic model API."
                )
            
        except Exception as e:
            print(f"Error in _get_token_probabilities: {e}")
            return np.ones(len(tokens), dtype=np.float32) / max(len(tokens), 1)
    
    def _parse_region_key_from_text(self, text: str) -> Optional[str]:
        """Parse region key (A-T) from free-form model output."""
        if not text:
            return None
        keys = set(self.config.regions_dict.keys())
        m = re.search(r"\b([A-Z])\b", text.upper())
        if m:
            key = m.group(1)
            if key in keys:
                return key
        text_lower = text.lower()
        for k, v in self.config.regions_dict.items():
            if v.lower() in text_lower:
                return k
        return None

    def classify_current_scene_region(self, image: Image.Image) -> Tuple[Optional[str], float]:
        """
        Functional Region Semantic Mapping: classify into region key (A-T), then map to region name.
        
        Returns:
            (region_name, confidence): Identified region and confidence score
        """
        region_keys = list(self.config.regions_dict.keys())
        regions_desc = ", ".join([f"{k}:{v}" for k, v in self.config.regions_dict.items()])
        
        prompt = f"""You are analyzing an indoor scene to identify its functional region.

Available functional regions: {regions_desc}

Based on the objects, layout, and appearance in this image, classify which functional region this scene belongs to.
Output ONLY one region KEY (single capital letter A-T). If uncertain, output T.

Region Key:"""
        
        try:
            if not self._use_prismatic_backend() and not self._can_use_hf_backend():
                raise RuntimeError(
                    "No valid backend for scene region classification: missing HF processor and prismatic model API."
                )

            probs = self._get_token_probabilities(image, prompt, region_keys)
            top_idx = int(np.argmax(probs)) if len(probs) > 0 else region_keys.index("T")
            pred_key = region_keys[top_idx]
            confidence = float(probs[top_idx]) if len(probs) > 0 else 0.0
            pred_name = self.region_key_to_name.get(pred_key, "unknown")
            self.last_region_key = pred_key
            print(f"[Fine-EQA] Region classify key={pred_key}, region={pred_name}, confidence={confidence:.4f}")
            return pred_name, confidence

        except Exception as e:
            print(f"Error classifying scene region: {e}")
            self.last_region_key = "T"
            return "unknown", 0.0
    
    def identify_representative_points(
        self, 
        image: Image.Image,
        identified_region: str,
        current_position: Dict,
        sampled_points: Optional[List[Tuple[float, float, Tuple[int, int]]]] = None,
        region_key: Optional[str] = None,
    ) -> List[Tuple[float, float]]:
        """
        Identify representative points q within the identified functional region.
        These points are used to update the functional region semantic map Mreg.
        
        Returns:
            List of (x, z) world coordinates for representative points
        """
        if sampled_points is None:
            sampled_points = []
        if identified_region == "unknown":
            return []

        # If sampled points are unavailable, fallback to current position.
        if len(sampled_points) == 0:
            x = current_position.get('x', 0.0)
            z = current_position.get('z', 0.0)
            return [(x, z)]

        img_with_markers = image.copy()
        draw = ImageDraw.Draw(img_with_markers)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        except Exception:
            font = ImageFont.load_default()

        draw_letters = ["A", "B", "C", "D"][:len(sampled_points)]
        for i, (_, _, (px, py)) in enumerate(sampled_points):
            draw.ellipse(
                (px - self.config.circle_radius, py - self.config.circle_radius,
                 px + self.config.circle_radius, py + self.config.circle_radius),
                fill=(200, 200, 200, 255),
                outline=(0, 0, 0, 255),
                width=3
            )
            draw.text((px, py), draw_letters[i], font=font, fill=(0, 0, 0, 255), anchor="mm")

        region_text = identified_region
        if region_key is not None:
            region_text = f"{identified_region} ({region_key})"
        prompt = (
            f"Based on the content of the image, judge which letter indicates the location "
            f"that belongs to the {region_text} area. Answer with a single letter."
        )

        try:
            probs = self._get_token_probabilities(img_with_markers, prompt, draw_letters)
            best_idx = int(np.argmax(probs)) if len(probs) > 0 else 0
            qx, qz, _ = sampled_points[best_idx]
            print(
                f"[Fine-EQA] Representative point region={identified_region}, "
                f"letter={draw_letters[best_idx]}, probs={probs}"
            )
            return [(qx, qz)]
        except Exception as e:
            print(f"[Fine-EQA] Representative point selection failed: {e}")

        x = current_position.get('x', 0.0)
        z = current_position.get('z', 0.0)
        return [(x, z)]
    
    def check_if_can_answer(self, image: Image.Image, question: str) -> bool:
        """
        Check if current view contains sufficient information to answer the question.
        This determines if exploration should terminate.
        """
        prompt = f"""You are an intelligent assistant tasked with determining whether the given image contains sufficient information to

answer the provided question.

The input consists of QUESTION and IMAGE. The QUESTION is what you need to evaluate, while the IMAGE
represents the currently observed environment.

Question: {question}

Respond only with "yes" or "no" without attempting to answer the question itself."""
        
        # Primary: Use GPT-5-mini via OpenAI API
        try:
            model_id = _resolve_gpt_model_id(getattr(self.config, "gpt_model", "gpt-5-mini"))
            print(f"[Fine-EQA] Termination check model_id={model_id}")
            full_output = run_gpt_action_prediction(image, prompt, model_id)
            print(f"[Fine-EQA] Termination check GPT output={full_output[:200].strip()}...")
            yn = parse_yes_no(full_output)
            if yn is not None:
                result = yn == "yes"
                print(f"[Fine-EQA] Termination check parsed={yn}, decision={result}")
                return result
        except Exception as e:
            print(f"Error checking via GPT: {e}")
        
        # Fallback: Use local VLM inference if GPT not available or unparsable
        try:
            if self._use_prismatic_backend():
                prompt_builder = self.vlm_model.get_prompt_builder()
                prompt_builder.add_turn(role="human", message=prompt)
                prompt_text = prompt_builder.get_prompt()
                
                output = self.vlm_model.generate(
                    image,
                    prompt_text,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=10,
                    min_length=1,
                )
                yn_fb = parse_yes_no(output) or ("yes" if "yes" in output.lower() else None)
                result = yn_fb == "yes"
                print(f"[Fine-EQA] Termination fallback output={output[:200].strip()}..., parsed={yn_fb}, decision={result}")
                return result
            elif self._can_use_hf_backend():
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }]
                
                chat_text = self.vlm_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                inputs = self.vlm_processor(
                    text=[chat_text],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                
                with torch.no_grad():
                    gen = self.vlm_model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                    )
                    
                    output = self.vlm_processor.batch_decode(gen, skip_special_tokens=True)[0]
                    yn_fb = parse_yes_no(output) or ("yes" if "yes" in output.lower() else None)
                    result = yn_fb == "yes"
                    print(f"[Fine-EQA] Termination fallback output={output[:200].strip()}..., parsed={yn_fb}, decision={result}")
                    return result
            else:
                raise RuntimeError(
                    "No valid backend for termination fallback: missing HF processor and prismatic model API."
                )
        except Exception as e:
            print(f"Error checking if can answer (fallback): {e}")
            return False
    
    def predict_action(
        self,
        image: Image.Image,
        question: str,
        current_position: Optional[Dict] = None,
        exploration_step: int = 0,
        render_metadata: Optional[Dict] = None,
    ) -> Tuple[List[float], str]:
        """
        Predict next action using Goal-Oriented Exploration (GOE) only.
        
        Implements the paper's GOE strategy:
        1. Task-relevant region prioritization
        2. Functional region semantic mapping (Mreg)
        3. Masked semantic mapping (Mmasked)
        4. Position selection: χ = argmax(Mmasked)
        
        Returns:
            actions: [rotation, forward, look] in degrees/cm
            thinking: Explanation of the action choice
        """
        thinking_parts = []
        thinking_parts.append("[GOE Strategy]")
        
        # Step 1: Check if we can answer from current view (termination condition)
        if self.check_if_can_answer(image, question):
            thinking_parts.append("✓ Termination: Current view contains sufficient information to answer.")
            return [0.0, 0.0, 0.0], "\n".join(thinking_parts)
        
        # Step 2: Task-Relevant Region Prioritization (done once at start)
        if exploration_step == 0 or not self.prioritized_regions:
            self.prioritized_regions = self.prioritize_task_relevant_regions(question)
            self._init_goe_state()
            region_names = [r[0] for r in self.prioritized_regions]
            thinking_parts.append(f"1. Task-Relevant Region Prioritization: {', '.join(region_names)}")
        
        # Step 3: Semantic Map Construction (FBE-style: Msem ← (psample, vl, vg))
        if current_position and render_metadata:
            # 3a. Farthest Point Sampling
            sampled_points = self.farthest_point_sampling(
                image, render_metadata, num_points=self.config.num_prompt_points
            )
            thinking_parts.append(f"2. Sampled {len(sampled_points)} points for semantic evaluation")
            
            if len(sampled_points) >= self.config.min_num_prompt_points:
                # 3b. Compute Local Semantic Value (vl)
                vl = self.compute_local_semantic_value(image, question, sampled_points)
                
                # 3c. Compute Global Semantic Value (vg)
                vg = self.compute_global_semantic_value(image, question)
                
                # 3d. Weighted Fusion: SV = vl * vg
                sv = vl * vg
                
                thinking_parts.append(f"   LSV: {vl}, GSV: {vg:.3f}")
                thinking_parts.append(f"   SV: {sv}")
                
                # 3e. Integrate into Msem (Equation: Msem ← (psample, vl, vg))
                world_points = [(x, z) for x, z, _ in sampled_points]
                self.semantic_map.integrate_semantic_values(
                    world_points, sv,
                    radius=self.config.semantic_radius,
                    obs_weight=self.config.semantic_integration_weight
                )
                # 3f. Smooth Msem dynamically after integration
                self.semantic_map.smooth_semantic_map(sigma=self.config.smooth_sigma)
                thinking_parts.append(f"   Integrated semantic values into Msem")
        
        # Step 4: Functional Region Semantic Mapping
        # Classify current scene into functional region
        identified_region, confidence = self.classify_current_scene_region(image)
        thinking_parts.append(f"3. Functional Region Classification: {identified_region} (confidence: {confidence:.2f})")
        
        # Update Mreg if confidence exceeds threshold
        if (
            current_position
            and identified_region != "unknown"
            and confidence >= self.config.region_confidence_threshold
        ):
            region_id = self.region_name_to_id.get(identified_region, -1)
            if region_id >= 0:
                repr_points = self.identify_representative_points(
                    image=image,
                    identified_region=identified_region,
                    current_position=current_position,
                    sampled_points=sampled_points if "sampled_points" in locals() else [],
                    region_key=self.last_region_key,
                )
                if len(repr_points) == 0:
                    qx = current_position.get('x', 0.0)
                    qz = current_position.get('z', 0.0)
                    repr_points = [(qx, qz)]

                for x, z in repr_points:
                    self.semantic_map.update_region_map(
                        x, z, region_id,
                        radius=self.config.region_neighborhood_radius
                    )
                    self._enqueue_region_point(
                        identified_region,
                        (x, z),
                        radius=self.config.region_neighborhood_radius,
                    )

                self.semantic_map.merge_adjacent_regions(
                    region_id,
                    merge_para=self.config.region_merge_para,
                    radius=self.config.region_neighborhood_radius,
                )
                thinking_parts.append(
                    f"   Updated Mreg for region '{identified_region}' at "
                    f"({repr_points[0][0]:.2f}, {repr_points[0][1]:.2f})"
                )

        # Step 5: Determine current priority region (with GOE state machine)
        state_is_fbe, exp_list = self._update_goe_state()
        active_idx = self._active_goe_index() if not state_is_fbe else None
        if active_idx is not None:
            self.current_priority_region = exp_list[active_idx]["region"]
            thinking_parts.append(
                f"4. GOE Active Region: {self.current_priority_region} "
                f"(count={exp_list[active_idx]['count']}, dir={exp_list[active_idx]['dir']}, "
                f"queued={len(exp_list[active_idx]['point'])})"
            )
        else:
            current_best_priority = float('inf')
            for region_name, priority in self.prioritized_regions:
                region_id = self.region_name_to_id.get(region_name, -1)
                if region_id >= 0 and self.semantic_map.region_map[self.semantic_map.region_map == region_id].size > 0:
                    if priority < current_best_priority:
                        current_best_priority = priority
                        self.current_priority_region = region_name
                        break
            if self.current_priority_region is None and self.prioritized_regions:
                self.current_priority_region = self.prioritized_regions[0][0]
            thinking_parts.append(f"4. Current Priority Region: {self.current_priority_region}")

        # Step 6: Masked Semantic Mapping (Equation 10)
        priority_region_id = self.region_name_to_id.get(self.current_priority_region, -1)
        rotation = 0.0
        forward = 0.0
        look = 0.0
        target_pos = None
        chosen_grid = None
        used_region_unresolved_rotate_fallback = False

        if priority_region_id >= 0:
            masked_map = self.semantic_map.get_masked_semantic_map(
                priority_region_id,
                visited_decay=self.config.visited_point_decay,
                smooth_sigma=self.config.smooth_sigma,
                explored_zero_radius=self.config.exploration_radius,
            )

            # GOE queued points first (similar to EXPRESS-Bench find_next_point_region)
            if active_idx is not None and active_idx < len(self.exp_list):
                queued_points = self.exp_list[active_idx]["point"]
                if len(queued_points) > 0:
                    vals = []
                    for gx, gz in queued_points:
                        if 0 <= gx < masked_map.shape[0] and 0 <= gz < masked_map.shape[1]:
                            vals.append(masked_map[gx, gz])
                        else:
                            vals.append(-1e9)
                    best_local_idx = int(np.argmax(vals))
                    chosen_grid = queued_points.pop(best_local_idx)

            # Otherwise pick from global maxima with random tie-break (closer to EXPRESS-Bench)
            if chosen_grid is None:
                max_value = float(np.max(masked_map))
                max_coords = np.argwhere(masked_map == max_value)
                if len(max_coords) > 0:
                    rand_idx = int(np.random.randint(len(max_coords)))
                    chosen_grid = tuple(max_coords[rand_idx].tolist())

            if chosen_grid is not None:
                gx, gz = int(chosen_grid[0]), int(chosen_grid[1])
                self.semantic_map.explored_points.append((gx, gz))
                target_x, target_z = self.semantic_map.grid_to_world(gx, gz)
                target_pos = (target_x, target_z)
                thinking_parts.append(f"5. Masked Semantic Map: Target position at ({target_x:.2f}, {target_z:.2f})")
            else:
                thinking_parts.append(f"5. No target candidates found for region '{self.current_priority_region}'")
        else:
            rotation = 0.0
            forward = 0.0
            look = 90.0
            used_region_unresolved_rotate_fallback = True
            thinking_parts.append(
                f"5. Priority region '{self.current_priority_region}' unresolved; use fixed look-right fallback"
            )

        if target_pos is not None and current_position:
            target_x, target_z = target_pos
            curr_x = current_position.get('x', 0.0)
            curr_z = current_position.get('z', 0.0)
            curr_rot = 0.0
            if isinstance(current_position.get("rotation"), dict):
                curr_rot = float(current_position["rotation"].get("y", 0.0))
            elif render_metadata and isinstance(render_metadata.get("rotation"), dict):
                curr_rot = float(render_metadata["rotation"].get("y", 0.0))

            dx = target_x - curr_x
            dz = target_z - curr_z
            target_angle = np.arctan2(dx, dz) * 180 / np.pi
            rotation = target_angle - curr_rot
            while rotation > 180:
                rotation -= 360
            while rotation < -180:
                rotation += 360

            distance = np.sqrt(dx**2 + dz**2)
            forward = min(distance * 100, self.config.max_forward_distance)
            look = 0.0
            thinking_parts.append(
                f"6. Action: Rotate {rotation:.1f}°, Forward {forward:.1f}cm → "
                f"Target region '{self.current_priority_region}'"
            )
        elif target_pos is not None:
            thinking_parts.append("6. Action: Hold (missing position metadata for navigation)")
        elif used_region_unresolved_rotate_fallback:
            thinking_parts.append("6. Action: Rotate 0.0°, Forward 0.0cm, Look 90.0° right")
        else:
            thinking_parts.append("6. Action: Hold (no valid navigation target)")
        
        # Update tracking
        if current_position:
            x, z = current_position.get('x', 0), current_position.get('z', 0)
            self.semantic_map.mark_visited(x, z, radius=self.config.exploration_radius)
        
        # Store observation
        self.observation_history.append(image)
        if current_position:
            self.position_history.append(current_position)
        
        actions = [float(rotation), float(forward), float(look)]
        thinking_text = "\n".join(thinking_parts)
        
        return actions, thinking_text


def main():
    parser = argparse.ArgumentParser(
        description="Test ProcTHOR action prediction with visualization"
    )
    parser.add_argument(
        "--model_path", default=None, help="Path to trained model checkpoint or Gemini model alias. Required unless using Prismatic VLM"
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
    parser.add_argument(
        "--fine_eqa_max_forward_cm",
        type=float,
        default=None,
        help="Override Fine-EQA max forward movement (cm). Default from config is 300.0cm",
    )
    parser.add_argument(
        "--use_prismatic",
        action="store_true",
        help="Use Prismatic VLM backend instead of HF AutoModel",
    )
    parser.add_argument(
        "--prismatic_model_id",
        type=str,
        default="prism-dinosiglip+7b",
        help="Prismatic model id (e.g., prism-dinosiglip+7b)",
    )
    parser.add_argument(
        "--hf_token_file",
        type=str,
        default=".hf_token",
        help="Path to a file containing the Hugging Face token for gated models",
    )
    # GEMINI
    parser.add_argument(
        "--use_gemini_verifier",
        action="store_true",
        help="Use Gemini API for verifier (instead of local Qwen model)",
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

    if not args.input_view_for_verifier and not args.gt_view_for_verifier:
        print(f"Loading model from {args.model_path}...")
        if args.use_prismatic:
            print(f"Loading Prismatic VLM model: {args.prismatic_model_id} ...")
            try:
                from pathlib import Path
                from prismatic import load as prism_load
            except Exception as e:
                raise RuntimeError(f"Failed to import prismatic. Please install it first. Error: {e}")
            
            # Resolve Hugging Face token for gated base LMs (e.g., Llama-2)
            hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
            if not hf_token:
                try:
                    if args.hf_token_file and os.path.exists(args.hf_token_file):
                        hf_token = Path(args.hf_token_file).read_text().strip()
                except Exception:
                    hf_token = None

            student_model = prism_load(args.prismatic_model_id, hf_token=hf_token)
            student_model.to(args.device, dtype=torch.bfloat16)
            student_processor = None
            print("✓ Prismatic VLM model loaded")
        else:
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

    # Initialize AI2-THOR controller.
    # We also use it for pixel-based visibility even in verifier-only modes.
    controller = None
    print("Initializing AI2-THOR controller...")
    controller = get_procthor_controller(headless=True)

    # Initialize Fine-EQA Action Predictor
    fine_eqa_predictor = None
    if not args.input_view_for_verifier and not args.gt_view_for_verifier:
        print("Initializing Fine-EQA Action Predictor...")
        config = FineEQAConfig()
        if args.fine_eqa_max_forward_cm is not None:
            config.max_forward_distance = float(args.fine_eqa_max_forward_cm)
        print(f"[Fine-EQA] max_forward_distance={config.max_forward_distance}cm")
        fine_eqa_predictor = FineEQAActionPredictor(
            vlm_model=student_model,
            vlm_processor=student_processor,
            config=config,
            device=args.device,
            controller=controller,
            use_prismatic=args.use_prismatic
        )

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
            "data_type": item.get("data_type", "single"),
            "house_id": item.get("scene_id", f"house_{scene_index}"),
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
            # Always initialize these so downstream saving/visualization works in all modes.
            # In verifier-only modes we won't append additional views/steps.
            current_images = [input_img]
            rollout_steps = []

            # Visibility via pixel count (gt pose vs generated final view)
            target_object_id = item.get("target_object_id")
            counting_object_type = item.get("counting_object") if item.get("question_type") == "counting" else None
            gt_object_pixels = None
            gen_object_pixels = None
            computed_visibility = None
            do_rollout = (not args.input_view_for_verifier) and (not args.gt_view_for_verifier)

            # Compute GT pixel count at the last-step pose (steps[-1] position/rotation)
            if controller is not None and (target_object_id or counting_object_type):
                try:
                    gt_render_metadata = dict(render_metadata)
                    gt_render_metadata["position"] = gt_position
                    gt_render_metadata["rotation"] = gt_rotation
                    if counting_object_type:
                        gt_render_metadata["pixel_count_object_type"] = counting_object_type
                        gt_render_metadata.pop("pixel_count_object_id", None)
                    else:
                        gt_render_metadata["pixel_count_object_id"] = target_object_id
                        gt_render_metadata.pop("pixel_count_object_type", None)
                    _, gt_meta = build_additional_view(controller, [0, 0, 0], gt_render_metadata)
                    if isinstance(gt_meta, dict):
                        gt_object_pixels = gt_meta.get("pixel_count")

                    # In input_view_for_verifier mode, the "final view" is the input pose/view.
                    # Measure pixels at the input pose so we can compute visibility = input/gt.
                    if args.input_view_for_verifier:
                        input_render_metadata = dict(render_metadata)
                        input_render_metadata["position"] = render_position
                        input_render_metadata["rotation"] = render_rotation
                        if counting_object_type:
                            input_render_metadata["pixel_count_object_type"] = counting_object_type
                            input_render_metadata.pop("pixel_count_object_id", None)
                        else:
                            input_render_metadata["pixel_count_object_id"] = target_object_id
                            input_render_metadata.pop("pixel_count_object_type", None)
                        _, input_meta = build_additional_view(
                            controller, [0, 0, 0], input_render_metadata
                        )
                        if isinstance(input_meta, dict):
                            gen_object_pixels = input_meta.get("pixel_count")
                except Exception:
                    gt_object_pixels = None

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
                # The verifier observes the GT view in this mode.
                current_images = [gt_img]

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
                
                for step in range(args.max_rollout_steps):
                    print(f"--- Rollout Step {step+1}/{args.max_rollout_steps} ---")
                    
                    
                    print(f"\n[Fine-EQA] Predicting action for step {step+1}...")
                    try:
                        pose_for_predictor = dict(render_metadata["position"])
                        pose_for_predictor["rotation"] = dict(render_metadata.get("rotation", {}))
                        predicted_actions, thinking_process = fine_eqa_predictor.predict_action(
                            image=current_images[-1],
                            question=vqa_question,
                            current_position=pose_for_predictor,
                            exploration_step=step,
                            render_metadata=render_metadata,
                        )
                        actions_text = f"rotation={predicted_actions[0]}, forward={predicted_actions[1]}, look={predicted_actions[2]}"
                        step_output = actions_text
                        
                        if sum(abs(x) for x in predicted_actions) < 0.1:
                            parsed = {"is_stop": True, "action": predicted_actions, "thinking": thinking_process, "action_text": "Terminate", "is_unknown": False}
                        else:
                            parsed = {"is_stop": False, "action": predicted_actions, "thinking": thinking_process, "action_text": actions_text, "is_unknown": False}
                    except Exception as e:
                        print(f"Error in Fine-EQA prediction: {e}")
                        traceback.print_exc()
                        parsed = {"is_stop": True, "action": [0.0, 0.0, 0.0], "thinking": "Error", "action_text": "Error", "is_unknown": True}
                        step_output = "Error"
                        predicted_actions = [0.0, 0.0, 0.0]
                        actions_text = "Error"
                        thinking_process = "Error"

                    print(f"Step {step+1} Output:\\n{step_output}\\n")
                    student_output = step_output if step_output else student_output
                    
                    chat.append({"role": "assistant", "content": [{"type": "text", "text": step_output}]})
                    
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
                    # Ask renderer to count pixels (id or type) if available
                    if counting_object_type:
                        render_metadata = dict(render_metadata)
                        render_metadata["pixel_count_object_type"] = counting_object_type
                        render_metadata.pop("pixel_count_object_id", None)
                    elif target_object_id:
                        render_metadata = dict(render_metadata)
                        render_metadata["pixel_count_object_id"] = target_object_id
                        render_metadata.pop("pixel_count_object_type", None)
                    gen_img, result_metadata = build_additional_view(
                        controller, effective_action, render_metadata
                    )
                    
                    if gen_img:
                        print("✓ View generated")
                        step_record["view_image"] = gen_img
                        if isinstance(result_metadata, dict) and "pixel_count" in result_metadata:
                            # Keep legacy key for visualization while also storing counting-specific key.
                            step_record["target_object_pixel_count"] = result_metadata.get("pixel_count")
                            if counting_object_type:
                                step_record["counting_object_type"] = counting_object_type
                                step_record["counting_object_pixel_count"] = result_metadata.get("pixel_count")
                            gen_object_pixels = result_metadata.get("pixel_count")
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

            # If the model stops before any view generation, we still want a "generated" pixel
            # count at the current pose (typically the initial pose). This enables visibility
            # computation even when the first step is <stop>.
            if (
                do_rollout
                and controller is not None
                and (target_object_id or counting_object_type)
                and gen_object_pixels is None
            ):
                try:
                    gen_render_metadata = dict(render_metadata)
                    if counting_object_type:
                        gen_render_metadata["pixel_count_object_type"] = counting_object_type
                        gen_render_metadata.pop("pixel_count_object_id", None)
                    else:
                        gen_render_metadata["pixel_count_object_id"] = target_object_id
                        gen_render_metadata.pop("pixel_count_object_type", None)

                    _, gen_meta = build_additional_view(
                        controller, [0, 0, 0], gen_render_metadata
                    )
                    if isinstance(gen_meta, dict):
                        gen_object_pixels = gen_meta.get("pixel_count", gen_object_pixels)
                except Exception:
                    gen_object_pixels = gen_object_pixels

            # In gt_view_for_verifier mode, the "final view" is the GT view/pose.
            # So gen pixels should match GT pixels if available.
            if args.gt_view_for_verifier and gen_object_pixels is None and isinstance(gt_object_pixels, int):
                gen_object_pixels = gt_object_pixels

            # Compute visibility ratio (min(1.0, gen/gt)) if pixel counts are available
            if (
                isinstance(gt_object_pixels, int)
                and gt_object_pixels > 0
                and isinstance(gen_object_pixels, int)
            ):
                computed_visibility = min(
                    1.0, float(gen_object_pixels) / float(gt_object_pixels)
                )

            # Check accuracy
            gt_answer = item.get("mcq_answer")
            if gt_answer is None:
                gt_answer = item.get("answer")
                
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
            metadata_dir = os.path.join(args.output_dir, "metadata")
            os.makedirs(thinking_dir, exist_ok=True)
            os.makedirs(raw_output_dir, exist_ok=True)
            os.makedirs(metadata_dir, exist_ok=True)
            
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
                target_object_id=target_object_id,
                gt_object_pixels=gt_object_pixels,
                gen_object_pixels=gen_object_pixels,
                computed_visibility=computed_visibility,
            )

            result = {
                "sample_id": idx,
                "input_image": input_img_path,
                "num_steps": len(rollout_steps),
                "final_predicted_action": predicted_actions,
                "verifier_answer": verifier_answer,
                "gt_answer": gt_answer,
                "correct": is_correct,
                "has_generated_view": len(current_images) > 1,
                "target_object_id": target_object_id,
                "target_object_pixels_gt": gt_object_pixels,
                "target_object_pixels_gen": gen_object_pixels,
                "target_object_visibility_gen_over_gt": computed_visibility,
            }

            result_json_path = os.path.join(metadata_dir, f"sample_{idx:04d}.json")
            with open(result_json_path, "w") as f:
                json.dump(result, f, indent=4)

            # Record result
            results.append(result)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            traceback.print_exc()
            print(f"Error processing sample {idx}: {e}")
            traceback.print_exc()

    # Save results JSON
    results_json_path = os.path.join(args.output_dir, "results.json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=4)

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
        "action_model_type": "local_model" if not (args.input_view_for_verifier or args.gt_view_for_verifier) else "N/A",
        "verifier_model": args.verifier_model,
        "verifier_model_type": "gemini_api" if use_gemini_verifier else "local_model",
        "test_jsonl": args.test_jsonl,
    }
    summary_json_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=4)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Mode: {summary['mode']}")
    if not (args.input_view_for_verifier or args.gt_view_for_verifier):
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
    if use_gemini_verifier:
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
