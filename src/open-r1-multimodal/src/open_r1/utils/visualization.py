from PIL import Image, ImageDraw, ImageFont, ImageOps
from typing import Any, List, Optional
import os
import textwrap


def visualize(primary_img, add_view, actions, question, verifier_ans, sol):
    pad = 24
    gap = 16
    text_gap = 16
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()

    header = f"Q: {question}\nActions: {actions}\nVerifier: {verifier_ans}\nSol: {sol}"

    # Calculate text size
    tmp = Image.new("RGB", (10, 10), "white")
    dtmp = ImageDraw.Draw(tmp)
    text_bbox = dtmp.multiline_textbbox((0, 0), header, font=font, spacing=4)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    # Canvas size
    combo_w = primary_img.width + gap + add_view.width
    canvas_w = max(combo_w, text_w) + pad * 2
    canvas_h = pad + text_h + text_gap + primary_img.height + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    # Text
    draw.multiline_text((pad, pad), header, fill=(0, 0, 0), font=font, spacing=4)

    # Image placement (left: p, right: a)
    y0 = pad + text_h + text_gap
    x_left = (canvas_w - combo_w) // 2
    canvas.paste(primary_img, (x_left, y0))
    canvas.paste(add_view, (x_left + primary_img.width + gap, y0))

    return canvas


def wrap_text(text: str, width: int = 60) -> str:
    """Wrap text to specified width."""
    return "\n".join(textwrap.wrap(text, width=width))


def create_visualization(
    input_img_or_path: Image.Image | str,
    generated_img_or_path: Image.Image | str,
    predicted_actions: List[int],
    gt_actions: List[int],
    action_question: str,
    vqa_question: str,
    thinking_process: str,
    verifier_answer: str,
    gt_answer: str,
    is_correct: bool,
    output_path: str,
    llm_score: Optional[float] = None,
    gt_img_or_path: Image.Image | str = None,
    actions_text: Optional[str] = None,
):
    """Create visualization combining all information."""
    if isinstance(input_img_or_path, str):
        img_input = Image.open(input_img_or_path).convert("RGB")
    else:
        img_input = input_img_or_path.convert("RGB")

    # Generated image may be None (e.g., generation failed) – fallback to input image
    if generated_img_or_path is None:
        img_generated = img_input.copy()
    elif isinstance(generated_img_or_path, str):
        img_generated = Image.open(generated_img_or_path).convert("RGB")
    elif isinstance(generated_img_or_path, Image.Image):
        img_generated = generated_img_or_path.convert("RGB")
    else:
        img_generated = Image.new("RGB", img_input.size, color=(128, 128, 128))

    if gt_img_or_path is None:
        img_gt = None
    elif isinstance(gt_img_or_path, str):
        img_gt = Image.open(gt_img_or_path).convert("RGB")
    else:
        img_gt = gt_img_or_path.convert("RGB")

    # Create canvas
    img_w, img_h = img_input.size
    
    text_panel_width = 600
    total_width = img_w * 2 + text_panel_width
    if img_gt is not None: total_width += img_w
    total_height = max(img_h, 1200)  # Increased height to prevent text truncation

    canvas = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    
    # paste images
    canvas.paste(img_input, (0, 0))
    canvas.paste(img_generated, (img_w, 0))
    if img_gt is not None:
        canvas.paste(img_gt, (img_w * 2, 0))

    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_normal = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font_title = ImageFont.load_default()
        font_normal = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    text_x = img_w * 2 + 20
    if img_gt is not None: text_x += img_w
    text_y = 20
    line_spacing = 22

    draw.text((10, img_h + 10), "Input View", fill=(0, 0, 0), font=font_normal)
    draw.text((img_w + 10, img_h + 10), "Generated View (predicted)", fill=(0, 0, 0), font=font_normal)
    if img_gt is not None:
        draw.text((img_w * 2 + 10, img_h + 10), "Ground Truth View", fill=(0, 0, 0), font=font_normal)


    # Write text info
    def write_section(title, content, color=(0, 0, 0), max_lines=None, title_font=None, content_font=None):
        nonlocal text_y
        title_font = font_title if title_font is None else title_font
        content_font = font_small if content_font is None else content_font
        draw.text((text_x, text_y), title, fill=color, font=title_font)
        text_y += line_spacing + 3
        
        # Skip content rendering if empty
        if content and content.strip():
            wrapped = wrap_text(str(content), width=80)
            line_count = 0
            for line in wrapped.split('\n'):
                if text_y > total_height - 50:
                    draw.text((text_x, text_y), "... (truncated)", fill=(100, 100, 100), font=content_font)
                    text_y += 15
                    break
                if max_lines and line_count >= max_lines:
                    draw.text((text_x, text_y), "... (truncated)", fill=(100, 100, 100), font=content_font)
                    text_y += 15
                    break
                draw.text((text_x, text_y), line, fill=(50, 50, 50), font=content_font)
                text_y += 16
                line_count += 1
        text_y += 12
    
    def write_single_line(text, color=(0, 0, 0)):
        """Write a single line without title."""
        nonlocal text_y
        draw.text((text_x, text_y), text, fill=color, font=font_small)
        text_y += 18
    
    write_section("Action Question:", action_question, color=(150, 150, 0), content_font=font_small)
    write_section("VQA Question:", vqa_question, color=(0, 0, 150), content_font=font_small)
    write_section("Thinking Process:", thinking_process, color=(100, 0, 100), content_font=font_normal)
    
    # Use actions_text if provided, otherwise format predicted_actions
    if actions_text is not None:
        predicted_actions_str = actions_text
    else:
        predicted_actions_str = f"head={predicted_actions[0]:.2f}°, fwd={predicted_actions[1]:.2f}cm, view={predicted_actions[2]:.2f}°"
    
    write_section("Predicted Actions:", predicted_actions_str, color=(150, 0, 0), content_font=font_normal)
    # GT actions are not displayed as requested
    
    write_section("Verifier Answer:", verifier_answer, color=(0, 150, 150), content_font=font_normal)
    write_section("GT Answer:", gt_answer, color=(0, 0, 0), content_font=font_normal)

    if llm_score is not None:
        write_single_line(f"LLM Score: {llm_score}/5", color=(100, 100, 100))
    else:
        accuracy_text = "✓ CORRECT" if is_correct else "✗ WRONG"
        accuracy_color = (0, 200, 0) if is_correct else (200, 0, 0)
        draw.text((text_x, text_y), accuracy_text, fill=accuracy_color, font=font_title)


    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    canvas.save(output_path)
    print(f"Saved visualization: {output_path}")


def _to_pil_image(img_or_path: Image.Image | str | None) -> Optional[Image.Image]:
    """Convert a path or PIL image to RGB PIL image."""
    if img_or_path is None:
        return None
    if isinstance(img_or_path, Image.Image):
        return img_or_path.convert("RGB")
    if isinstance(img_or_path, str):
        try:
            return Image.open(img_or_path).convert("RGB")
        except Exception:
            return None
    return None


def _safe_wrap(text: str, width: int) -> str:
    """Wrap multiline text while preserving explicit new lines."""
    if text is None:
        return ""
    raw = str(text).replace("\r\n", "\n").replace("\r", "\n")
    wrapped: List[str] = []
    for part in raw.split("\n"):
        part = part.strip()
        if not part:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(part, width=max(8, width)))
    return "\n".join(wrapped).strip()


def create_multistep_rollout_visualization(
    input_images: Image.Image | str | List[Image.Image | str] | None,
    trajectory_steps: List[dict[str, Any]],
    vqa_question: str,
    verifier_answer: Optional[str],
    gt_answer: Optional[str],
    verifier_reward: Optional[float],
    output_path: str,
    target_visibility: Optional[str] = None,
    turn_type: Optional[str] = None,
    gt_image: Image.Image | str | None = None,
    target_object_id: Optional[str] = None,
    gt_object_pixels: Optional[int] = None,
    gen_object_pixels: Optional[int] = None,
    computed_visibility: Optional[float] = None,
):
    """Create a single-image summary of a multistep rollout trajectory.
    
    Shows observation images the model sees at each step, matching SFT training flow.
    - Step 1: input image (anchor) — what the model sees initially
    - Step 2: view generated after step 1's action — what the model sees at step 2
    - Step N: view generated after step N-1's action
    - GT View: ground truth image (rightmost)
    """
    # Resolve input image
    input_pil = None
    if isinstance(input_images, (Image.Image, str)):
        input_pil = _to_pil_image(input_images)
    elif isinstance(input_images, list) and input_images:
        input_pil = _to_pil_image(input_images[0])

    # Determine thumbnail size from first available image
    first_img = input_pil
    if first_img is None and trajectory_steps:
        first_img = _to_pil_image(trajectory_steps[0].get("view_image"))
    if first_img is None:
        first_img = Image.new("RGB", (320, 240), (230, 230, 230))

    base_w, base_h = first_img.size
    thumb_w = max(220, min(360, base_w))
    thumb_h = max(160, min(280, base_h))

    # Build columns: shift images so each step shows the observation the model sees
    # Step 1 image = input (anchor), Step 2 image = step 1's generated view, etc.
    columns: List[dict[str, Any]] = []
    steps = trajectory_steps or []

    for idx, step in enumerate(steps, start=1):
        status = "move"
        if step.get("is_stop"):
            status = "stop"
        elif step.get("is_unknown"):
            status = "unknown"

        action_text = str(step.get("action_text") or "N/A")
        thinking_text = str(step.get("thinking") or "N/A")
        termination_reason = step.get("termination_reason")

        summary = (
            f"status: {status}\n"
            f"action: {action_text}\n"
            f"thinking: {thinking_text}"
        )
        if termination_reason:
            summary += f"\ntermination: {termination_reason}"

        # Shifted image assignment:
        #   Step 1 → input image (anchor)
        #   Step N → step N-1's generated view_image
        if idx == 1:
            step_image = input_pil
        else:
            prev_step = steps[idx - 2]  # 0-indexed: step N maps to steps[N-2]
            step_image = prev_step.get("view_image")

        columns.append(
            {
                "title": f"Step {idx}",
                "image": step_image,
                "text": summary,
            }
        )

    # If the last step generated a view, add it as the final observation column
    if steps and steps[-1].get("view_image") and not steps[-1].get("is_stop"):
        extra = "Last generated observation."
        if gen_object_pixels is not None:
            extra += f"\nTarget pixels (gen): {gen_object_pixels}"
        columns.append(
            {
                "title": f"Final View",
                "image": steps[-1].get("view_image"),
                "text": extra,
            }
        )

    # Add GT image as rightmost column
    gt_pil = _to_pil_image(gt_image)
    if gt_pil is not None:
        gt_text = f"Ground truth answer: {gt_answer or 'N/A'}"
        if gt_object_pixels is not None:
            gt_text += f"\nTarget pixels (gt): {gt_object_pixels}"
        columns.append(
            {
                "title": "GT View",
                "image": gt_pil,
                "text": gt_text,
            }
        )

    pad = 20
    gap = 12
    title_h = 24
    canvas_w = pad * 2 + len(columns) * thumb_w + max(0, len(columns) - 1) * gap

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()

    reward_text = (
        f"{float(verifier_reward):.3f}" if isinstance(verifier_reward, (int, float)) else "N/A"
    )
    header_parts = [
        f"Question: {vqa_question}",
        f"GT answer: {gt_answer or 'N/A'}",
        f"Verifier answer: {verifier_answer or 'N/A'}",
        f"Verifier reward: {reward_text}"
    ]
    if turn_type or target_visibility:
        meta = []
        if turn_type: meta.append(f"Turn: {turn_type}")
        if target_visibility: meta.append(f"Visibility: {target_visibility}")
        header_parts.insert(1, " | ".join(meta))

    if target_object_id:
        header_parts.append(f"Target object: {target_object_id}")
    if gt_object_pixels is not None or gen_object_pixels is not None:
        gt_txt = str(gt_object_pixels) if gt_object_pixels is not None else "N/A"
        gen_txt = str(gen_object_pixels) if gen_object_pixels is not None else "N/A"
        header_parts.append(f"Target pixels (gen/gt): {gen_txt}/{gt_txt}")
    if computed_visibility is not None:
        header_parts.append(f"Computed visibility min(1, gen/gt): {computed_visibility:.4f}")
    
    header = "\n".join(header_parts)
    header_wrap = _safe_wrap(header, width=max(60, int((canvas_w - 2 * pad) / 9)))

    tmp = Image.new("RGB", (10, 10), "white")
    tmp_draw = ImageDraw.Draw(tmp)
    header_bbox = tmp_draw.multiline_textbbox((0, 0), header_wrap, font=font_title, spacing=4)
    header_h = (header_bbox[3] - header_bbox[1]) + 10

    line_h = max(14, font_body.getbbox("Ag")[3] - font_body.getbbox("Ag")[1] + 2)
    max_lines = 1
    for col in columns:
        wrapped = _safe_wrap(col["text"], width=42)
        col["wrapped_text"] = wrapped
        line_count = len(wrapped.splitlines()) if wrapped else 1
        max_lines = max(max_lines, line_count)
    text_h = max(90, max_lines * line_h + 16)

    canvas_h = pad + header_h + 10 + title_h + thumb_h + 8 + text_h + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    draw.multiline_text((pad, pad), header_wrap, fill=(0, 0, 0), font=font_title, spacing=4)
    draw.line([(pad, pad + header_h), (canvas_w - pad, pad + header_h)], fill=(220, 220, 220), width=2)

    y_top = pad + header_h + 10
    for idx, col in enumerate(columns):
        x = pad + idx * (thumb_w + gap)

        draw.text((x, y_top), col["title"], fill=(0, 0, 0), font=font_title)

        panel = Image.new("RGB", (thumb_w, thumb_h), (245, 245, 245))
        source_img = _to_pil_image(col.get("image"))
        if source_img is not None:
            fitted = ImageOps.contain(
                source_img,
                (thumb_w - 8, thumb_h - 8),
                method=Image.Resampling.LANCZOS,
            )
            px = (thumb_w - fitted.width) // 2
            py = (thumb_h - fitted.height) // 2
            panel.paste(fitted, (px, py))

        img_y = y_top + title_h
        canvas.paste(panel, (x, img_y))
        draw.rectangle((x, img_y, x + thumb_w - 1, img_y + thumb_h - 1), outline=(200, 200, 200), width=1)

        draw.multiline_text(
            (x, img_y + thumb_h + 8),
            col["wrapped_text"],
            fill=(35, 35, 35),
            font=font_body,
            spacing=2,
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas.save(output_path)
