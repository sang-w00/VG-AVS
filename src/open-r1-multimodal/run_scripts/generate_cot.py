import json
import os
import base64
import requests
import argparse
from tqdm import tqdm

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def encode_image(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



def convert_gt_action_to_text(action, visibility_level=None):
    if visibility_level and visibility_level.lower() == "undecidable":
        return "<unknown>"
    elif visibility_level and visibility_level.lower() == "visible":
        return "<stop>"
    if action is None:
        return "<stop>"
    
    rot0 = int(action[0])
    if rot0 > 180: rot0 -= 360
    view0 = int(action[2])
    if view0 > 180: view0 -= 360
    return f"<head> {rot0} </head> <fwd> {int(action[1])} </fwd> <view> {view0} </view>"


def process_example(example, image_folder):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable is required")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    question = example["question"]
    steps = example["steps"]
    target_vis_level = example.get("target_object_visibility_level", "")
    
    new_steps = []
    
    for i, step in enumerate(steps):
        img_path = step.get("view_image")
        if img_path and not os.path.isabs(img_path) and image_folder:
            img_path = os.path.join(image_folder, img_path)
            
        base64_img = encode_image(img_path) if img_path else None
        
        step_vis_level = step.get("target_object_visibility_level", target_vis_level)
        gt_action_str = convert_gt_action_to_text(step.get("action"), step_vis_level)
        
        user_content = []
        if base64_img:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}",
                     "detail": "low"
                }
            })
        
        text_prompt = f"Observation.\nGround Truth Action: {gt_action_str}\nProvide your thinking process:"
        user_content.append({
            "type": "text", 
            "text": text_prompt
        })
        
        # Create a fresh messages list for each step (System + Current User Observation)
        step_messages = [
            {
                "role": "system",
                "content": (
                    "You are an annotation assistant for an embodied navigation dataset. "
                    f"Your objective is to determine the action required to answer the question: '{question}'. You will receive an image observation for the current step. "
                    "I will give you the observation and the Ground Truth action you must take.\n\n"
                    "Action space includes:\n"
                    "- <head>: Azimuth rotation (yaw) in degrees before moving. Positive is right, negative is left.\n"
                    "- <fwd>: Forward movement distance (cm).\n"
                    "- <view>: Azimuth rotation (yaw) in degrees after moving. Positive is right, negative is left.\n"
                    "- <stop>: Stop and end the episode if the answer is completely visible.\n"
                    "- <unknown>: The current view lacks query-relevant visual cues, so the appropriate action cannot be determined.\n\n"
                    "Note: There is NO tilt or pitch action. Both head and view are strictly left/right azimuth (yaw) rotations.\n\n"
                    "You need to provide the 'thinking' process (about 3 short sentences) that explains "
                    "why you are taking this Ground Truth action based on what you see, what you are trying to find, and your goal.\n"
                    "Respond ONLY with the thinking content."
                )
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        payload = {
            "model": "gpt-5-mini-2025-08-07",
            "messages": step_messages
        }
        
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            raw_content = result['choices'][0]['message'].get('content', '')
            print(f"API Success. Raw content: {repr(raw_content)}")
            thinking = raw_content.strip() if raw_content else ""
        except Exception as e:
            print(f"API Error at step {i}: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"Response text: {response.text}")
            thinking = ""
            
        step["thinking"] = thinking
        new_steps.append(step)
        
    example["steps"] = new_steps
    return example

def process_file(input_file, output_file, image_folder, max_items=None):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    if max_items:
        lines = lines[:max_items]
        
    print(f"Processing {len(lines)} items...")
    
    with open(output_file, 'w') as f_out:
        for idx, line in enumerate(lines):
            print(f"Processing line {idx+1}/{len(lines)}...")
            example = json.loads(line)
            example = process_example(example, image_folder)
            f_out.write(json.dumps(example) + '\n')
            f_out.flush()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--max_items", type=int, default=None, help="Limit number of items for testing")
    args = parser.parse_args()
    
    process_file(args.input, args.output, args.image_folder, args.max_items)
