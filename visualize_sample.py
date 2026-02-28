import json
import matplotlib.pyplot as plt
from PIL import Image
import textwrap
import os
import argparse

def wrap_text(text, width=60):
    return "\n".join(textwrap.wrap(text, width=width))

def visualize_sample(jsonl_path, image_folder, output_path, sample_idx=0):
    # Read the first sample
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i == sample_idx:
                sample = json.loads(line)
                break
    
    steps = sample.get("steps", [])
    num_steps = len(steps)
    if num_steps == 0:
        print("No steps found in the sample.")
        return
        
    fig, axes = plt.subplots(1, num_steps, figsize=(5 * num_steps, 8))
    if num_steps == 1:
        axes = [axes]
        
    fig.suptitle(f"Question: {sample.get('question', '')}\nGT Answer: {sample.get('answer', '')}", fontsize=14, fontweight='bold')
    
    for i, step in enumerate(steps):
        ax = axes[i]
        
        # Load image
        img_path = step.get("view_image", "")
        if not os.path.isabs(img_path) and image_folder:
            img_path = os.path.join(image_folder, img_path)
            
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f"Image not found:\n{img_path.split('/')[-1]}", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            
        # Parse text details
        action = step.get("action")
        thinking = step.get("thinking", "")
        
        action_text = f"Action: {action}" if action else "Action: STOP"
        
        # Format thinking text with wrapping
        wrapped_thinking = wrap_text(thinking, width=50)
        
        caption = f"Step {i+1}\n\n{action_text}\n\nThinking:\n{wrapped_thinking}"
        
        # Add text below image
        ax.text(0.5, -0.1, caption, ha='center', va='top', transform=ax.transAxes, 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
                
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Save the figure
    print(f"Saving visualization to {output_path}")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()
    
    visualize_sample(args.input, args.image_folder, args.output, args.idx)
