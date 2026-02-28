import json

input_file = "/home/andy2884/workspace/VG-AVS/data/avs_existence_train_final_with_cot.jsonl"
output_file = "/home/andy2884/workspace/VG-AVS/data/avs_existence_train_final_with_cot_multiturn.jsonl"

filtered_count = 0
total_count = 0

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        if not line.strip():
            continue
        total_count += 1
        data = json.loads(line)
        
        if 'steps' in data:
            steps = data['steps']
            if len(steps) >= 3:
                # Count actions that are not null and not 'stop'
                action_count = sum(1 for step in steps if step.get('action') is not None and str(step.get('action')).lower() != 'stop')
                if action_count >= 2:
                    fout.write(line)
                    filtered_count += 1

print(f"Total rows: {total_count}")
print(f"Filtered rows: {filtered_count}")
print(f"Saved to: {output_file}")
