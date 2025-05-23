#%%
import json
import random

path1 = "/NAS/zhangyi/shenc/AlphaSteer/data/instructions/train_val/harmful_train_1000.json"
path2 = "/NAS/zhangyi/shenc/NullSpaceSteering/data/instructions/circuit_breakers_train.json"

# Load data from both files
with open(path1, 'r', encoding='utf-8') as f:
    data1 = json.load(f)

with open(path2, 'r', encoding='utf-8') as f:
    data2 = json.load(f)

# Calculate how many samples to take from data2
samples_needed = 1000 - len(data1)

# Sample from data2 if we need more data
if samples_needed > 0:
    sampled_data2 = random.sample(data2, min(samples_needed, len(data2)))
    # Transform data2 format: extract "prompt" field and store as "query"
    transformed_data2 = [{"query": item["prompt"]} for item in sampled_data2]
    # Combine data1 with transformed data from data2
    combined_data = data1 + transformed_data2
else:
    combined_data = data1

# Save the combined data back to path1
with open(path1, 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, ensure_ascii=False, indent=4)
