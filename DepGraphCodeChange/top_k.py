import json
import numpy as np
import sys
import os

project_name = sys.argv[1]
model_name = sys.argv[2]

def calculate_metrics(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    top1, top3, top5, top10 = 0, 0, 0, 0
    mfr_list, mar_list = [], []

    for project, info in data.items():
        ranking = info['ranking']
        ground_truth = info['ans']

        min_rank, ranks = float('inf'), []

        for gt in ground_truth:
            if gt in ranking:
                rank = ranking.index(gt)
                ranks.append(rank)
                min_rank = min(min_rank, rank)

        mfr_list.append(min_rank)
        mar_list.append(np.mean(ranks))

        # Update top-k counts based on minimum rank
        if min_rank == 0:
            top1 += 1
        if min_rank < 3:
            top3 += 1
        if min_rank < 5:
            top5 += 1
        if min_rank < 10:
            top10 += 1

    mfr_avg = np.mean(mfr_list)
    mar_avg = np.mean(mar_list)

    return {
        'Top-1': top1,
        'Top-3': top3,
        'Top-5': top5,
        'Top-10': top10,
        'MFR': mfr_avg,
        'MAR': mar_avg
    }

# Usage example
json_file = f'crossvalidation/{project_name}/{model_name}/{project_name}_merged_data.json'
metrics = calculate_metrics(json_file)

# Output directory and file
output_dir = 'dataleakage'
output_file = os.path.join(output_dir, f'{model_name}_cross.json')

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Check if the file already exists and load it
if os.path.exists(output_file):
    with open(output_file, 'r') as file:
        existing_data = json.load(file)
else:
    existing_data = {}

# Update the data with new metrics
if model_name not in existing_data:
    existing_data[model_name] = {}
existing_data[model_name][project_name] = metrics

# Write the updated data back to the file
with open(output_file, 'w') as file:
    json.dump(existing_data, file, indent=4)

print(f"Metrics saved to {output_file}")
