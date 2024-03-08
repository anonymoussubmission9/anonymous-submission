import os
import pickle
import json
import sys

def read_and_merge_pkl_files(folder_path, output_json_file, additional_pkl_file, project_name):
    merged_data = {}

    # Read additional .pkl file first
    with open(additional_pkl_file, 'rb') as file:
        additional_data = pickle.load(file)
        additional_info = {d['proj']: d['ans'] for d in additional_data}

    # Iterate over all files in the given folder
    for file in os.listdir(folder_path):
        if file.endswith('.pkl'):
            file_path = os.path.join(folder_path, file)

            # Read the .pkl file
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

                # Assuming the data is a tuple with the format (key, value)
                if isinstance(data, tuple) and len(data) == 2:
                    # Merge additional info into each project's data
                    project_key = data[0]
                    merged_data[project_key] = data[1]
                    if project_key in additional_info:
                        merged_data[project_key]['ans'] = additional_info[project_key]

    # Write merged data to a JSON file
    with open(output_json_file, 'w') as json_file:
        json.dump(merged_data, json_file, indent=4)

    print(f'Merged data written to {output_json_file}')

# Example usage
project_name = sys.argv[1]
trained_model = sys.argv[2]
folder_path = f'crossvalidation/{project_name}/{trained_model}'  # Folder with .pkl files
additional_pkl_file = f'{project_name}.pkl'  # Path to the additional .pkl file
output_json_file = f'crossvalidation/{project_name}/{trained_model}/{project_name}_merged_data.json'
read_and_merge_pkl_files(folder_path, output_json_file, additional_pkl_file, project_name)
