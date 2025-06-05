import json
import os

def clean_json(json_file):
    """
    Cleans a JSON file by removing any keys that are not 'frame' or 'action'.
    
    Args:
        json_file (str): Path to the input JSON file.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    cleaned_data = []
    for frame, action in data.items():
        cleaned_data[frame] = {k: v for k, v in action.items() if k in ['frame', 'action']}

    with open(json_file, 'w') as f:
        json.dump(cleaned_data, f, indent=4)

    print(f"Cleaned '{json_file}'.")