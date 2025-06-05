import json
import os

def convert_jsonl_to_json(jsonl_file, json_file):
    """
    Converts a JSONL file to a JSON file.

    Args:
        jsonl_file (str): Path to the input JSONL file.
        json_file (str): Path to the output JSON file.
    """
    actions = {}
    with open(jsonl_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            frame = entry["frame"]
            actions[frame] = entry["action"]

    with open(json_file, 'w') as f:
        json.dump(actions, f)

    print(f"Converted '{jsonl_file}' to '{json_file}'.")
    os.remove(jsonl_file)  # Optionally remove the original JSONL file after conversion

# Example usage
if __name__ == "__main__":
    for i in range(11, 30):
        convert_jsonl_to_json(
            f"./data/labeller-training/mc-{i}.jsonl",
            f"./data/labeller-training/mc-{i}.json"
        )
