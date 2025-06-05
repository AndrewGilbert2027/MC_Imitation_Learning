import json
import os

def convert_json(input_file, output_file):
    """
    Converts a JSON file from the current format to the desired format.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file.
    """
    # Load the original JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Convert the data to the desired format
    converted_data = [value for key, value in data.items()]

    # Save the converted data to the output file
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=4)

    print(f"Converted JSON saved to: {output_file}")

# Example usage
if __name__ == "__main__":

    for i in range(11, 30):
        convert_json(
            f"/home/andrew-gilbert/minecraft_imitation/vpt/data/labeller-training/mc-{i}.json",
            f"/home/andrew-gilbert/minecraft_imitation/vpt/data/labeller-training/mc-{i}.json"
        )
