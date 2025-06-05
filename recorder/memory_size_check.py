import os
import json

def get_file_size_in_mb(file_path):
    """Returns the size of a file in megabytes."""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def main():
    mp4_file_path = "./data/labeller-training/mc-0.mp4"  # Path to your MP4 file
    json_file_path = "./data/labeller-training/mc-0.jsonl"  # Path to your JSON file

    try:
        mp4_size_mb = get_file_size_in_mb(mp4_file_path)
        print(f"MP4 file size: {mp4_size_mb:.2f} MB")
    except FileNotFoundError as e:
        print(e)

    try:
        json_size_mb = get_file_size_in_mb(json_file_path)
        print(f"JSON file size: {json_size_mb:.2f} MB")
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()