import cv2
import os
import json
import numpy as np

"""
Numpy output format (feature vector): (11,)
------------------------------------
forward       : 1 if moving forward, else 0
back          : 1 if moving backward, else 0
left          : 1 if moving left, else 0
right         : 1 if moving right, else 0
sprint        : 1 if sprinting, else 0
jump          : 1 if jumping, else 0
attack        : 1 if attacking, else 0
look_up       : 1 if looking up, else 0
look_down     : 1 if looking down, else 0
look_left     : 1 if looking left, else 0
look_right    : 1 if looking right, else 0
------------------------------------
"""

def action_to_feature_vector(action):
    """
    Converts an action dictionary to a numpy feature vector based on a fixed key order.
    """
    feature_vector = []
    binary_vals = ['forward', 'back', 'left', 'right', 'sprint', 'jump', 'attack']
    for key in binary_vals:
        feature_vector.append(1 if key in action and action[key] else 0)
    
    # Camera movement directions (binary values)
    feature_vector.append(1 if action.get('camera', [0.0, 0.0])[1] < -0.001 else 0)  # Look up
    feature_vector.append(1 if action.get('camera', [0.0, 0.0])[1] > 0.001 else 0)   # Look down
    feature_vector.append(1 if action.get('camera', [0.0, 0.0])[0] < -0.001 else 0)  # Look left
    feature_vector.append(1 if action.get('camera', [0.0, 0.0])[0] > 0.001 else 0)   # Look right

    return np.array(feature_vector, dtype=np.float32)

def preprocess_data(data_dir, output_dir):
    """
    Preprocess multiple video-action file pairs and save intermediate results to reduce memory usage.

    Args:
        data_dir (str): Path to the directory containing video and action files.
        output_dir (str): Path to the directory where preprocessed data will be saved.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_inputs = []
    all_targets = []

    # Iterate through all video-action file pairs
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".mp4"):
            video_file = os.path.join(data_dir, file_name)
            action_file = os.path.join(data_dir, file_name.replace(".mp4", ".jsonl"))

            # Check if video file exists
            if not os.path.exists(video_file):
                print(f"Error: Video file '{video_file}' does not exist.")
                continue

            # Check if corresponding action file exists
            if not os.path.exists(action_file):
                print(f"Warning: Action file '{action_file}' not found for video '{video_file}'. Skipping.")
                continue

            # Open video file
            print(f"Attempting to open video file: {video_file}")
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print(f"Error: Could not open video file '{video_file}'. Skipping.")
                continue

            # Load actions from JSON file
            print(f"Loading actions from file: {action_file}")
            actions = []
            with open(action_file, 'r') as f:
                actions = json.load(f) # Assuming the file is in JSON format, not JSONL

            # Extract frames from the video
            print(f"Extracting frames from video: {video_file}")
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Resize frame to 256x256 and normalize RGB values
                frame = cv2.resize(frame, (256, 256))  # Resize to 256x256
                frames.append(frame / 255.0)  # Normalize pixel values to [0, 1]
            cap.release()

            # Validate frame and action counts
            print(f"Validating frame and action counts for '{file_name}'...")
            if len(frames) != len(actions):
                print(f"Error: Mismatch between frame count ({len(frames)}) and action count ({len(actions)}). Skipping.")
                continue

            # Generate input-output pairs
            print(f"Generating input-output pairs for '{file_name}'...")
            inputs = []
            targets = []
            for i in range(19, len(frames)):  # Updated range to start from 19 for 20-frame depth
                # Input: Previous 19 frames + current frame
                input_frames = np.stack(frames[i-19:i+1], axis=-1)  # Shape: (256, 256, 60) for RGB (20 frames x 3 channels)
                inputs.append(input_frames)
                # Target: Convert current frame's action dictionary to feature vector
                feature_vector = action_to_feature_vector(actions[i])
                targets.append(feature_vector)

            # Save intermediate results
            print(f"Saving preprocessed data for '{file_name}'...")
            np.save(os.path.join(output_dir, f"{file_name}_inputs.npy"), np.array(inputs))
            np.save(os.path.join(output_dir, f"{file_name}_targets.npy"), np.array(targets))
            print(f"Processed and saved data for '{file_name}'.")

    print("Preprocessing complete. Intermediate results saved.")

if __name__ == "__main__":
    # Paths to data directory and output directory
    data_dir = "./data/labeller-training"
    output_dir = "./data/preprocessed"

    preprocess_data(data_dir, output_dir)
