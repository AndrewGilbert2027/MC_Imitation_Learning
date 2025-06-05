import cv2
import os

def validate_frames_actions(video_file, action_file):
    # Check if files exist
    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' does not exist.")
        return
    if not os.path.exists(action_file):
        print(f"Error: Action file '{action_file}' does not exist.")
        return

    # Count actions in JSONL file
    action_count = 0
    with open(action_file, 'r') as f:
        for line in f:
            action_count += 1

    # Open video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_file}'.")
        return

    # Count frames in the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Compare frame count with action count
    if frame_count == action_count:
        print("Validation successful: Frame count matches action count.")
    else:
        print(f"Validation failed: {frame_count} frames but {action_count} actions.")
        if frame_count > action_count:
            print("Warning: Missing actions for some frames.")
        else:
            print("Warning: Extra actions without corresponding frames.")

if __name__ == "__main__":
    # Paths to video and action files
    video_file = "./data/labeller-training/mc-0.mp4"
    action_file = "./data/labeller-training/mc-0.jsonl"

    validate_frames_actions(video_file, action_file)
