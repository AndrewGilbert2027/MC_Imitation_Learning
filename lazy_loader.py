import torch
import json
import cv2
import numpy as np
import os
from typing import Tuple, List, Dict

class LazyLoader:
    def __init__(self, data_dir: str):
        """
        Initialize the lazy loader with a directory containing video and json files.
        
        Args:
            data_dir (str): Path to directory containing the data files
        """
        self.data_dir = data_dir
        self.video_pairs = []  # List of (video_path, json_path) pairs
        self.sample_pairs = []  # List of (video_path, json_path, frame_idx) tuples
        self._initialize_pairs()

    def _initialize_pairs(self):
        """Initialize video pairs and sample pairs."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # First, collect all video-json pairs
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.mp4'):
                video_path = os.path.join(self.data_dir, filename)
                json_path = os.path.join(self.data_dir, filename.replace('.mp4', '.json'))
                
                if os.path.exists(json_path):
                    self.video_pairs.append((video_path, json_path))
                    
                    # Get number of frames in video
                    cap = cv2.VideoCapture(video_path)
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    # Create sample pairs for each valid frame index
                    # Start from 19 to ensure we have enough previous frames
                    for frame_idx in range(19, n_frames):
                        self.sample_pairs.append((video_path, json_path, frame_idx))
                else:
                    print(f"Warning: No JSON file found for {filename}")

        if not self.video_pairs:
            raise FileNotFoundError(f"No valid video-json pairs found in {self.data_dir}")

    def load_sequence(self, video_path: str, frame_idx: int, sequence_length: int = 20) -> torch.Tensor:
        """Load a sequence of frames from the video."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Load previous frames and current frame
        start_idx = max(0, frame_idx - sequence_length + 1)
        for idx in range(start_idx, frame_idx + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
            frame = cv2.resize(frame, (256, 256))
            frames.append(frame / 255.0)
        cap.release()

        # Pad with zeros if we don't have enough previous frames
        while len(frames) < sequence_length:
            frames.insert(0, np.zeros((256, 256, 3), dtype=np.float32))

        return torch.tensor(np.stack(frames, axis=0), dtype=torch.float32).permute(3, 0, 1, 2)

    def load_action(self, json_path: str, frame_idx: int) -> torch.Tensor:
        """Load the action for the specified frame."""
        with open(json_path, 'r') as f:
            actions = json.load(f)
        action = actions[frame_idx]
        return torch.tensor(self._action_to_feature_vector(action), dtype=torch.float32)

    def _action_to_feature_vector(self, action: Dict) -> np.ndarray:
        """Convert action dictionary to feature vector."""
        feature_vector = []
        binary_vals = ['forward', 'back', 'left', 'right', 'sprint', 'jump', 'attack']
        for key in binary_vals:
            feature_vector.append(1 if key in action and action[key] else 0)
        
        # Camera movement directions
        camera = action.get('camera', [0.0, 0.0])
        feature_vector.extend([
            1 if camera[1] < -0.001 else 0,  # Look up
            1 if camera[1] > 0.001 else 0,   # Look down
            1 if camera[0] < -0.001 else 0,  # Look left
            1 if camera[0] > 0.001 else 0,   # Look right
        ])
        
        return np.array(feature_vector, dtype=np.float32)

    def __len__(self):
        """Return the total number of samples."""
        return len(self.sample_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample by index."""
        video_path, json_path, frame_idx = self.sample_pairs[idx]
        frames = self.load_sequence(video_path, frame_idx)
        action = self.load_action(json_path, frame_idx)
        return frames, action
