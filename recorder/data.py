import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PreprocessedDataset(Dataset):
    def __init__(self, data_dir):
        """
        Initializes the dataset by loading preprocessed inputs and targets.

        Args:
            data_dir (str): Path to the directory containing 'inputs.npy' and 'targets.npy'.
        """
        self.inputs_path = os.path.join(data_dir, "inputs.npy")
        self.targets_path = os.path.join(data_dir, "targets.npy")

        # Load data
        if not os.path.exists(self.inputs_path) or not os.path.exists(self.targets_path):
            raise FileNotFoundError(f"Preprocessed data files not found in {data_dir}")
        
        self.inputs = np.load(self.inputs_path)
        self.targets = np.load(self.targets_path)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Returns a single sample (input, target) from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (input_tensor, target_tensor)
        """
        input_tensor = torch.tensor(self.inputs[idx], dtype=torch.float32)
        target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
        return input_tensor, target_tensor
