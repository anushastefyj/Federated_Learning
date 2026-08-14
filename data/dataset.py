import torch
from torch.utils.data import Dataset
import numpy as np

class AirQualityDataset(Dataset):
    """
    PyTorch Dataset for time-series forecasting.
    Takes an array of shape (time_steps, features) and creates
    sliding windows of `seq_length` to predict `pred_horizon` steps ahead.
    """
    def __init__(self, data: np.ndarray, seq_length: int, pred_horizon: int = 1, target_col_idx: int = 0):
        """
        Args:
            data: Numpy array of shape (N, features)
            seq_length: Number of past time steps to use as input
            pred_horizon: Number of time steps ahead to predict
            target_col_idx: Index of the feature to predict (default 0, e.g. PM2.5)
        """
        self.data = data
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon
        self.target_col_idx = target_col_idx
        
        # Calculate how many valid windows we can extract
        self.num_samples = len(data) - seq_length - pred_horizon + 1

    def __len__(self):
        return max(0, self.num_samples)

    def __getitem__(self, idx):
        # Extract input sequence
        x = self.data[idx : idx + self.seq_length]
        
        # Extract target value(s)
        # We predict a single value `pred_horizon` steps ahead from the end of the sequence
        y = self.data[idx + self.seq_length + self.pred_horizon - 1, self.target_col_idx]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
