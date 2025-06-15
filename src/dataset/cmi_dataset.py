import gc
import random
import time

import numpy as np
import scipy.interpolate
import torch
from braindecode.augmentation import GaussianNoise
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


def batch_to_device(batch, device):
    """
    Move all tensors in a batch dictionary to the specified device.

    Args:
        batch (dict): Dictionary containing tensors as values
        device (torch.device): The device to move the tensors to (e.g., 'cuda' or 'cpu')

    Returns:
        dict: Dictionary with all tensors moved to the specified device
    """
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


def collate(batch):
    """
    Collates a batch of samples into a single batch dictionary.

    This function takes a list of dictionaries (samples) and stacks their values
    along a new batch dimension, creating a single dictionary where each key maps
    to a batched tensor.

    Args:
        batch (List[Dict[str, torch.Tensor]]): A list of dictionaries, where each dictionary
            contains tensors as values with matching keys across all samples.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing batched tensors, where each key
            from the input samples maps to a tensor of shape [batch_size, ...].

    Example:
        >>> samples = [
        ...     {'x': torch.tensor([1]), 'y': torch.tensor([2])},
        ...     {'x': torch.tensor([3]), 'y': torch.tensor([4])}
        ... ]
        >>> collated = collate(samples)
        >>> collated
        {'x': tensor([[1], [3]]), 'y': tensor([[2], [4]])}
    """
    keys = batch[0].keys()
    batch_dict = {key: torch.stack([b[key] for b in batch]) for key in keys}
    return batch_dict


collate_train = collate
collate_valid = collate


class CustomDataset(Dataset):
    """A custom PyTorch Dataset for loading and processing CMI data."""

    def __init__(self, df, config, augmentations, mode):
        super().__init__()

        tm = time.time()
        self.augmentations = augmentations
        self.mode = mode
        self.config = config

        self.df = df.reset_index()
        self.window_past = np.zeros((config.WINDOW_PAST, config.INPUT_DIM[0]))
        self.window_future = np.zeros((config.WINDOW_FUTURE, config.INPUT_DIM[0]))
        self.half_window_past = self.config.WINDOW_PAST // 2

        self.dfs = [self.read(row[-2]) for row in df.itertuples(index=False)]

        self.dfs = [np.concatenate([self.window_past, df, self.window_future], axis=0) for df in self.dfs]
        self.sequence_ids = df["sequence_id"].to_list()
        self.sequence_lengths = torch.tensor(df["count"].to_numpy())
        self.sequence_lengths_expanded = self.sequence_lengths + self.config.WINDOW_SIZE
        self.sequence_middles = self.sequence_lengths_expanded // 2
        self.length = len(df)

        self.df["target1"] = (self.df["sequence_type"] == "Target") * 1.0

        config.logger.info(f"Dataset initialized in {time.time() - tm} secs!")
        gc.collect()

    def __getitem__(self, index):
        row_idx = self.sequence_middles[index]

        x = self.dfs[index][row_idx - self.config.WINDOW_PAST : row_idx + self.config.WINDOW_FUTURE, :].T

        x = torch.tensor(x.astype("float32"))

        if self.mode == "test":
            feature_dict = {"input": x, "id": self.sequence_ids[index]}
            return feature_dict

        y = self.df.loc[index, "target"].astype("float32")
        y = torch.tensor(y)

        y1 = self.df.loc[index, "target1"].astype("float32")
        y1 = torch.tensor(y1)

        feature_dict = {"input": x, "target": y, "target1": y1}

        return feature_dict

    def __len__(self):
        return self.length

    def read(self, filepath):
        return np.load(f"./data/preprocessed/sequences/{filepath}.npy")
