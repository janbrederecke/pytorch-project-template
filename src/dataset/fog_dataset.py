import gc
import os
import random
import time

import numpy as np
import pandas as pd
import torch
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
    """A custom PyTorch Dataset for loading and processing FOG data.

    This dataset class handles loading and preprocessing of Parkinson's Freezing of Gait (FOG) data
    from CSV files. It supports training, testing and inference modes with different data handling
    approaches for each mode.

    Args:
        df (pandas.DataFrame): DataFrame containing file paths and data type information
        config (Config): Configuration object containing model parameters
        augmentations (dict): Dictionary of data augmentation functions to apply
        mode (str): Operating mode - one of ['train', 'test', 'inference', 'submission']

    Attributes:
        augmentations (dict): Stored augmentation functions
        mode (str): Operating mode
        config (Config): Stored configuration
        df (pandas.DataFrame): Input DataFrame
        dfs (numpy.ndarray): Concatenated data from all input files, normalized to [0,1]
        f_ids (list): List of file IDs extracted from file paths
        end_indices (list): Cumulative ending indices for each input file
        shapes (list): Original shapes of each input file
        length (int): Total number of rows in concatenated data

    Methods:
        read(filepath, _type): Reads and preprocesses a single CSV file
        __getitem__(index): Returns a single sample with its features and target
        __len__(): Returns the total length of the dataset

    Returns:
        For training/validation:
            dict: {'input': tensor(Window_size x 3),
                   'target': tensor(3),
                   'time': tensor(1)}

        For submission:
            dict: {'input': tensor(Window_size x 3),
                   'id': str,
                   'time': tensor(1)}
    """

    def __init__(self, df, config, augmentations, mode):
        super().__init__()

        tm = time.time()
        self.augmentations = augmentations
        self.mode = mode
        self.config = config

        self.df = df
        self.dfs = [self.read(row[0], row[1]) for row in df.itertuples(index=False)]

        self.f_ids = [os.path.basename(row[0])[:-4] for row in self.df.itertuples(index=False)]

        self.end_indices = []
        self.shapes = []
        _length = 0
        for df in self.dfs:
            self.shapes.append(df.shape[0])
            _length += df.shape[0]
            self.end_indices.append(_length)

        self.dfs = np.concatenate(self.dfs, axis=0)
        self.dfs = (self.dfs - self.dfs.min()) / (self.dfs.max() - self.dfs.min())  # Normalize to [0, 1]
        self.dfs = self.dfs.astype(np.float16)  # Cast to 16-bit to save memory
        self.length = self.dfs.shape[0]

        shape1 = self.dfs.shape[1]

        self.dfs = np.concatenate(
            [
                np.zeros((self.config.WINDOW_PAST, shape1)),
                self.dfs,
                np.zeros((self.config.WINDOW_FUTURE, shape1)),
            ],
            axis=0,
        )
        config.logger.info(f"Dataset initialized in {time.time() - tm} secs!")
        gc.collect()

    def read(self, filepath, _type):
        df = pd.read_csv(filepath)
        if self.mode == "test":
            return np.array(df)

        if _type == "tdcs":
            df["Valid"] = 1
            df["Task"] = 1
            df["tdcs"] = 1
        else:
            df["tdcs"] = 0

        return np.array(df)

    def __getitem__(self, index):
        if self.mode == "train":
            row_idx = random.randint(0, self.length - 1) + self.config.WINDOW_PAST
        elif self.mode in ["test", "inference"]:
            for i, e in enumerate(self.end_indices):
                if index >= e:
                    continue
                df_idx = i
                break

            row_idx_true = self.shapes[df_idx] - (self.end_indices[df_idx] - index)
            _id = self.f_ids[df_idx] + "_" + str(row_idx_true)
            row_idx = index + self.config.WINDOW_PAST
        else:
            row_idx = index + self.config.WINDOW_PAST

        x = self.dfs[row_idx - self.config.WINDOW_PAST : row_idx + self.config.WINDOW_FUTURE, 1:4]

        x = torch.tensor(x.astype("float32")).T

        t = self.dfs[row_idx, -3] * self.dfs[row_idx, -2]
        t = torch.tensor(t).half().view(1)

        if self.mode in ["submission"]:
            feature_dict = {"input": x, "id": _id, "time": t}
            return feature_dict

        y = self.dfs[row_idx, 4:7].astype("float32")
        y = torch.tensor(y)

        feature_dict = {"input": x, "target": y, "time": t}

        return feature_dict

    def __len__(self):
        if self.mode == "train":
            return self.length
        return self.length
