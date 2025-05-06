import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score


def read(filepath, _type):
    """Read data from a CSV file and add binary columns based on type.

    Args:
        filepath (str): Path to the CSV file to read.
        _type (str): Type of data to read. If "tdcs", adds Valid=1, Task=1, tdcs=1 columns.
            Otherwise adds tdcs=0.

    Returns:
        np.ndarray: NumPy array containing the data from the CSV file with added columns.
    """
    df = pd.read_csv(filepath)
    if _type == "tdcs":
        df["Valid"] = 1
        df["Task"] = 1
        df["tdcs"] = 1
    else:
        df["tdcs"] = 0

    return np.array(df)


def calculate_metric(config, pp_out, val_df, pre):
    """Calculate mean average precision (mAP) score for predictions across three event types.
    This function compares model predictions against ground truth labels for three types of events:
    start hesitation, turn, and walking. It computes the average precision for each event type
    and returns their mean as the final metric.
    Args:
        config: Configuration object containing model settings
        pp_out (dict): Model predictions containing 'logits' key with shape [N, 3]
        val_df (pandas.DataFrame): DataFrame containing file paths for validation data
        pre: Preprocessing object (unused in current implementation)
    Returns:
        float: Mean Average Precision (mAP) score across all three event types
    Notes:
        - Input data is expected to have ground truth labels in columns 4-6
        - Data is cast to float16 to optimize memory usage
        - The three event types evaluated are:
            1. Start hesitation
            2. Turn
            3. Walking
    """

    dfs = [read(row[0], row[1]) for row in val_df.itertuples(index=False)]
    dfs = np.concatenate(dfs, axis=0)
    dfs = torch.tensor(dfs[: pp_out["logits"].shape[0], 4:7].astype(np.float16))  # Cast to 16-bit to save memory

    average_precision_start_hesitation = average_precision_score(dfs[:, 0], pp_out["logits"][:, 0].cpu())
    average_precision_turn = average_precision_score(dfs[:, 1], pp_out["logits"][:, 1].cpu())
    average_precision_walking = average_precision_score(dfs[:, 2], pp_out["logits"][:, 2].cpu())

    map_score = np.mean([average_precision_start_hesitation, average_precision_turn, average_precision_walking])
    return map_score
