import argparse
import importlib
import os
import sys
from copy import deepcopy

from src.training.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with the specified configuration.")
    parser.add_argument(
        "-F",
        "--fold",
        required=True,
        help="Choose fold for training, -1 results in training on all training data and validation on fold 0.",
    )
    parser.add_argument(
        "-C",
        "--config",
        required=True,
        help="Configuration module name (without .py extension).",
    )

    args = parser.parse_args()

    try:
        config_module = importlib.import_module(f"configs.{args.config}")
        config = deepcopy(config_module.config)
    except ModuleNotFoundError:
        print(f"Error: Configuration module '{args.config}' not found.")
        sys.exit(1)

    config.FOLD = int(args.fold)
    config.CONFIG = args.config
    config.CustomDataset = importlib.import_module(config.DATASET).CustomDataset
    config.collate_train = importlib.import_module(config.DATASET).collate_train
    config.collate_valid = importlib.import_module(config.DATASET).collate_valid
    config.batch_to_device = importlib.import_module(config.DATASET).batch_to_device
    config.post_processing_pipeline = importlib.import_module(config.POST_PROCESSING_PIPELINE).post_processing_pipeline
    config.calculate_metric = importlib.import_module(config.METRIC).calculate_metric

    os.makedirs(str(config.OUTPUT_DIRECTORY + f"/{config.CONFIG}/fold{config.FOLD}/"), exist_ok=True)

    result = train(config)

    print(result)
