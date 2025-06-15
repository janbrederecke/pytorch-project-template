import math
from collections import defaultdict

import mlflow
import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

from src.utils import sync_across_gpus


def evaluate(model, val_dataloader, config, pre="val", current_epoch=0):
    """
    Evaluates a model on a validation dataset and logs metrics.
    This function performs model evaluation, calculates metrics, handles distributed training,
    and optionally saves validation data and predictions.
    Args:
        model: PyTorch model to evaluate
        val_dataloader: DataLoader containing validation data
        config: Configuration object containing evaluation settings and parameters
        pre (str, optional): Prefix for logging metrics. Defaults to "val"
        current_epoch (int, optional): Current training epoch. Defaults to 0
    Returns:
        dict or float: Validation score(s). Returns dict of scores if multiple metrics,
                      otherwise returns single score value
    Key Features:
        - Supports mixed precision evaluation
        - Handles distributed evaluation across multiple GPUs
        - Can save validation predictions and data
        - Calculates and logs various loss metrics
        - Optional post-processing pipeline
        - Progress bar with tqdm
        - MLflow metric logging
    Notes:
        - Expects model outputs as dictionary containing loss/score keys
        - For distributed training, synchronizes data across GPUs if EVAL_DDP is True
        - Metrics are calculated every N epochs as specified in config
        - First batch predictions can be saved if SAVE_FIRST_BATCH_PREDS is True
    """

    saved_images = False
    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    val_data = defaultdict(list)
    val_score = 0
    for _, data in enumerate(tqdm(val_dataloader, disable=(config.LOCAL_RANK != 0) | config.DISABLE_TQDM, ascii=True)):
        batch = config.batch_to_device(data, config.DEVICE)

        if config.MIXED_PRECISION:
            with autocast("cuda"):
                output = model(batch)
        else:
            output = model(batch)

        if (
            (config.LOCAL_RANK == 0)
            and (config.calculate_metric)
            and (((current_epoch + 1) % config.CALCULATE_METRIC_EVERY_N_EPOCHS) == 0)
        ):
            # per batch calculations not implemented
            pass

        if (not saved_images) & (config.SAVE_FIRST_BATCH_PREDS):
            # config.save_first_batch_preds(batch, output, config)
            saved_images = True

        for key, _ in output.items():
            val_data[key] += [output[key]]

    for key, _ in output.items():
        value = val_data[key]
        if isinstance(value[0], list):
            val_data[key] = [item for sublist in value for item in sublist]

        else:
            if len(value[0].shape) == 0:
                val_data[key] = torch.stack(value)
            else:
                val_data[key] = torch.cat(value, dim=0)

    if (
        (config.LOCAL_RANK == 0)
        and (config.calculate_metric)
        and (((current_epoch + 1) % config.CALCULATE_METRIC_EVERY_N_EPOCHS) == 0)
    ):
        pass

    if config.DISTRIBUTED and config.EVAL_DDP:
        for key, _ in output.items():
            val_data[key] = sync_across_gpus(val_data[key], config.WORLD_SIZE)

    if config.LOCAL_RANK == 0:
        if config.SAVE_VAL_DATA:
            if config.DISTRIBUTED:
                for k, v in val_data.items():
                    val_data[k] = v[: len(val_dataloader.dataset)]
            torch.save(
                val_data,
                f"{config.OUTPUT_DIRECTORY}/{config.CONFIG}/fold{config.FOLD}/{pre}_data_seed{config.SEED}.pth",
            )

    loss_names = [key for key in output if "loss" in key]
    loss_names += [key for key in output if "score" in key]
    for loss_name in loss_names:
        if config.LOCAL_RANK == 0 and loss_name in val_data:
            losses = val_data[loss_name].cpu().numpy()
            loss = np.mean(losses)

            config.logger.info(f"Mean {pre}_{loss_name}: {loss}")
            if not math.isinf(loss) and not math.isnan(loss):
                mlflow.log_metric(f"{pre}_{loss_name}", loss, config.CURRENT_STEP)

    if (
        (config.LOCAL_RANK == 0)
        and (config.calculate_metric)
        and (((current_epoch + 1) % config.CALCULATE_METRIC_EVERY_N_EPOCHS) == 0)
    ):
        val_df = val_dataloader.dataset.df

        if config.POST_PROCESSING:
            output_post_processing = config.post_processing_pipeline(config, val_data, val_df)
            val_score, oof_predictions = config.calculate_metric(config, output_post_processing, val_df, pre)
        else:
            val_score, oof_predictions = config.calculate_metric(config, val_data, val_df, pre)

        if type(val_score) is not dict:
            val_score = {"score": val_score}

        for k, v in val_score.items():
            config.logger.info(f"{pre}_{k}: {v:.3f}")

    if config.DISTRIBUTED:
        torch.distributed.barrier()

    return val_score, oof_predictions
