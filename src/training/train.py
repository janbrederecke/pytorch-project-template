import gc
import importlib
import math
import os

import mlflow
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from tqdm import tqdm

from src.evaluation.evaluation import evaluate
from src.model.get_preds import get_preds
from src.utils import (
    calculate_grad_norm,
    calculate_weight_norm,
    create_checkpoint,
    get_data,
    get_dataloader,
    get_dataset,
    get_logger,
    get_model,
    get_optimizer,
    get_scheduler,
    set_seed,
    setup_mlflow,
    sync_across_gpus,
)

# Set important ENV variables to avoid oversubscription of CPU resources and conflicts with other parallel processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(config):
    """Train a model with the given configuration.
    This function handles the complete training pipeline including data loading, model training,
    validation, and testing with support for distributed training.
    Args:
        config: A configuration object containing all training parameters including:
            - SEED: Random seed for reproducibility
            - DISTRIBUTED: Boolean flag for distributed training
            - GPU: GPU device number to use
            - FOLD: Current fold number for cross-validation
            - TEST: Boolean flag to run evaluation on test set
            - TRAIN_VAL: Boolean flag to run validation on training set
            - COMPILE_MODEL: Boolean flag to compile model for speed
            - SYNCBN: Boolean flag for synchronized batch normalization
            - MIXED_PRECISION: Boolean flag for mixed precision training
            - HIGH_METRIC_BETTER: Boolean flag indicating if higher metric is better
            - EPOCHS: Number of training epochs
            - BATCH_SIZE: Batch size for training
            - GRAD_ACCUMULATION: Number of gradient accumulation steps
            - TRACK_GRAD_NORM: Boolean flag to track gradient norms
            - CLIP_GRAD: Gradient clipping value
            - EVAL_STEPS: Number of steps between evaluations
            - EVAL_EPOCHS: Number of epochs between evaluations
            - SAVE_CHECKPOINT: Boolean flag to save checkpoints
            - SAVE_ONLY_LAST_CHECKPOINT: Boolean flag to save only final checkpoint
    Returns:
        val_score (dict): Dictionary containing validation metrics including:
            - score: Main validation metric score
            - Additional metrics as defined in the evaluation function
    Note:
        - Supports distributed training across multiple GPUs
        - Handles mixed precision training
        - Includes MLflow logging for experiments
        - Supports gradient accumulation and clipping
        - Implements checkpoint saving and loading
        - Provides flexible evaluation scheduling
    """

    # Create a logger
    config.logger = get_logger("training", getattr(config, "LOG_LEVEL", "INFO"))

    # Set seed
    if config.SEED < 0:
        config.SEED = np.random.randint(1_000_000)
    config.logger.info("Seed set to: {config.SEED}")

    if config.DISTRIBUTED:
        config.LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{config.LOCAL_RANK}"
        config.DEVICE = device

        torch.cuda.set_device(config.LOCAL_RANK)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        config.WORLD_SIZE = torch.distributed.get_world_size()
        config.RANK = torch.distributed.get_rank()
        config.GROUP = torch.distributed.new_group(np.arange(config.WORLD_SIZE))

        # Sync the seed
        config.SEED = int(
            sync_across_gpus(torch.Tensor([config.SEED]).to(device), config.WORLD_SIZE).detach().cpu().numpy()[0]
        )

        config.logger.info(f"Local rank: {config.LOCAL_RANK}, device: {config.DEVICE}, seed: {config.SEED}")

    else:
        config.LOCAL_RANK = 0
        config.WORLD_SIZE = 1
        config.RANK = 0

        device = f"cuda:{config.GPU}"
        config.DEVICE = device

    set_seed(config.SEED)

    # MLflow setup should be done after seed setting but before any training starts
    if config.LOCAL_RANK == 0:  # Only log from main process
        setup_mlflow(config=config, fold=config.FOLD)

    train_df, valid_df, test_df = get_data(config)

    train_dataset = get_dataset(train_df, config, mode="train")
    train_dataloader = get_dataloader(train_dataset, config, mode="train")

    val_dataset = get_dataset(valid_df, config, mode="valid")
    val_dataloader = get_dataloader(val_dataset, config, mode="valid")

    if config.TEST:
        test_dataset = get_dataset(test_df, config, mode="test")
        test_dataloader = get_dataloader(test_dataset, config, mode="test")

    if config.TRAIN_VAL:
        train_val_dataset = get_dataset(train_df, config, mode="valid")
        train_val_dataloader = get_dataloader(train_val_dataset, config, "valid")

    model = get_model(config=config)
    if config.COMPILE_MODEL:
        config.logger.info("Compiling model for increased training speed.")
        model = torch.compile(model)
    model.to(device)

    if config.DISTRIBUTED:
        if config.SYNCBN:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = NativeDDP(model, device_ids=[config.LOCAL_RANK], find_unused_parameters=config.FIND_UNUSED_PARAMETERS)

    total_steps = len(train_dataset)
    if train_dataloader.sampler is not None:
        if "WeightedRandomSampler" in str(train_dataloader.sampler.__class__):
            total_steps = train_dataloader.sampler.num_samples

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(config, optimizer, total_steps)

    if config.MIXED_PRECISION:
        scaler = GradScaler()
    else:
        scaler = None

    config.CURRENT_STEP = 0
    i = 0
    if config.HIGH_METRIC_BETTER:
        best_val_metric = -np.inf
    else:
        best_val_metric = np.inf

    optimizer.zero_grad()
    total_grad_norm = None
    total_weight_norm = None
    total_grad_norm_after_clip = None

    for epoch in range(config.EPOCHS):
        set_seed(config.SEED + epoch + config.LOCAL_RANK)

        config.CURRENT_EPOCH = epoch
        if config.LOCAL_RANK == 0:
            config.logger.info(f"Current epoch: {epoch}")

        if config.DISTRIBUTED:
            train_dataloader.sampler.set_epoch(epoch)

        progress_bar = tqdm(range(len(train_dataloader)), disable=config.DISABLE_TQDM, ascii=True)
        tr_it = iter(train_dataloader)

        losses = []

        gc.collect()

        if config.TRAINING:
            for itr in progress_bar:
                i += 1

                config.CURRENT_STEP += config.BATCH_SIZE * config.WORLD_SIZE

                try:
                    data = next(tr_it)
                except Exception as e:
                    config.logger.info(f"Data fetch error occured: {e}")

                model.train()
                torch.set_grad_enabled(True)

                batch = config.batch_to_device(data, device)

                if config.MIXED_PRECISION:
                    with autocast("cuda"):
                        output_dict = model(batch)
                else:
                    if config.BF16:
                        with autocast("cuda", dtype=torch.bfloat16):
                            output_dict = model(batch)
                    else:
                        output_dict = model(batch)

                loss = output_dict["loss"]

                losses.append(loss.item())

                if config.GRAD_ACCUMULATION > 1:
                    loss /= config.GRAD_ACCUMULATION

                # Backward pass
                if config.MIXED_PRECISION:
                    scaler.scale(loss).backward()

                    if i % config.GRAD_ACCUMULATION == 0:
                        if (config.TRACK_GRAD_NORM) or (config.CLIP_GRAD > 0):
                            scaler.unscale_(optimizer)
                        if config.TRACK_GRAD_NORM:
                            total_grad_norm = calculate_grad_norm(model.parameters(), config.GRAD_NORM_TYPE)
                        if config.CLIP_GRAD > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD)
                        if config.TRACK_GRAD_NORM:
                            total_grad_norm_after_clip = calculate_grad_norm(model.parameters(), config.GRAD_NORM_TYPE)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                else:
                    loss.backward()
                    if i % config.GRAD_ACCUMULATION == 0:
                        if config.TRACK_GRAD_NORM:
                            total_grad_norm = calculate_grad_norm(model.parameters())
                        if config.CLIP_GRAD > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD)
                        if config.TRACK_GRAD_NORM:
                            total_grad_norm_after_clip = calculate_grad_norm(model.parameters(), config.GRAD_NORM_TYPE)
                        if config.TRACK_WEIGHT_NORM:
                            total_weight_norm = calculate_weight_norm(model.parameters(), config.GRAD_NORM_TYPE)
                        optimizer.step()
                        optimizer.zero_grad()

                if config.DISTRIBUTED:
                    torch.cuda.synchronize()

                if scheduler is not None:
                    scheduler.step()

                if config.LOCAL_RANK == 0 and config.CURRENT_STEP % config.BATCH_SIZE == 0:
                    loss_names = [key for key in output_dict if "loss" in key]
                    for loss_name in loss_names:
                        loss_value = output_dict[loss_name].item()
                        if not math.isinf(loss_value) and not math.isnan(loss_value):
                            mlflow.log_metric(f"train_{loss_name}", loss_value, config.CURRENT_STEP)
                    mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], config.CURRENT_STEP)
                    progress_bar.set_description(f"loss: {np.mean(losses[-10:]):.4f}")

                if config.EVAL_STEPS != 0:
                    if i % config.EVAL_STEPS == 0:
                        if config.DISTRIBUTED and config.EVAL_DDP:
                            val_score = evaluate(model, val_dataloader, config, pre="val", current_epoch=epoch)
                        else:
                            if config.LOCAL_RANK == 0:
                                val_score = evaluate(model, val_dataloader, config, pre="val", current_epoch=epoch)
                    else:
                        val_score = 0

            config.logger.info(f"Average training loss: {np.mean(losses):.4f}")
            mlflow.log_metric("train_loss_epoch", np.mean(losses), config.CURRENT_STEP)

        if config.DISTRIBUTED:
            torch.cuda.synchronize()

        if config.FORCE_FP16:
            model = model.half().float()

        if config.VALID:
            if (epoch + 1) % config.EVAL_EPOCHS == 0 or (epoch + 1) == config.EPOCHS:
                if config.DISTRIBUTED and config.EVAL_DDP:
                    val_score = evaluate(model, val_dataloader, config, pre="val", current_epoch=epoch)
                else:
                    if config.LOCAL_RANK == 0:
                        val_score = evaluate(model, val_dataloader, config, pre="val", current_epoch=epoch)
            else:
                val_score = 0

                # Check if this is the best validation metric and save the model
            if config.LOCAL_RANK == 0:
                if (config.HIGH_METRIC_BETTER and val_score["score"] > best_val_metric) or (
                    not config.HIGH_METRIC_BETTER and val_score["score"] < best_val_metric
                ):
                    best_val_metric = val_score["score"]
                    config.logger.info(f"New best validation metric: {best_val_metric}")
                    mlflow.log_metric("val_metric", best_val_metric, config.CURRENT_STEP)

                    # Save the best model checkpoint
                    checkpoint = create_checkpoint(config, model, optimizer, epoch, scheduler=scheduler, scaler=scaler)
                    torch.save(
                        checkpoint,
                        f"{config.OUTPUT_DIRECTORY}/{config.CONFIG}/fold{config.FOLD}/checkpoint_best_val_score_seed{config.SEED}.pth",
                    )

        if config.TRAIN_VAL:
            if (epoch + 1) % config.CALCULATE_METRIC_EVERY_N_EPOCHS == 0 or (epoch + 1) == config.EPOCHS:
                if config.DISTRIBUTED and config.EVAL_DDP:
                    _ = get_preds(model, train_val_dataloader, config, pre=config.PRE_TRAIN_VAL)
                else:
                    if config.LOCAL_RANK == 0:
                        _ = get_preds(model, train_val_dataloader, config, pre=config.PRE_TRAIN_VAL)

        if not config.VALID and not config.TRAIN_VAL:
            val_score = 0

        if config.DISTRIBUTED:
            torch.distributed.barrier()

        if (config.LOCAL_RANK == 0) and (config.EPOCHS > 0) and (config.SAVE_CHECKPOINT):
            if not config.SAVE_ONLY_LAST_CHECKPOINT:
                checkpoint = create_checkpoint(config, model, optimizer, epoch, scheduler=scheduler, scaler=scaler)
                torch.save(
                    checkpoint,
                    f"{config.OUTPUT_DIRECTORY}/{config.CONFIG}/fold{config.FOLD}/checkpoint_last_seed{config.SEED}.pth",
                )

    if (config.LOCAL_RANK == 0) and (config.EPOCHS > 0) and (config.SAVE_CHECKPOINT):
        checkpoint = create_checkpoint(config, model, optimizer, epoch, scheduler=scheduler, scaler=scaler)
        torch.save(
            checkpoint,
            f"{config.OUTPUT_DIRECTORY}/{config.CONFIG}/fold{config.FOLD}/checkpoint_last_seed{config.SEED}.pth",
        )

    # Evaluate on the test set
    if config.TEST:
        evaluate(model, test_dataloader, test_df, config, pre="test")

    # Log artifacts and config
    mlflow.log_artifacts(
        f"{config.OUTPUT_DIRECTORY}/{config.CONFIG}/fold{config.FOLD}",
    )
    mlflow.log_artifact(f"./configs/{config.CONFIG}.py")

    # Ensure the process group is destroyed
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    return val_score
