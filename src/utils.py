import importlib
import logging
import math
import os
import random

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    Sampler,
    SequentialSampler,
    WeightedRandomSampler,
)
from torch.utils.data.distributed import DistributedSampler


# Set the random seed for reproducibility
def set_seed(seed=1408, performance_mode=True):
    """Sets random seeds for reproducibility across multiple libraries.

    This function sets random seeds for Python's random module, NumPy, PyTorch CPU and CUDA operations.
    It also configures CUDNN backend settings based on whether performance or reproducibility is prioritized.

    Args:
        seed (int, optional): Random seed value. Defaults to 1408.
        performance_mode (bool, optional): If True, optimizes for performance by allowing non-deterministic
            algorithms. If False, enforces deterministic algorithms for better reproducibility.
            Defaults to True.

    Note:
        Setting performance_mode=False may result in slower execution but ensures better reproducibility
        across different runs.

    Example:
        >>> set_seed(42, performance_mode=False)  # For reproducible results
        >>> set_seed(42, performance_mode=True)   # For better performance
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Choose improved performance or improved reproducibility
    if performance_mode:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Initialize individual worker
def init_worker(worker_id):
    """Initialize a worker with a random seed.

    This function sets a unique random seed for each worker in a parallel processing context
    by adding the worker ID to the current random state.

    Args:
        worker_id (int): The ID of the worker process.

    Notes:
        This function is typically used in multiprocessing contexts to ensure
        different workers generate different random sequences.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# Load model
def get_model(config):
    """
    Initializes and loads a neural network model based on the provided configuration.
    This function creates a neural network instance from the specified model class and optionally
    loads pretrained weights. It handles both single weight files and fold-specific weight files,
    and supports selective weight loading by allowing certain weights to be excluded.
    Args:
        config: Configuration object containing model specifications including:
            - MODEL: Path to model module containing Net class
            - PRETRAINED_WEIGHTS: Path to pretrained weights file or list of files for folds
            - FOLD: Current fold number when using fold-specific weights
            - LOCAL_RANK: Process rank for distributed training
            - POP_WEIGHTS: List of weight keys to exclude from loading
            - PRETRAINED_WEIGHTS_STRICT: Boolean for strict weight loading
            - logger: Logger object for tracking weight loading process
    Returns:
        net: Initialized neural network model with optionally loaded weights
    Example:
        config = ModelConfig()
        config.MODEL = "models.resnet"
        config.PRETRAINED_WEIGHTS = "weights.pth"
        model = get_model(config)
    """

    Net = importlib.import_module(config.MODEL).Net
    net = Net(config)
    if config.PRETRAINED_WEIGHTS is not None:
        if type(config.PRETRAINED_WEIGHTS) is list:
            config.PRETRAINED_WEIGHTS = config.PRETRAINED_WEIGHTS[config.FOLD]
        config.logger.info(f"{config.LOCAL_RANK}: Loading weights from {config.PRETRAINED_WEIGHTS}")
        state_dict = torch.load(config.PRETRAINED_WEIGHTS, map_location="cpu")
        if "model" in state_dict.keys():
            state_dict = state_dict["model"]
        state_dict = {key.replace("module.", ""): val for key, val in state_dict.items()}
        if config.POP_WEIGHTS is not None:
            config.logger.info(f"Popping {config.POP_WEIGHTS}")
            to_pop = []
            for key in state_dict:
                for item in config.POP_WEIGHTS:
                    if item in key:
                        to_pop += [key]
            for key in to_pop:
                config.logger.info(f"Popping {key}")
                state_dict.pop(key)

        net.load_state_dict(state_dict, strict=config.PRETRAINED_WEIGHTS_STRICT)
        config.logger.info(f"{config.LOCAL_RANK}: Weights loaded from {config.PRETRAINED_WEIGHTS}")

    return net


# Load the specified dataset
def get_dataset(df, config, mode="train"):
    """
    Create and return a dataset based on the specified mode.

    Args:
        df (pandas.DataFrame): Input dataframe containing the dataset.
        config (Config): Configuration object containing dataset parameters.
        mode (str, optional): Dataset mode. Can be 'train', 'valid', or 'test'. Defaults to "train".

    Returns:
        Dataset: The constructed dataset object for the specified mode.

    Raises:
        ValueError: If an invalid mode is provided.
    """
    config.logger.info(f"Loading {mode} dataset")

    if mode == "train":
        dataset = get_train_dataset(df, config)
    elif mode == "valid":
        dataset = get_val_dataset(df, config)
    elif mode == "test":
        dataset = get_test_dataset(df, config)
    else:
        pass
    return dataset


def get_train_dataset(df, config):
    """Creates and returns a training dataset from the given dataframe.

    Args:
        df (pandas.DataFrame): Input dataframe containing training data
        config (Config): Configuration object containing dataset parameters

    Returns:
        torch.utils.data.Dataset: Training dataset object, optionally subsampled

    Notes:
        Uses CustomDataset class and augmentations defined in config.
        Returns subset of size config.DATA_SAMPLE if > 0.
    """
    train_dataset = config.CustomDataset(df, config, augmentations=config.TRAIN_AUGMENTATIONS, mode="train")
    if config.DATA_SAMPLE > 0:
        train_dataset = torch.utils.data.Subset(train_dataset, np.arange(config.DATA_SAMPLE))
    return train_dataset


def get_val_dataset(df, config):
    """Creates and returns a validation dataset from the given dataframe.

    Args:
        df (pandas.DataFrame): Input dataframe containing validation data
        config (Config): Configuration object containing dataset parameters

    Returns:
        torch.utils.data.Dataset: Validation dataset object
    """
    val_dataset = config.CustomDataset(df, config, augmentations=config.VAL_AUGMENTATIONS, mode="valid")
    return val_dataset


def get_test_dataset(df, config):
    """Creates and returns a test dataset from the given dataframe.

    Args:
        df (pandas.DataFrame): Input dataframe containing test data
        config (Config): Configuration object containing dataset parameters

    Returns:
        torch.utils.data.Dataset: Test dataset object
    """
    test_dataset = config.CustomDataset(df, config, aug=config.valid_aug, mode="test")
    return test_dataset


def get_dataloader(ds, config, mode="train"):
    """
    Creates and returns a DataLoader object based on the specified mode.

    Args:
        ds (Dataset): The dataset to create a DataLoader for
        config (dict): Configuration dictionary containing DataLoader parameters
        mode (str, optional): Mode for DataLoader creation. Must be one of:
            - "train": Creates training DataLoader
            - "valid": Creates validation DataLoader
            - "test": Creates test DataLoader
            Defaults to "train".

    Returns:
        DataLoader: PyTorch DataLoader object configured for the specified mode

    Raises:
        ValueError: If mode is not one of "train", "valid" or "test"
    """
    if mode == "train":
        dataloader = get_train_dataloader(ds, config)
    elif mode == "valid":
        dataloader = get_val_dataloader(ds, config)
    elif mode == "test":
        dataloader = get_test_dataloader(ds, config)
    return dataloader


def get_train_dataloader(ds, config):
    """Creates and returns a DataLoader for training data.

    Args:
        ds (Dataset): The dataset to create a DataLoader for
        config (Config): Configuration object containing training parameters
            Required attributes:
            - DISTRIBUTED (bool): Whether to use distributed training
            - WORLD_SIZE (int): Number of processes for distributed training
            - LOCAL_RANK (int): Local process rank for distributed training
            - SEED (int): Random seed
            - RANDOM_SAMPLER_FRAC (float): Fraction of dataset to sample randomly
            - USE_CUSTOM_BATCH_SAMPLER (bool): Whether to use custom batch sampler
            - BATCH_SIZE (int): Size of training batches
            - DROP_LAST (bool): Whether to drop last incomplete batch
            - NUM_WORKERS (int): Number of worker processes for data loading
            - PIN_MEMORY (bool): Whether to pin memory for GPU transfer
            - collate_train (callable): Function to collate training batches
            - logger: Logger object for logging information

    Returns:
        DataLoader: PyTorch DataLoader configured for training

    Notes:
        Supports different sampling strategies:
        - Distributed sampling for distributed training
        - Weighted random sampling based on sample weights
        - Custom batch sampling
        - Regular random sampling
    """
    if config.DISTRIBUTED:
        sampler = DistributedSampler(
            ds,
            num_replicas=config.WORLD_SIZE,
            rank=config.LOCAL_RANK,
            shuffle=True,
            seed=config.SEED,
        )
    else:
        try:
            if config.RANDOM_SAMPLER_FRAC > 0:
                num_samples = int(len(ds) * config.RANDOM_SAMPLER_FRAC)
                sample_weights = ds.sample_weights
                sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples)
            else:
                sampler = None
        except (TypeError, ValueError, AttributeError) as e:
            config.logger.info(f"An error occrured: {e}")
            sampler = None

    if config.USE_CUSTOM_BATCH_SAMPLER:
        sampler = RandomSampler(ds)
        bsampler = CustomBatchSampler(sampler, batch_size=config.BATCH_SIZE, drop_last=config.DROP_LAST)
        train_dataloader = DataLoader(
            ds,
            batch_sampler=bsampler,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            collate_fn=config.collate_train,
            worker_init_fn=init_worker,
        )
    else:
        train_dataloader = DataLoader(
            ds,
            sampler=sampler,
            shuffle=(sampler is None),
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            collate_fn=config.collate_train,
            drop_last=config.DROP_LAST,
            worker_init_fn=init_worker,
        )
    config.logger.info(f"Train: dataset {len(ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader


# Ordered distributed sampler for validation and testing consistency
class OrderedDistributedSampler(Sampler):
    """
    A sampler that ensures data is distributed across multiple processes in a distributed training setup
    while maintaining the original order of samples.

    This sampler is particularly useful for distributed training scenarios where you want to:
    1. Ensure each process gets a unique subset of the data
    2. Maintain the original ordering of samples within each subset
    3. Handle cases where the dataset size is not perfectly divisible by the number of processes

    Parameters
    ----------
    config : object
        Configuration object that contains a logger attribute for logging information
    dataset : Dataset
        The dataset to sample from
    num_replicas : int, optional
        Number of processes participating in distributed training.
        If None, obtained from the distributed package
    rank : int, optional
        Rank of the current process in distributed training.
        If None, obtained from the distributed package

    Attributes
    ----------
    num_samples : int
        The number of samples each process will see per epoch
    total_size : int
        The total number of samples across all processes

    Methods
    -------
    __iter__()
        Returns an iterator over the indices assigned to this process
    __len__()
        Returns the number of samples for this process

    Raises
    ------
    RuntimeError
        If distributed package is not available when num_replicas or rank is None
    """

    def __init__(self, config, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.config = config
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.config.logger.info(f"Total size: {self.total_size}")

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank * self.num_samples : self.rank * self.num_samples + self.num_samples]
        self.config.logger.info(
            f"Samples: {self.rank * self.num_samples},{self.rank * self.num_samples + self.num_samples}"
        )
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def get_val_dataloader(val_ds, config):
    """
    Creates and returns a validation DataLoader for the given dataset.

    Args:
        val_ds: The validation dataset to create DataLoader for
        config: Configuration object containing DataLoader parameters including:
            - DISTRIBUTED: Boolean indicating if distributed training is enabled
            - EVAL_DDP: Boolean indicating if DDP evaluation is enabled
            - WORLD_SIZE: Number of distributed processes
            - LOCAL_RANK: Rank of current process
            - BATCH_SIZE_VAL: Batch size for validation (if None, uses BATCH_SIZE)
            - BATCH_SIZE: Default batch size
            - NUM_WORKERS: Number of worker processes for data loading
            - PIN_MEMORY: Boolean indicating if tensors should be pinned in memory
            - collate_valid: Collate function for validation batches
            - logger: Logger object for printing info

    Returns:
        DataLoader: DataLoader configured for validation with appropriate sampler and batch size
    """
    if config.DISTRIBUTED and config.EVAL_DDP:
        sampler = OrderedDistributedSampler(config, val_ds, num_replicas=config.WORLD_SIZE, rank=config.LOCAL_RANK)
    else:
        sampler = SequentialSampler(val_ds)

    if config.BATCH_SIZE_VAL is not None:
        batch_size = config.BATCH_SIZE_VAL
    else:
        batch_size = config.BATCH_SIZE
    val_dataloader = DataLoader(
        val_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=config.collate_valid,
        worker_init_fn=init_worker,
    )
    config.logger.info(f"Valid: dataset {len(val_ds)}, dataloader {len(val_dataloader)}")
    return val_dataloader


def get_test_dataloader(test_ds, config):
    """Creates and returns a test dataloader with specified configuration.

    This function sets up a DataLoader for test dataset with appropriate sampling strategy
    based on whether distributed evaluation is enabled.

    Args:
        test_ds: The test dataset to create loader from
        config: Configuration object containing following attributes:
            - DISTRIBUTED: bool, whether using distributed training
            - EVAL_DDP: bool, whether to use distributed evaluation
            - WORLD_SIZE: int, total number of processes in distributed setting
            - LOCAL_RANK: int, rank of current process
            - BATCH_SIZE_VAL: int or None, batch size for validation/testing
            - BATCH_SIZE: int, default batch size if BATCH_SIZE_VAL is None
            - NUM_WORKERS: int, number of worker processes for data loading
            - PIN_MEMORY: bool, whether to pin memory in data loading
            - collate_valid: callable, function to collate samples into batches
            - logger: logging object for output messages

    Returns:
        DataLoader: Configured PyTorch DataLoader for test dataset

    Example:
        >>> test_loader = get_test_dataloader(test_dataset, config)
    """
    if config.DISTRIBUTED and config.EVAL_DDP:
        sampler = OrderedDistributedSampler(config, test_ds, num_replicas=config.WORLD_SIZE, rank=config.LOCAL_RANK)
    else:
        sampler = SequentialSampler(test_ds)

    if config.BATCH_SIZE_VAL is not None:
        batch_size = config.BATCH_SIZE_VAL
    else:
        batch_size = config.BATCH_SIZE
    test_dataloader = DataLoader(
        test_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=config.collate_valid,
        worker_init_fn=init_worker,
    )
    config.logger.info(f"Test: dataset {len(test_ds)}, dataloader {len(test_dataloader)}")
    return test_dataloader


def sync_across_gpus(tensor, world_size):
    """
    Synchronizes tensor across multiple GPUs in a distributed setting.

    This function ensures tensor synchronization across all GPUs by creating a barrier,
    gathering tensors from all processes, and concatenating them.

    Args:
        tensor (torch.Tensor): The tensor to be synchronized across GPUs.
        world_size (int): Total number of processes in the distributed environment.

    Returns:
        torch.Tensor: Concatenated tensor containing gathered data from all GPUs.

    Note:
        - Requires torch.distributed to be initialized
        - All processes must enter this function in a lockstep manner
    """
    torch.distributed.barrier()
    gather_tensor = [torch.ones_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gather_tensor, tensor)
    return torch.cat(gather_tensor)


def read_df(filepath):
    """
    Read a dataframe from a file.

    This function reads data from either a parquet or csv file and returns it as a pandas DataFrame.

    Args:
        filepath (str): Path to the file to be read. The file can be either parquet or csv format.

    Returns:
        pandas.DataFrame: The dataframe containing the data from the file.

    Examples:
        >>> df = read_df("data.parquet")
        >>> df = read_df("data.csv")
    """
    if "parquet" in filepath:
        df = pd.read_parquet(filepath, engine="fastparquet")
    else:
        df = pd.read_csv(filepath)
    return df


def get_data(config):
    """
    Loads and preprocesses training, validation, and test datasets based on configuration.

    This function handles the loading of datasets and their splitting based on fold configuration.
    It supports both predefined splits and fold-based splitting of data.

    Parameters
    ----------
    config : object
        Configuration object containing:
        - TRAIN_DF : str or list
            Path or list of paths to training data file(s)
        - TEST_DF : str
            Path to test data file
        - VALID_DF : str or list, optional
            Path or list of paths to validation data file(s)
        - FOLD : int
            Fold number for cross-validation (-1 for using first fold as validation)
        - TEST : bool
            Flag indicating whether to load test data
        - logger : object
            Logger object for logging information

    Returns
    -------
    tuple
        Contains three DataFrames:
        - train_df : DataFrame
            Training data
        - valid_df : DataFrame
            Validation data
        - test_df : DataFrame or None
            Test data if TEST is True, None otherwise
    """
    if type(config.TRAIN_DF) is list:
        config.TRAIN_DF = config.TRAIN_DF[config.FOLD]
    config.logger.info(f"Reading {config.TRAIN_DF}")
    df = read_df(config.TRAIN_DF)

    if config.TEST:
        test_df = read_df(config.TEST_DF)
    else:
        test_df = None

    if config.VALID_DF:
        if type(config.VALID_DF) is list:
            config.VALID_DF = config.VALID_DF[config.FOLD]
        valid_df = read_df(config.VALID_DF)
        if config.FOLD > -1:
            if "fold" in valid_df.columns:
                valid_df = valid_df[valid_df["fold"] == config.FOLD]
                train_df = df[df["fold"] != config.FOLD]
            else:
                train_df = df
        else:
            train_df = df
    else:
        if config.FOLD == -1:
            valid_df = df[df["fold"] == 0]
        else:
            valid_df = df[df["fold"] == config.FOLD]

        train_df = df[df["fold"] != config.FOLD]

    return train_df, valid_df, test_df


def calculate_grad_norm(parameters, norm_type=2.0):
    """
    Calculate the gradient norm of the given parameters.

    This function computes the norm of the gradients for a set of parameters. It can handle
    both single tensors and collections of parameters, making it useful for monitoring
    gradient behavior during training.

    Args:
        parameters (torch.Tensor or Iterable[torch.Tensor]): The parameters whose gradients
            will be used for norm calculation. Can be a single tensor or an iterable of tensors.
        norm_type (float, optional): Type of the norm to be calculated. Default: 2.0

    Returns:
        torch.Tensor or None: The calculated norm of the gradients. Returns None if the
        resulting norm is NaN or infinite. Returns tensor(0.0) if no parameters with
        gradients are found.

    Example:
        >>> model = torch.nn.Linear(10, 5)
        >>> loss = criterion(model(x), y)
        >>> loss.backward()
        >>> grad_norm = calculate_grad_norm(model.parameters())
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type
    )
    if torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        total_norm = None

    return total_norm


def calculate_weight_norm(parameters, norm_type=2.0):
    """
    Calculate the average norm of model parameters.

    This function computes the average L-p norm across all provided parameters.
    If any parameter's norm is NaN or Inf, returns None.

    Args:
        parameters (Union[torch.Tensor, Iterable[torch.Tensor]]): Single tensor or iterable of tensors
            containing model parameters to calculate norm for. Only parameters with non-None gradients are included.
        norm_type (float, optional): Type of vector norm to calculate (e.g. L1=1.0, L2=2.0). Defaults to 2.0.

    Returns:
        torch.Tensor or None: Mean L-p norm of parameters if valid, None if any norm is NaN or Inf.
            Returns tensor(0.0) if no valid parameters found.

    Example:
        >>> model = torch.nn.Linear(10, 10)
        >>> norm = calculate_weight_norm(model.parameters())
        >>> print(norm)
        tensor(0.7071)
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device

    total_norm = torch.stack([torch.norm(p.detach(), norm_type).to(device) for p in parameters]).mean()
    if torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        total_norm = None

    return total_norm


def create_checkpoint(config, model, optimizer, epoch, scheduler=None, scaler=None):
    """Creates a checkpoint dictionary containing model and training states.
    This function saves the current state of training including model weights,
    optimizer state, epoch number, and optionally scheduler and scaler states.
    Args:
        config: Configuration object containing training settings
        model: PyTorch model to save state from
        optimizer: Optimizer instance to save state from
        epoch: Current epoch number
        scheduler: Optional learning rate scheduler to save state from
        scaler: Optional gradient scaler for mixed precision training
    Returns:
        dict: Checkpoint dictionary containing saved states. If config.SAVE_WEIGHTS_ONLY
            is True, only returns model weights dictionary.
    Example:
        >>> checkpoint = create_checkpoint(config, model, optimizer, epoch,
        ...                              scheduler, scaler)
        >>> torch.save(checkpoint, 'checkpoint.pth')
    """

    state_dict = model.state_dict()
    if config.SAVE_WEIGHTS_ONLY:
        checkpoint = {"model": state_dict}
        return checkpoint

    checkpoint = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint


def load_checkpoint(config, model, optimizer, scheduler=None, scaler=None):
    """Load a training checkpoint to resume training.
    This function loads a previously saved checkpoint to restore the state of training,
    including model weights, optimizer state, learning rate scheduler state, gradient
    scaler state (if using mixed precision), and the last completed epoch.
    Args:
        config: Configuration object containing logging and checkpoint path info
        model: PyTorch model to load weights into
        optimizer: Optimizer to restore state for
        scheduler: Optional learning rate scheduler to restore state for
        scaler: Optional gradient scaler for mixed precision training
    Returns:
        tuple: Updated (model, optimizer, scheduler_dict, scaler, epoch) with restored states
    Raises:
        FileNotFoundError: If checkpoint file specified in config.RESUME_FROM does not exist
    """

    config.logger.info(f"Loading checkpoint {config.RESUME_FROM}")
    checkpoint = torch.load(config.RESUME_FROM, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler_dict = checkpoint["scheduler"]
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    epoch = checkpoint["epoch"]
    return model, optimizer, scheduler_dict, scaler, epoch


def get_optimizer(model, config):
    """Get optimizer for model based on configuration.

    This function initializes and returns an optimizer based on the specified configuration.
    Supported optimizers include Adam, AdamW, AdamW+ (with custom parameter groups), and SGD.

    Args:
        model: PyTorch model to optimize
        config: Configuration object containing optimizer settings with following attributes:
            - OPTIMIZER: String specifying optimizer type ('Adam', 'AdamW', 'AdamW_plus', 'SGD')
            - LR: Learning rate
            - WEIGHT_DECAY: Weight decay factor
            - SGD_MOMENTUM: Momentum factor for SGD (only used if OPTIMIZER='SGD')
            - SGD_NESTEROV: Boolean flag for Nesterov momentum in SGD

    Returns:
        torch.optim.Optimizer: Configured optimizer instance

    Examples:
        >>> optimizer = get_optimizer(model, config)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """
    params = model.parameters()

    if config.OPTIMIZER == "Adam":
        optimizer = optim.Adam(params, lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    elif config.OPTIMIZER == "AdamW_plus":
        paras = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        params = [
            {
                "params": [param for name, param in paras if (not any(nd in name for nd in no_decay))],
                "lr": config.LR,
                "weight_decay": config.WEIGHT_DECAY,
            },
            {
                "params": [param for name, param in paras if (any(nd in name for nd in no_decay))],
                "lr": config.LR,
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(params, lr=config.LR)

    elif config.OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(params, lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    elif config.OPTIMIZER == "SGD":
        optimizer = optim.SGD(
            params,
            lr=config.LR,
            momentum=config.SGD_MOMENTUM,
            nesterov=config.SGD_NESTEROV,
            weight_decay=config.WEIGHT_DECAY,
        )

    return optimizer


def get_scheduler(config, optimizer, total_steps):
    """
    Creates and returns a learning rate scheduler based on configuration settings.
    This function initializes different types of learning rate schedulers depending on the
    configuration specified. Supported schedulers include StepLR, Cosine with warmup,
    Linear with warmup, and CosineAnnealingLR.
    Parameters
    ----------
    config : object
        Configuration object containing scheduler settings including:
        - SCHEDULER: str, type of scheduler to use
        - EPOCHS: int, number of training epochs
        - EPOCHS_STEP: int, step size for StepLR
        - BATCH_SIZE: int, training batch size
        - WORLD_SIZE: int, number of distributed processes
        - STEPLR_GAMMA: float, multiplicative factor for StepLR
        - WARMUP: int, number of warmup steps for schedulers with warmup
        - NUM_CYCLES: int, number of cycles for cosine scheduler
    optimizer : torch.optim.Optimizer
        The optimizer whose learning rate will be scheduled
    total_steps : int
        Total number of training steps
    Returns
    -------
    scheduler : torch.optim.lr_scheduler._LRScheduler or None
        The initialized learning rate scheduler, or None if no scheduler is specified
    """

    if config.SCHEDULER == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.EPOCHS_STEP * (total_steps // config.BATCH_SIZE) // config.WORLD_SIZE,
            gamma=config.STEPLR_GAMMA,
        )
    elif config.SCHEDULER == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.WARMUP * (total_steps // config.BATCH_SIZE) // config.WORLD_SIZE,
            num_training_steps=config.EPOCHS * (total_steps // config.BATCH_SIZE) // config.WORLD_SIZE,
            num_cycles=config.NUM_CYCLES,
        )
    elif config.SCHEDULER == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=config.EPOCHS * (total_steps // config.BATCH_SIZE) // config.WORLD_SIZE,
        )
    elif config.SCHEDULER == "CosineAnnealingLR":
        T_max = int(np.ceil(0.5 * total_steps))
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-8)
    elif config.SCHEDULER == "ReduceLROnPlateau":
        mode = "max" if config.HIGH_METRIC_BETTER else "min"
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=config.FACTOR, patience=config.PATIENCE, min_lr=0.00001
        )
    else:
        scheduler = None

    return scheduler


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    from https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    from https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_level(level_str):
    """Converts a string representation of a logging level to its corresponding numeric value.
    Args:
        level_str (str): String representation of the logging level (e.g., 'info', 'debug', 'warning', 'error', 'critical')
    Returns:
        int: The numeric logging level value. If the input string doesn't match any known level, returns logging.INFO (20)
    Examples:
        >>> get_level('debug')
        10
        >>> get_level('INFO')
        20
        >>> get_level('unknown')
        20
    """
    l_names = {logging.getLevelName(lvl).lower(): lvl for lvl in [10, 20, 30, 40, 50]}  # noqa
    return l_names.get(level_str.lower(), logging.INFO)


def get_logger(name, level_str):
    """Get a configured logger instance.

    Args:
        name (str): The name of the logger.
        level_str (str): The logging level as a string (e.g. 'INFO', 'DEBUG', 'ERROR').

    Returns:
        logging.Logger: A configured logger instance with stream handler and formatter.

    Example:
        >>> logger = get_logger('my_app', 'INFO')
        >>> logger.info('This is a log message')
        2023-01-01 12:00:00,000 - my_app - INFO - This is a log message
    """
    logger = logging.getLogger(name)
    logger.setLevel(get_level(level_str))
    handler = logging.StreamHandler()
    handler.setLevel(level_str)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))  # pylint: disable=C0301 # noqa
    logger.addHandler(handler)

    return logger


def setup_mlflow(config, fold=None):
    """
    Sets up MLflow experiment tracking and logging.

    This function initializes MLflow tracking with specified configuration, creates or reuses
    a parent run, and starts a new child run. It also logs all configuration parameters.

    Args:
        config: Configuration object containing:
            - MLFLOW_TRACKING_URI (str): MLflow tracking server URI (default: "http://mlflow:5000")
            - MLFLOW_EXPERIMENT (str): Name of the MLflow experiment
            - CONFIG (str): Name for the parent run
            - FOLD (int/str): Name/number for the child run
            - logger: Logger object for logging setup progress
            - Additional parameters to be logged to MLflow
        fold (int, optional): Fold number to be logged as a parameter. Defaults to None.

    Returns:
        mlflow.ActiveRun: Active MLflow run object representing the child run.

    Raises:
        ValueError: If MLFLOW_EXPERIMENT is not specified in config
        Exception: If MLflow setup fails for any reason

    Example:
        >>> config = Config(MLFLOW_EXPERIMENT="my_experiment", CONFIG="run1")
        >>> run = setup_mlflow(config, fold=0)
        >>> # Use run for tracking metrics, artifacts etc.
        >>> run.end()
    """
    try:
        tracking_uri = getattr(config, "MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        config.logger.info(f"MLflow tracking URI set to {tracking_uri}")

        experiment_name = getattr(config, "MLFLOW_EXPERIMENT", None)
        if experiment_name:
            mlflow.set_experiment(experiment_name)
            config.logger.info(f"MLflow experiment set to {experiment_name}")
            # Check if parent run exists
            existing_runs = mlflow.search_runs(
                experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id],
                filter_string=f"tags.mlflow.runName = '{config.CONFIG}' and attributes.status = 'RUNNING'",
            )

            if len(existing_runs) > 0:
                # Use existing parent run
                parent_run_id = existing_runs.iloc[0].run_id
                parent_run = mlflow.get_run(parent_run_id)
            else:
                # Create new parent run
                parent_run = mlflow.start_run(
                    run_name=config.CONFIG, experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id
                )
        else:
            raise ValueError(f"Experiment name must be specified in config.MLFLOW_EXPERIMENT")

        run = mlflow.start_run(run_name=str(config.FOLD), nested=True, parent_run_id=parent_run.info.run_id)

        config.logger.info("MLflow run started.")

        def serialize_params(params_dict):
            serialized = {}
            for key, value in params_dict.items():
                try:
                    mlflow.log_param(key, value)
                    serialized[key] = value
                except Exception:
                    serialized[key] = str(value)
            return serialized

        params = serialize_params(config.__dict__)
        mlflow.log_params(params)
        config.logger.info("Logged configuration parameters.")

        if fold is not None:
            mlflow.log_param("fold", fold)
            config.logger.info(f"Logged fold parameter: {fold}")
        return run
    except Exception as e:
        config.logger.error(f"Error setting up MLflow: {e}")
        raise
