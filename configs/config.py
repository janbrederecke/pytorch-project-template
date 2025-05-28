from types import SimpleNamespace

# Initialize the configuration
config = SimpleNamespace()

# MLflow
config.MLFLOW_EXPERIMENT = "Cool Name"

# Data Paths
config.TRAIN_DF = "./data/preprocessed/metadata.csv"
config.VALID_DF = None
config.TEST_DF = None
config.OUTPUT_DIRECTORY = "./results"

# Training Parameters
config.EPOCHS = 1
config.LR = 0.001
config.MIXED_PRECISION = True
config.BF16 = False
config.FORCE_FP16 = False
config.GRAD_ACCUMULATION = 1
config.TRACK_GRAD_NORM = True
config.CLIP_GRAD = 0
config.COMPILE_MODEL = False
config.TRACK_WEIGHT_NORM = True
config.GRAD_NORM_TYPE = 2.0
config.DISABLE_TQDM = False

# Checkpointing
config.SAVE_ONLY_LAST_CHECKPOINT = False
config.SAVE_CHECKPOINT = True
config.SAVE_VAL_DATA = False
config.SAVE_FIRST_BATCH_PREDS = False
config.SAVE_WEIGHTS_ONLY = True

# Evaluation
config.EVAL_EPOCHS = 1
config.EVAL_STEPS = 0
config.EVAL_DDP = True
config.CALCULATE_METRIC_EVERY_N_EPOCHS = 1
config.METRIC = "src.evaluation.metric"
config.HIGH_METRIC_BETTER = True
config.POST_PROCESSING = False
config.POST_PROCESSING_PIPELINE = "src.postprocessing.post_processing_pipeline"

# Stages
config.TRAINING = True
config.VALID = True
config.TEST = False
config.TRAIN_VAL = False
config.PRE_TRAIN_VAL = "train_val"

# General Settings
config.FOLDS = 5
config.BATCH_SIZE = 1024
config.BATCH_SIZE_VAL = None
config.NUM_WORKERS = 15
config.SEED = 1408
config.PERFORMANCE_MODE = True

# Model
## Model specification
config.MODEL_ROOT_FOLDER = "src.model"
config.MODEL_FILE_NAME = "resnet1d"
config.MODEL = f"{config.MODEL_ROOT_FOLDER}.{config.MODEL_FILE_NAME}"
config.N_CLASSES = 3
config.INPUT_DIM = (3, 1000)
config.BLOCKS_DIM = list(zip([32, 64, 128, 256, 384], [1000, 500, 250, 125, 25]))
config.KERNEL_SIZE = 17
config.DROPOUT_RATE = 0.0

## Model Pretraining
config.PRETRAINED_WEIGHTS = None
config.PRETRAINED_WEIGHTS_STRICT = True
config.POP_WEIGHTS = None

# GPU
## If training on single GPU, choose number
config.GPU = 0
config.AVAILABLE_GPUS = 1

## Distributed GPU stuff
config.DISTRIBUTED = True
config.SYNCBN = True
config.FIND_UNUSED_PARAMETERS = False

# Testing
config.DATA_SAMPLE = 50_000

# Augmentations
config.MIXUP = False
config.MIXUP_P = 0.5
config.MIX_BETA = 1
config.MIX_ADD = True
config.TRAIN_AUGMENTATIONS = []
config.VAL_AUGMENTATIONS = []

# Dataset
config.DATASET_ROOT_FOLDER = "src.dataset"
config.DATASET_FILE_NAME = "fog_dataset"
config.DATASET = f"{config.DATASET_ROOT_FOLDER}.{config.DATASET_FILE_NAME}"
config.RANDOM_SAMPLER_FRAC = 0

## Project specific arguments
config.WINDOW_SIZE = 1000
config.WINDOW_FUTURE = 50
config.WINDOW_PAST = config.WINDOW_SIZE - config.WINDOW_FUTURE

# Dataloader
config.PIN_MEMORY = True
config.DROP_LAST = True
config.USE_CUSTOM_BATCH_SAMPLER = False

# Scheduler
config.SCHEDULER = "ReduceLROnPlateau"
config.SCHEDULER_STEP_AFTER_VALIDATION = True if config.SCHEDULER == "ReduceLROnPlateau" else False
config.WARMUP = 0.0

## StepLR
config.EPOCHS_STEP = 0.5
config.STEPLR_GAMMA = 0.5

## Cosine
config.NUM_CYCLES = 1

# Optimizer
config.OPTIMIZER = "Adam"
config.WEIGHT_DECAY = 0.0

## SGD
config.SGD_MOMENTUM = 0.0
config.SGD_NESTEROV = True


#
config.RESUME_FROM = 0
