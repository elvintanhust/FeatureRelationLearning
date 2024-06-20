import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Dataset name
_C.DATA.NAME = 'imagenet'
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'vgg'
# Model name
_C.MODEL.NAME = 'vgg'
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate  base=0.1
_C.MODEL.DROP_PATH_RATE = 0.0
# Label Smoothing  base=0.1
_C.MODEL.LABEL_SMOOTHING = 0.0

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 1
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.3
_C.TRAIN.BASE_LR = 0.003
_C.TRAIN.WARMUP_LR = 5e-6
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 20.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = False
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
_C.TRAIN.LR_SCHEDULER.MILESTONES = [60, 120, 160]

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor  base=0.4
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"'rand-m9-mstd0.5-inc1'
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob  base=0.25
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count  base=1
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0  base=0.8
_C.AUG.MIXUP = 0
# Cutmix alpha, cutmix enabled if > 0  base=1.0
_C.AUG.CUTMIX = 0.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled  base=1.0
_C.AUG.MIXUP_PROB = 0.0
# Probability of switching to cutmix when both mixup and cutmix enabled  base=0.5
_C.AUG.MIXUP_SWITCH_PROB = 0.0
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'
# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 50
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 666
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
_C.RELEASE_MODE = False
_C.ROOT_PATH = "../model"


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.learning_rate:
        config.TRAIN.BASE_LR = args.learning_rate
    if args.epochs:
        config.TRAIN.EPOCHS = args.epochs
    if args.start_epoch:
        config.TRAIN.START_EPOCH = args.start_epoch
    if args.optimizer:
        config.TRAIN.OPTIMIZER.NAME = args.optimizer
    if args.auto_resume:
        config.TRAIN.AUTO_RESUME = args.auto_resume
    if args.eval:
        config.EVAL_MODE = True
    if args.release_mode:
        config.RELEASE_MODE = True

    if "cifar" in config.DATA.NAME:
        config.DATA.IMG_SIZE = 32
    elif config.DATA.NAME == 'miniImagenet':
        config.DATA.IMG_SIZE = 84
    elif config.DATA.NAME == 'tinyImagenet':
        config.DATA.IMG_SIZE = 64
    elif config.DATA.NAME == 'office':
        config.DATA.IMG_SIZE = 96
        config.DATA.DOMAIN = "Real World"  ## "Art", "Clipart", "Product", "Real World"

    config.TAG = "0.5"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config.TRAIN.EPOCHS = 200
    config.TRAIN.BASE_LR = 0.1
    config.TRAIN.WARMUP_EPOCHS = 0
    config.DATA.BATCH_SIZE = 256
    config.DATA.TEST_BATCH_SIZE = 400
    config.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
    config.TRAIN.LR_SCHEDULER.MILESTONES = [120, 150, 180]

    config.TRAIN.LR_SCHEDULER.NAME = 'multi_step'
    config.TRAIN.OPTIMIZER.NAME = 'SDG'
    from scripts.utils import setup_seed
    setup_seed(config.SEED)
    # output folder
    config.OUTPUT = os.path.join(config.ROOT_PATH, config.MODEL.NAME, config.DATA.NAME, config.TAG)
    if not os.path.exists(config.OUTPUT):
        os.makedirs(config.OUTPUT)
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
