# Training config template
from yacs.config import CfgNode as CN

_C = CN()

_C.DATALOADER = CN()
# Data directory for training
_C.DATALOADER.DATA_DIR = 'data/train'
# Number of workers for loading data
_C.DATALOADER.NUM_WORKERS = 2
# Batch size per iteration
_C.DATALOADER.BATCH_SIZE = 64
# Input image size
_C.DATALOADER.IMAGE_SIZE = 64


_C.GENERATOR = CN()
# Number of channels in output image
_C.GENERATOR.NUM_CHANNELS = 3
# Size of latent vector
_C.GENERATOR.LATENT_SIZE = 32
# Size of generator feature map
_C.GENERATOR.FEATURE_SIZE = 64

_C.DISCRIMINATOR = CN()
# Number of channels in input image
_C.DISCRIMINATOR.NUM_CHANNELS = 3
# Size of discriminator feature map
_C.DISCRIMINATOR.FEATURE_SIZE = 64

_C.TRAINING = CN()
# Number of epochs to train
_C.TRAINING.NUM_EPOCHS = 5
# Learning rate
_C.TRAINING.LEARNING_RATE = 1e-4
# Beta 1 of Adam optimizer
_C.TRAINING.BETA_1 = 0.5
# Number of GPUs
_C.TRAINING.NUM_GPU = 1


def get_cfg_defaults():
    return _C.clone()


def load_config(config_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg
