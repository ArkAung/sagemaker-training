"""
Entry script for SageMaker training
"""

import argparse
import os

import torch

from dataloader import creat_dataloader
from net import Generator, Discriminator
from train import train
from training_default_config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_cfg_filename',
        type=str,
        required=True,
        help='Training config filename')

    parser.add_argument(
        '--config_dir',
        type=str,
        help='Directory for config files',
        default=os.environ.get('SM_CHANNEL_CONFIG'))

    parser.add_argument(
        '--train_data',
        type=str,
        help='Directory for training images',
        default=os.environ.get('SM_CHANNEL_TRAINDATA')
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where model checkpoints, logs and other artefacts are saved."
             "This directory will be synced with s3://SAGEMAKER_RESULTS_BUCKET/EXPERIMENT_NAME/CHECKPOINT_DIR on S3.")
    return parser.parse_args()


def populate_data_dir(dataloader_cfg, data_dir):
    dataloader_cfg.defrost()
    dataloader_cfg.DATA_DIR = data_dir
    dataloader_cfg.freeze()
    return dataloader_cfg


def main():
    args = parse_args()
    cfg = load_config(os.path.join(args.config_dir, args.train_cfg_filename))
    dataloader_cfg = cfg.DATALOADER
    dataloader_cfg = populate_data_dir(dataloader_cfg, args.train_data)
    generator_cfg = cfg.GENERATOR
    discriminator_cfg = cfg.DISCRIMINATOR
    training_cfg = cfg.TRAINING

    dataloader = creat_dataloader(dataloader_cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator_network = Generator(generator_cfg, device)
    discriminator_network = Discriminator(discriminator_cfg, device)

    train(training_cfg, generator_cfg, dataloader, generator_network, discriminator_network, device)


if __name__ == "__main__":
    main()
