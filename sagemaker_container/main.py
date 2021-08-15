"""
Entry script for SageMaker training
"""

import argparse
import ast
import os

from net import Generator, Discriminator
from train import train
from config import load_config
from dataloader import creat_dataloader

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_config',
        type=str,
        help='Training config file',
        default=os.environ.get('SM_CHANNEL_TRAINCFG'))

    parser.add_argument(
        '--train_data',
        type=str,
        help='Directory where training images from S3 is downloaded',
        default=os.environ.get('SM_CHANNEL_TRAINDATA')
    )

    parser.add_argument(
        '--dataset_base_dir',
        type=str,
        default='/opt/ml/datasets',
        help='Path to store all training data')

    parser.add_argument(
        "--num_gpus",
        type=int,
        default=os.environ.get('SM_NUM_GPUS'))

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where model checkpoints, logs and tensorboard logs are writtern and annotations are cached."
             "This directory will be synced with S3.")

    parser.add_argument("--model_dir", type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument("--hosts", type=str, default=ast.literal_eval(os.environ.get('SM_HOSTS')))
    parser.add_argument("--current_host", type=str, default=os.environ.get('SM_CURRENT_HOST'))
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.train_config)
    dataloader_cfg = cfg.DATALOADER
    generator_cfg = cfg.GENERATOR
    discriminator_cfg = cfg.DISCRIMINATOR
    training_cfg = cfg.TRAINING
    dataloader = creat_dataloader(dataloader_cfg)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and training_cfg.NUM_GPU > 0) else "cpu")
    generator_network = Generator(generator_cfg, device)
    discriminator_network = Discriminator(discriminator_cfg, device)

    train(training_cfg, dataloader, generator_network, discriminator_network, device)


if __name__ == "__main__":
    main()
