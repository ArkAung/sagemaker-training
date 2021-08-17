"""
Script to start SageMaker training job
"""
import argparse
import os

import sagemaker
from sagemaker.estimator import Estimator

from sagemaker_default_config import load_config


def create_metrics_regex():
    metrics = [
        {"Name": "discriminator:loss", "Regex": "Loss_D: ([0-9\\.]+)", },
        {"Name": "generator:loss", "Regex": "Loss_G: ([0-9\\.]+)", },
        {"Name": "discriminator:real_images", "Regex": "D(x): ([0-9\\.]+)", },
        {"Name": "discriminator:fake_images_before_update", "Regex": "D(G(z))_before: ([0-9\\.]+)", },
        {"Name": "discriminator:fake_images_after_update", "Regex": "D(G(z))_after: ([0-9\\.]+)", }
    ]
    return metrics


def upload_configs(cfg, sagemaker_config, sm_session):
    s3_train_config_path = sm_session.upload_data(cfg.TRAINING.CONFIG_FILE,
                                                  bucket=sm_session.default_bucket(),
                                                  key_prefix=f'{cfg.S3.EXPERIMENT_NAME}/{cfg.S3.CONFIG_FOLDER}')
    print(f"Training config uploaded to {s3_train_config_path}")

    s3_sagemaker_config_path = sm_session.upload_data(sagemaker_config,
                                                      bucket=sm_session.default_bucket(),
                                                      key_prefix=f'{cfg.S3.EXPERIMENT_NAME}/{cfg.S3.CONFIG_FOLDER}')
    print(f"SageMaker job config uploaded to {s3_sagemaker_config_path}")

    return {'train_config_uri': s3_train_config_path,
            'sagemaker_config_uri': s3_sagemaker_config_path}


def start_sagemaker_job(cfg, sagemaker_config):
    results_bucket = cfg.S3.SAGEMAKER_RESULTS_BUCKET
    experiment_name = cfg.S3.EXPERIMENT_NAME

    output_uri = f"s3://{results_bucket}/{experiment_name}/{cfg.S3.OUTPUT_DIR}"

    sm_session = sagemaker.Session(default_bucket=results_bucket)
    uploaded_configs = upload_configs(cfg, sagemaker_config, sm_session)

    role = cfg.SAGEMAKER.ARN
    region = sm_session.boto_region_name
    account = sm_session.account_id()

    # Container configuration
    docker_image_name = cfg.ECR.IMAGE_NAME
    docker_tag = cfg.ECR.TAG
    training_image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/{docker_image_name}:{docker_tag}"

    # Training instance type
    training_instance = cfg.SAGEMAKER.INSTANCE_TYPE
    if training_instance.startswith("local"):
        training_session = sagemaker.LocalSession()
        training_session.config = {"local": {"local_code": True}}

        checkpoint_uri = None
        use_spot = False
        checkpoint_output_dir = None
    else:
        training_session = sm_session

        checkpoint_uri = f"s3://{results_bucket}/{experiment_name}/{cfg.S3.CHECKPOINT_DIR}"
        use_spot = cfg.SAGEMAKER.USE_SPOT_INSTANCE
        checkpoint_output_dir = cfg.CONTAINER.OUTPUT_DIR

    # Metrics to monitor during training, each metric is scraped from container Stdout
    metrics = create_metrics_regex()

    hyperparameters = {"train_cfg_filename": os.path.basename(cfg.TRAINING.CONFIG_FILE),
                       "output_dir": cfg.CONTAINER.OUTPUT_DIR
                       }

    if use_spot:
        max_wait_time = cfg.SAGEMAKER.MAX_WAIT_TIME
    else:
        max_wait_time = None

    estimator = Estimator(
        image_uri=training_image_uri,
        role=role,
        sagemaker_session=training_session,
        instance_count=cfg.SAGEMAKER.INSTANCE_COUNT,
        instance_type=training_instance,
        hyperparameters=hyperparameters,
        metric_definitions=metrics,
        output_path=output_uri,
        checkpoint_s3_uri=checkpoint_uri,
        checkpoint_local_path=checkpoint_output_dir,
        use_spot_instances=use_spot,
        max_run=cfg.SAGEMAKER.MAX_RUN_TIME,
        max_wait=max_wait_time,
        base_job_name=cfg.SAGEMAKER.JOB_NAME,
    )

    input_channels = {"traindata": cfg.S3.TRAINING_DATA,
                      "config": uploaded_configs['train_config_uri']}

    estimator.fit(input_channels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, metavar='FILE',
                        help='SageMaker job config')

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    start_sagemaker_job(cfg, sagemaker_config=args.config)


if __name__ == "__main__":
    main()
