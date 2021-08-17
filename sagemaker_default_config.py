# SageMaker job config template
from yacs.config import CfgNode as CN

_C = CN()

# Configs for Sagemaker related info
_C.SAGEMAKER = CN()
# ARN of SageMaker Execution Role
_C.SAGEMAKER.ARN = ""
# Type of compute instance to run the SageMaker job
# local and local_gpu to run on your computer to test before deploying to SageMaker
_C.SAGEMAKER.INSTANCE_TYPE = 'local_gpu'
# Number of instances that will be use for training
_C.SAGEMAKER.INSTANCE_COUNT = 1
# Whether to use spot instances for SageMaker training. Does not apply for
# local mode
_C.SAGEMAKER.SPOT_INSTANCE = False
# SageMaker job name
_C.SAGEMAKER.JOB_NAME = 'sagemaker-training-job'
# Max time in seconds to wait until next spot instance is available
_C.SAGEMAKER.MAX_WAIT_TIME = 86400
# Max time in seconds for training. Training will be killed if it exceeds this time limit
_C.SAGEMAKER.MAX_RUN_TIME = 86400

# Configs for Amazon ECR related info
_C.ECR = CN()
_C.ECR.IMAGE_NAME = 'docker_image_name'
_C.ECR.TAG = 'latest'

# Configs for S3 related info
_C.S3 = CN()
# S3 URI where training data resides on S3
_C.S3.TRAINING_DATA = ''
# Name of S3 bucket where SageMaker results will be stored
_C.S3.SAGEMAKER_RESULTS_BUCKET = ''
# Name of experiment which will be saved on S3. Can create as a path.
_C.S3.EXPERIMENT_NAME = 'user_1/experiment_1'
# S3 folder in which dataset_config and train_config will be uploaded
_C.S3.CONFIG_FOLDER = 'configs'
# S3 folder in which model checkpoints will be written this will be automatically synced with
# CONTAINER.OUTPUT_DIR on training instance
_C.S3.CHECKPOINT_DIR = 'model_outputs'
# S3 folder in which output sagemaker related artifacts will be written
_C.S3.OUTPUT_DIR = 'sagemaker_outputs'

# Configs for information used in SageMaker container running training scripts
_C.CONTAINER = CN()
# Base directory in Docker container where datasets will be downloaded to
_C.CONTAINER.OUTPUT_DIR = '/opt/model_outputs'

# Configs for training related info
_C.TRAINING = CN()
_C.TRAINING.CONFIG_FILE = 'configs/train.yaml'


def get_cfg_defaults():
    return _C.clone()


def load_config(config_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg
