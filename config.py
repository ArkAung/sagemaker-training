# SageMaker job config template
from yacs.config import CfgNode as CN

_C = CN()

_C.SAGEMAKER = CN()
# ARN of SageMaker Execution Role
_C.SAGEMAKER.ARN = ""
# Type of compute instance to run the SageMaker job
# local and local_gpu to run on your computer to test before deploying to SageMaker
_C.SAGEMAKER.INSTANCE_TYPE = 'local_gpu'
# Whether to use spot instances for SageMaker training. Does not apply for
# local mode
_C.SAGEMAKER.SPOT_INSTANCE = 'False'


_C.ECR = CN()
_C.ECR.IMAGE = 'docker_image_name'
_C.ECR.TAG = 'latest'

_C.S3 = CN()
_C.S3.TRAINING_DATA = ''

_C.TRAINING = CN()
_C.TRAINING.CFG = 'configs/sagemaker_config.yaml'


def get_cfg_defaults():
    return _C.clone()


def load_config(config_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg
