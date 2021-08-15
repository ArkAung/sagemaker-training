## What this is

This repo demonstrates how a good SageMaker training job should be structured.
The main goals are:
* To provide visibility to the team on SageMaker experiments.
* To be able to configure parameters for SageMaker job and training script in separate config files.
* The user should be able to start SageMaker job with a single command from his/her computer.

## Preliminary Steps

In order to do so, a couple of things need to be set up:
* Conda environment having SageMaker SDK
  * This can be done with `conda env create --file=sagemaker_environment.yaml`
* AWS CLI tools with AWS credentials setup
  * [Setting up AWS CLI tool](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)

SageMaker job will be started according to AWS credentials created using AWS CLI tool.

## Config Files

### SageMaker Config

See an example SageMaker config file `configs/sagemaker_config.yaml`. All the parameters required to start a SageMaker 
job is set in this config file. This includes:
* Where the training data is on S3
* Where the configs will be uploaded on S3
* Where the outputs will be saved on S3
* ECR image that will be used for training
* ARN for SageMaker execution role
* Training specific config file

### Training Specific Config

We can pass training specific parameters as `hyperparameters` when we create [SageMaker Estimator](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#estimators).
If you happen to have a lot of parameters for your training script, setting individual `hyperparameters` and parsing the arguments in 
entry script in Docker container will become very cumbersome. Therefore, I prefer having a config file which can be read by the training script.
The config file will be provided to entry script as an input channel. We are also going to upload this training config file to S3 so that
we can provide team-wide visibility to the experiment's configurations.


## Efficient Debugging