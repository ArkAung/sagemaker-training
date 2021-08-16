## SageMaker Training Structure and Flow [WIP]

This repo demonstrates how a good SageMaker training structure and flow can look like.
The main goals are:
* I want to start a SageMaker training job from my computer using a single command.
* I want to configure parameters for the SageMaker job and for training using config files rather
than modifying parameters in scripts.
* For every experiment I run, I want to provide visibility to my ML team. I want my team to quickly
know what parameters I have used to start my training job on SageMaker and how they can replicate it.

To meet these goals, I will:
* Have a script called `run_sagemaker_job.py` which I can run from my computer with a config file as an input.
* Have two YAML files: one for SageMaker related configs and one for training-related configs.
* Upload the configs that I have used to start my SageMaker job as well as the config file which contains training
parameters to S3. 
* Upload results (trained model, model outputs, model logs) to S3.

## Preliminary Steps

A couple of things need to be set up.

On my computer:
* Conda environment having SageMaker SDK
  * This can be done with `conda env create --file=sagemaker_environment.yaml`
* AWS CLI tools with AWS credentials setup
  * [Setting up AWS CLI tool](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)

SageMaker job will be started according to AWS credentials created using AWS CLI tool.

On AWS:
* Create a SageMaker execution Role and get the ARN.
* Upload training data on S3.
* Create a repository on ECR.

## Directories and Files
* `configs`: Directory containing config files for SageMaker and training script
* `sagemaker_container`: Directory containing all the files required for training. 
These files will be packaged in a Docker image and pushed to Amazon ECR.
  * `main.py`: `SAGEMAKER_PROGRAM` script. Script which SageMaker will run as soon as Docker container is loaded. The 
environment variable `SAGEMAKER_PROGRAM` is set in `Dockerfile.sagemaker`.
  * `<*>.py`: Training, dataloader, config loader, utility scripts.
  * `requirements.txt`: Requirements file to be used when building Docker image.
* `build_and_push.sh`: A shell script to build Docker image and push to Amazon ECR.
* `Dockerfile.sagemaker`: Dockerfile to build Docker image. Provide this Docker file when running `build_and_push.sh`.
* `run_sagemaker_job.py`: The script which the user will have to use to start a SageMaker job from his/her computer.
* `sagemaker_environment.yaml`: To create a SageMaker training environment on the user's computer.

## Config Files

### Managing Configs

I use [YACS](https://github.com/rbgirshick/yacs) as a mean to manage configurations.

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

We can pass training-specific parameters as `hyperparameters` when we create [SageMaker Estimator](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#estimators).
If you happen to have a lot of parameters for your training script, setting individual `hyperparameters` and parsing the arguments in entry script in Docker container will become very cumbersome. Therefore, I prefer having a config file that can be read by the training script.
The config file will be provided to the entry script as an input channel. We are also going to upload this training config file to S3 so that
we can provide team-wide visibility to the experiment's configurations.


## Efficient Debugging

Once you have started a SageMaker training job on a SageMaker training job instance, it can be quite challenging to debug your code properly.
First few tries can be quite agonizing if you haven't properly tested your code before running a SageMaker job on
SageMaker training job instance. Starting a SageMaker training job instance is quite time consuming since it needs to pull 
ECR image and training data before it actually reads the script pointed by `SAGEMAKER_PROGRAM`. If you have an error 
or a bug in your code, you will have to wait 10-15 minutes before you find the error out. 
Therefore, it is always advisable to run your SageMaker job in `local` mode and debug at your heart's 
content before you start a real SageMaker job on SageMaker training instance.

Here, I would like to provide some tips on debugging your code running in a Docker container that you have created for 
SageMaker.
