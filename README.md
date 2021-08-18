## SageMaker Training Structure and Flow

This repo demonstrates how a custom SageMaker training structure and flow can look like.
The main goals are:
* I want to run custom training scripts on SageMaker.
* I want to start a SageMaker training job from my computer using a single command.
* I want to configure parameters for the SageMaker job and for training using config files rather
than modifying parameters in scripts.
* For every experiment I run, I want to provide visibility to my ML team. I want my team to quickly
know what parameters I have used to start my training job on SageMaker and how they can replicate it.

To meet these goals, I will:
* Build a Docker image with PyTorch base image and my custom training scripts added to it.
* Have a script called `run_sagemaker_job.py` which I can run from my computer with a config file.
* Have two YAML files: one for SageMaker related configs and one for training-related configs.
* Upload the configs that I have used to start my SageMaker job as well as the config file which contains training
parameters to S3. 
* Upload results (trained model, model outputs, model logs) to S3.

## Preliminary Steps

A couple of things need to be set up.

On my computer:
* Conda environment having SageMaker SDK
  * This can be done with `conda env create --file=sagemaker_environment.yaml`
* AWS CLI tools with AWS credentials
  * [Setting up AWS CLI tool](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
  * SageMaker job will be started according to AWS credentials created using AWS CLI tool.
* Set up Docker and [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

On AWS:
* Create a SageMaker Execution Role and get the ARN
* Upload training data on S3

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

I use [YACS](https://github.com/rbgirshick/yacs) to manage configurations. With YACS, I create 
default config templates for SageMaker `sagemaker_default_config.py` and training `sagemaker_container/training_default_config.py`.
They serve as the one-stop reference points for all configurable options. They should be well documented and provide sensible 
defaults for all options. Every ML team member should be able to easily understand what each parameter is by referencing
these when creating configs for their experiments (e.g. `configs/sagemaker_config.yaml` and `configs/training_config.yaml`).


### SageMaker Config

`sagemaker_default_config.py` holds all the possible parameters of SageMaker config (and defaults). If the user wants
to update the default parameters, the user can create a SageMaker config file like `configs/sagemaker_config.yaml`. 
All the parameters required to start a SageMaker job is set in this config file. This includes:
* Where the training data is on S3
* Where the configs will be uploaded on S3
* Where the outputs will be saved on S3
* ECR image that will be used for training
* What instances, how many instances, and whether to use spot instances or not for training
* ARN for SageMaker Execution Role
* Training specific config file

### Training Specific Config

We can pass training-specific parameters as `hyperparameters` when we create [SageMaker Estimator](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#estimators).
If you happen to have a lot of parameters for your training script, setting individual `hyperparameters` and parsing the arguments in entry script in Docker container will become very cumbersome. 
I prefer having a config file that can be read by the training script. The config file will be provided to the entry script as an input channel.
We are also going to upload this training config file to S3 so that we can provide team-wide visibility to the experiment's configurations.

`sagemaker_container/training_default_config.py` holds all the possible parameters of training (and defaults). If the user wants
to update the default parameters, the user can create a training config file like `configs/training_config.yaml` and update
the value for `TRAINING.CONFIG_FILE` in SageMaker config file.

## Building Docker Image for SageMaker

Build Docker image by running `./build_and_push.sh <docker_image_name> <tag> <docker file>`.
For example, `./build_and_push.sh gan_training build_0_1 Dockerfile.sagemaker` will pull
the base docker image, create an ECR repository with the <docker_image_name> if it does not exist, 
build and tag docker image and push to ECR.

After you have built it, you can check the built docker image with `docker images`. You will
see your pushed Docker image on Amazon ECR as well (make sure to choose the right region).

You can check the files in Docker image by running:
`docker run -it --entrypoint /bin/bash <docker_image_id>`.

## Efficiently Testing and Debugging Training Scripts

Once you have started a SageMaker training job on a SageMaker training job instance, it can be quite challenging to debug your code properly.
First few tries can be quite agonizing if you haven't properly tested your code before running a SageMaker job on
SageMaker training job instance. Starting a SageMaker training job instance is quite time consuming since it needs to pull 
ECR image and training data before it executes the script pointed by `SAGEMAKER_PROGRAM`. If you have an error 
or a bug in your code, you will have to wait 10-15 minutes before you find the error out. 
Therefore, it is always advisable to run your SageMaker job in `local` mode and debug at your heart's 
content before you start a real SageMaker job on SageMaker training instance. Even if you 
independently test your training code, debugging whether your script has got the correct file and directory paths in the 
Docker container can be quite annoying.

To test the Docker container running your training code in `local` mode:
1. Set `INSTANCE_TYPE` in your SageMaker config to `local` or `local-gpu`
2. Make changes in your code
3. Re-build (with `build_and_push.sh`) Docker image to incorporate this change
4. Re-run SageMaker job (with `run_sage_maker_job.py`)
5. Check for errors in your code. If there is an error, go back to step 2.

Debugging code with `print` statements or logging to a file is not very efficient. 
Using debuggers make your debugging experience a lot smoother. Here is how you
can use [Python Debugger (pdb)](https://docs.python.org/3/library/pdb.html) to debug
code running in Docker container built for SageMaker training job.

1. Make changes in your code
2. Place `import pdb; pdb.set_trace()` in your code as a breakpoint. Program execution will
pause when it reaches that line.
3. Re-build (with `build_and_push.sh`) Docker image to incorporate additional breakpoint line.
4. Re-run SageMaker job (with `run_sagemaker_job.py`).
5. The standard output will pause when it reaches the breakpoint. However, you will
not get an interactive `pdb` console. To get the interactive `pdb` console, we have to attach to that running container.
6. In a separate terminal tab, find out the container ID with `docker ps`.
7. Attach to the running container with `docker attach <container_id>`. You can start
debugging your code with `pdb` just like usual.

### Stepping up the debugging game with pudb

`pdb` is a good debugger, but you can be even more efficient with [pudb](https://documen.tician.de/pudb/). `pudb` 
provides all the niceties of modern GUI-based debuggers in a more lightweight and keyboard-friendly package

To use `pudb`, you would have to add `pudb` in `sagemaker_container/requirements.txt`, place a breakpoint with
`import pudb; pudb.set_trace()` and rebuild the Docker image. Attach to running Docker container just like before
and now you will be greeted with a sleek terminal Python Debugger.

View [this video](https://youtu.be/Mhl7j9BsXLA) for a demo on how to debug training code in Docker container.

## Running on SageMaker Training Instances

Now that you have put everything together and tested your training code, you are ready to train on SageMaker training instances:
* Update `INSTANCE_TYPE` in SageMaker config (e.g. configs/sagemaker_config.yaml) to the instance type 
you want to train on (refer [this](https://aws.amazon.com/sagemaker/pricing/); scroll down to the table and click on Training tab)
* Start SageMaker job with `python run_sagemaker_job.py --config configs/sagemaker_config.yaml`
