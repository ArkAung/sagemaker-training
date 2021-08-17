#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR

# 4 arguments can be provided in this script:
#    - image - required, this will be used as the image on the local machine and combined with the account and
#      region to form the repository name for ECR;
#    - tag - optional, if provided, it will be used as ":tag" of your image; otherwise, ":latest" will be used;
#    - Dockerfile.sagemaker - optional, if provided, then docker will try to build image from provided dockerfile
#      (e.g. "Dockerfile.sagemaker.sagemaker"); otherwise, default "Dockerfile.sagemaker" will be used.
#    - --local - optional, if provided, the image will only be built locally
# Usage examples:
#    1. "./build_and_push.sh gan-training build_0_1 Dockerfile.sagemaker.sagemaker" # Build and push to ECR
#    2. "./build_and_push.sh gan-training build_0_1 Dockerfile.sagemaker.sagemaker --local" # Only build locally
image=$1
tag=$2
dockerfile=$3
local=$4

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    echo "Set up AWS CLI first and set AWS credentials"
    exit 255
fi

# Get the region defined in the current configuration
region=$(aws configure get region)

echo "Working in region ${region}"

if [ "$tag" == "" ]
  then
      fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"
  else
      fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:${tag}"
fi

if [ "$local" != "--local" ]
  then
    # If the repository doesn't exist in ECR, create it.
    echo "Checking whether ${image} exists in AWS ECR at region ${region}"
    aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

    if [ $? -ne 0 ]
     then
        echo "${image} does not exist at region ${region}. Creating ${image}."
        aws ecr create-repository --repository-name "${image}" > /dev/null
     fi
    echo "Authenticating ECR to upload to private repo"
    # To upload to private ECR repo
    #aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin "${account}.dkr.ecr.us-east-2.amazonaws.com"
fi

echo "Authenticating ECR to get base container image"
# To get public base container
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-2.amazonaws.com

echo "Building docker image locally"
# Build the docker image locally with the image name
if [ "$dockerfile" == "" ]
  then
    docker build  -t ${image} . --build-arg REGION=${region}
  else
    docker build -t ${image} . -f ${dockerfile} --build-arg REGION=${region}
fi

docker tag ${image} ${fullname}

if [ "$local" != "--local" ]
  then
    echo "Pushing docker image to ECR"
    docker push ${fullname}
fi