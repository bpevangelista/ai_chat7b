#!/bin/bash

sudo apt update
sudo apt install unzip
sudo apt install python3
sudo apt install python3-pip
sudo apt install docker

# aws
mkdir aws_temp && cd aws_temp
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
cd .. && rm -rf aws_temp

echo "configure your aws"
aws configure

#python
pip3 install sagemaker

#AWS_CONFIG_FILE=$(eval echo "~/.aws/config")
#AWS_CONFIG_PATH=$(dirname "$AWS_CONFIG_FILE")
#mkdir -p "$AWS_CONFIG_PATH"
#if [[ ! -f "$AWS_CONFIG_FILE" ]]; then
#  echo -e '' \
#    '[default]\n' \
#    'region=us-west-2\n' > "$AWS_CONFIG_FILE"
#fi
