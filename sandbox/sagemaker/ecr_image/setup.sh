#!/bin/bash

# install docker -- add repo
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list
# install docker
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli
# fix docker permission
sudo groupadd docker
sudo usermod -aG docker ${USER}
# run docker
#dockerd
sudo service docker start

# create ecr repository
aws ecr create-repository --repository-name hugging-pytorch-inference --encryption-configuration encryptionType=AES256 --image-scanning-configuration scanOnPush
=true
# list ecr repositories
IMAGE_URI=$(aws ecr describe-repositories --query 'repositories[].repositoryUri' --output text)

# docker build image
docker build docker -t $IMAGE_URI
# docker connect to aws
aws ecr get-login-password | docker login --username AWS --password-stdin $IMAGE_URI
# docker push image
docker push $IMAGE_URI
# debug issues (bash into it)
#docker run -it --entrypoint /bin/bash  $IMAGE_URI