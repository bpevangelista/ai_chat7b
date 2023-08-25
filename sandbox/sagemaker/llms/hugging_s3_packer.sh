#!/bin/bash

# install git lfs
if ! git lfs &> /dev/null; then
  sudo apt install git-lfs
  git lfs install
fi

if [[ -z "$1" ]] || [[ -z "$2" ]]; then
  echo "usage: hugging_s3_packer EleutherAI/gpt-j-6b float16"
  echo "usage: hugging_s3_packer EleutherAI/gpt-j-6b main"
  exit 1
fi

MODELS_BASE_FOLDER="models"
S3_PACKED_MODELS_FOLDER="s3_packed_models"

REPO_NAME=$1
REPO_BRANCH_NAME=$2
REPO_FOLDER=$(basename "$1")

# save path
pushd . >/dev/null
cd "$MODELS_BASE_FOLDER"

# cloning
echo -e "\ncloning repo..."

if [ ! -d "$REPO_FOLDER/.git" ]; then
  echo "git clone -b $REPO_BRANCH_NAME https://huggingface.co/$REPO_NAME"
  git clone -b $REPO_BRANCH_NAME https://huggingface.co/$REPO_NAME
else
  echo "  already done"
fi

# save path
cd "$REPO_FOLDER"

# copy requirements file
echo -e "\ncreating requirements.txt..."
if [[ ! -f "./code/requirements.txt" ]]; then
  mkdir code &>/dev/null
  cp ../../requirements.txt ./code/requirements.txt
else
  echo "  already done"
fi

# zip it now
echo -e "\nzipping..."
tar --exclude="./.*" -czvf "../../$S3_PACKED_MODELS_FOLDER/$REPO_FOLDER".tar.gz .

# restore path
popd >/dev/null
echo -e "\ndone!"