#!/bin/bash

if [[ -z "$1" ]]; then
  echo "usage: copy_model_to_s3 s3_packed_models/gpt-j-6b.tar.gz"
  exit 1
fi

# list our sagemaker bucket
SAGE_BUCKET_NAME=$(aws s3api list-buckets --query 'Buckets[?contains(Name, `sagemaker`)].Name' --output text)
echo "sage bucket: $SAGE_BUCKET_NAME"

echo "copying to s3..."
echo "  $1"
aws s3 cp $1 "s3://$SAGE_BUCKET_NAME"