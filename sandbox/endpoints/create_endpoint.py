from datetime import datetime
import os, sys

import boto3
from sagemaker import image_uris # sagemaker only used for images

region = "us-west-2"
role = "SageMakerFullAccessRole"
role_arn = "arn:aws:iam::459678513027:role/SageMakerFullAccessRole"
s3_bucket_name = "sagemaker-hf-inference"

s3 = boto3.client("s3")
sm_client = boto3.client("sagemaker")

def cleanup_endpoint(model_name):
    print(f"{datetime.now()} Cleaning up endpoints...")
    try:
        sm_client.delete_endpoint(EndpointName=f"{model_name}-endpoint")
    except Exception as e:
        pass
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=f"{model_name}-config")
    except Exception as e:
        pass
    try:
        sm_client.delete_model(ModelName=model_name)
    except Exception as e:
        pass

def create_endpoint(model_name, instance_type):
    cleanup_endpoint(model_name)
    
    # My custom docker container
    print(f"{datetime.now()} Retrieving image...")
    image_uri = "459678513027.dkr.ecr.us-west-2.amazonaws.com/hugging-pytorch-inference"

    """
    # AWS containers
    # https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html
    # https://github.com/aws/deep-learning-containers/tree/master
    image_uri = image_uris.retrieve(
        framework="huggingface", # huggingface, pytorch, tensorflow, djl-deepspeed
        region=region,
        version="4.28", # transformers version (start on 4.28, but require 4.31)
        py_version="py310",
        instance_type=instance_type,
        image_scope="inference",
        base_framework_version="pytorch2.0",
    )
    """
    """
    image_uri = image_uris.retrieve(
        framework="pytorch", # huggingface, pytorch, tensorflow, djl-deepspeed
        region=region,
        version="1.12.1",
        py_version="py38",
        instance_type=instance_type,
        image_scope="inference",
    )
    """
    print(f"  {image_uri}")
    
    print(f"{datetime.now()} Creating model...")
    sagemaker_model = sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            # https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html
            "Image": image_uri,
            #"ModelDataUrl": f"s3://{s3_bucket_name}/{model_name}.tar.gz", # compressed
            "ModelDataSource": { # uncompressed
                "S3DataSource": {
                    "CompressionType": "None",
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{s3_bucket_name}/{model_name}/",
                },
            },
            "Environment": {
            }
        },
        ExecutionRoleArn = role_arn,
    )

    print(f"{datetime.now()} Creating endpoint...")
    sm_client.create_endpoint_config(
        EndpointConfigName=f"{model_name}-config",
        ProductionVariants=[
            {
                "VariantName": f"{model_name}-variant",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
                "ModelDataDownloadTimeoutInSeconds": 600,   # 10min
            }
        ],
    )

    endpoint = sm_client.create_endpoint(
        EndpointName=f"{model_name}-endpoint",
        EndpointConfigName=f"{model_name}-config"
    )

    print(f"  to-delete: aws sagemaker delete-endpoint --endpoint-name {model_name}-endpoint")
    return endpoint

def upload_latest_inference(model_name):
    print(f"{datetime.now()} Uploading latest inference.py...")
    code_file_name = "inference.py"
    src_path = f"../llms/artifacts/{code_file_name}"
    dst_path = f"{model_name}/{code_file_name}"
    print(f"  {src_path}-->{dst_path}")
    s3.upload_file(src_path, s3_bucket_name, dst_path)

def upload_latest_artifacts(model_name):
    print(f"{datetime.now()} Uploading latest artifacts...")
    for root, _, files in os.walk("../llms/artifacts"):
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(model_name, file)
            print(f"  {src_path}-->{dst_path}")
            s3.upload_file(src_path, s3_bucket_name, dst_path)

def list_models():
    print(f"{datetime.now()} Listing models...")
    s3_objects = s3.list_objects_v2(Bucket=s3_bucket_name, Prefix="", Delimiter="/")
    s3_models = []
    if "CommonPrefixes" in s3_objects:
        s3_objects = s3_objects["CommonPrefixes"]
        s3_models = [ item["Prefix"][:-1] for item in s3_objects]
    return s3_models

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4 or sys.argv[2] not in ["dev-cpu", "dev-gpu", "prod"]:
        print("usage: python create_endpoint.py [model_name] [dev-cpu|dev-gpu|prod]")
        print("usage: python create_endpoint.py pygmalion-6b-sep01 dev")
        exit(1)

    model_name = sys.argv[1]
    dev_or_prod = sys.argv[2]

    s3_models = list_models()
    print(f"  {s3_models}")
    if model_name in s3_models:
        print(f"  --> {model_name} (selected)")
    else:
        print(f"  {model_name} not found!")
        exit(1)

    if dev_or_prod == "prod":
        instance_name = "ml.p3.2xlarge"     # usd 3.8
    elif dev_or_prod == "dev-gpu":
        instance_name = "ml.g4dn.xlarge"    # usd $0.7364 ($0.5260 as ec2)
        #instance_name = "ml.g4dn.2xlarge"   # gpu + works, usd 0.94
    elif dev_or_prod == "dev-cpu":
        #instance_name = "ml.c7g.2xlarge"    # usd 0.37
        instance_name = "ml.c6i.2xlarge"    # usd 0.37

    print(f"  --> {instance_name} (selected)")
    upload_latest_artifacts(model_name)
    create_endpoint(model_name, instance_type=instance_name)

main()