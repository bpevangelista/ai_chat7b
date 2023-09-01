from datetime import datetime
import sys

import boto3

region = "us-west-2"
role = "SageMakerFullAccessRole"
role_arn = "arn:aws:iam::459678513027:role/SageMakerFullAccessRole"
s3_bucket_name = "sagemaker-hf-inference"

sm_client = boto3.client('sagemaker')

def create_endpoint(model_name, instance_type):
    print(f"{datetime.now()} Cleaning up endpoints...")
    try:
        sm_client.delete_endpoint(EndpointName=f"{model_name}-endpoint")
    except Exception as e:
        pass

    try:
        sm_client.delete_endpoint_config(EndpointConfigName=f"{model_name}-config")
        sm_client.delete_model(ModelName=model_name)
    except Exception as e:
        pass

    # My custom docker container
    print(f"{datetime.now()} Retrieving image...")
    image_uri = "459678513027.dkr.ecr.us-west-2.amazonaws.com/hugging-pytorch-inference"

    # AWS containers
    # https://github.com/aws/deep-learning-containers/tree/master
    """
    image_uri = image_uris.retrieve(
        framework="huggingface", # huggingface, pytorch, tensorflow, djl-deepspeed
        region=region,
        version="4.28", # transformers version (start on 4.28, but require 4.31)
        #version="4.31", # required for to_bettertransform
        py_version="py310",
        instance_type=instance_type,
        image_scope="inference",
        base_framework_version="pytorch2.0",
    )

    image_uri = image_uris.retrieve(
        framework="pytorch",
        region=region,
        version="2.0",
        py_version="py310",
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

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("usage: python create_endpoint.py [model_name] [dev|prod] [optional model_version]")
        print("usage: python create_endpoint.py pygmalion-6b dev aug29")
        exit(1)

    model_prefix = sys.argv[1]
    dev_or_prod = sys.argv[2]
    today_str = datetime.now().strftime('%b%d').lower()
    model_suffix = sys.argv[3] if len(sys.argv) >= 4 else today_str
    model_name = f"{model_prefix}-{model_suffix}"

    print(f"{datetime.now()} Listing models...")
    s3 = boto3.client('s3')
    s3_objects = s3.list_objects_v2(Bucket=s3_bucket_name, Prefix=model_prefix, Delimiter='/')
    s3_models = []
    if "CommonPrefixes" in s3_objects:
        s3_objects = s3_objects["CommonPrefixes"]
        s3_models = [ item["Prefix"][:-1] for item in s3_objects]
    print(f"  {s3_models}")

    if model_name in s3_models:
        print(f"  using: {model_name}")
    else:
        print(f"  {model_name} not found!")
        exit(1)

    if dev_or_prod == "prod":
        create_endpoint(model_name, instance_type="ml.p3.2xlarge")  # usd 3.8
    else:
        create_endpoint(model_name, instance_type="ml.g4dn.2xlarge")  # usd 0.94
        # create_endpoint(model_name, instance_type="ml.g5.2xlarge") # usd 1.5

main()

# Testing (may have issues with CUDA not being available)