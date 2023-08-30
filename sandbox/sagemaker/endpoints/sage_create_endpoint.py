from datetime import datetime

import boto3
from sagemaker import image_uris

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

    print(f"{datetime.now()} Retrieving Image...")
    image_uri = image_uris.retrieve(
        framework="huggingface", # huggingface, pytorch, tensorflow, djl-deepspeed
        region=region,
        version="4.28", # transformers version
        py_version="py310",
        instance_type=instance_type,
        image_scope="inference",
        base_framework_version="pytorch2.0",
    )
    """
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

    print(f"{datetime.now()} Creating Model...")
    sagemaker_model = sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            # https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html
            "Image": image_uri,
            #"ModelDataUrl": f"s3://{s3_bucket_name}/{model_name}.tar.gz",
            "ModelDataSource": {
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

    print(f"{datetime.now()} Creating Endpoint...")
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

    return sm_client.create_endpoint(
        EndpointName=f"{model_name}-endpoint",
        EndpointConfigName=f"{model_name}-config"
    )

today_str = datetime.now().strftime('%b%d').lower()
model_name=f"pygmalion-6b-{today_str}"

# ml.p2.xlarge, ml.p3.2xlarge
create_endpoint(model_name, "ml.p3.2xlarge")
