from datetime import datetime

import boto3, sagemaker
from sagemaker import image_uris

region = "us-west-2"
role = "SageMakerFullAccessRole"
s3_bucket_name = "sagemaker-hf-inference"

def create_endpoint(model_name, instance_type):
    print(f"{datetime.now()} Cleaning up endpoints...")
    try:
        sagemaker_session.delete_endpoint(f"{model_name}-endpoint")
    except Exception as e:
        print(f"  {e}")

    try:
        sagemaker_session.delete_endpoint_config(f"{model_name}-config")
        sagemaker_session.delete_model(model_name)
    except Exception as e:
        print(f"  {e}")

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
    print(f"  {image_uri}")

    print(f"{datetime.now()} Creating Model...")
    sagemaker_model = sagemaker_session.create_model(
        name=model_name,
        role=role,
        container_defs=sagemaker.container_def(
            # https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html
            image_uri=image_uri,
            model_data_url=f"s3://{s3_bucket_name}/{model_name}.tar.gz",
            #model_data_url={
            #    "CompressionType": "None",
            #    "S3DataType": "S3Prefix",
            #    "S3Uri": f"s3://{s3_bucket_name}/{model_name}/",
            #},
            env={
            }
        ),
    )

    print(f"{datetime.now()} Creating Endpoint...")
    sagemaker_session.create_endpoint_config(
        name=f"{model_name}-config",
        model_name=model_name,
        initial_instance_count=1,
        instance_type="ml.p2.xlarge",       # ml.p2.xlarge 0.90, ml.p3.2xlarge 3.02
        model_data_download_timeout=600,    # 10min
    )

    return sagemaker_session.create_endpoint(
        endpoint_name=f"{model_name}-endpoint",
        config_name=f"{model_name}-config"
    )

print(f"{datetime.now()} Creating sagemaker session...")
sagemaker_session = sagemaker.Session()
sage_bucket = sagemaker_session.default_bucket()

create_endpoint("pygmalion-6b-aug28", "ml.p2.xlarge")

#sagemaker_session.delete_endpoint(EndpointName=endpoint_name)
#sagemaker_session.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
#sagemaker_session.delete_model(ModelName=model_name)