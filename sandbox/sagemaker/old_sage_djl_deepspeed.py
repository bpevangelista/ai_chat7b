from datetime import datetime

import boto3, sagemaker
from sagemaker.model import Model
from sagemaker.djl_inference import DJLModel
from sagemaker.huggingface import HuggingFaceModel
from sagemaker import image_uris

region = "us-west-2"
role = "SageMakerFullAccessRole"
s3_bucket_name = "sagemaker-hf-inference"

def create_model(model_name):
    print(f"{datetime.now()} Getting Image URI...")
    image_uri = sagemaker.image_uris.retrieve(
        framework="djl-deepspeed", # huggingface, pytorch, tensorflow, djl-deepspeed
        region=region,
        version="0.21.0",
    )

    print(f"{datetime.now()} Creating Model...")
    # Don"t I need to use DJLModel here? To take advantage of deepseed?
    model = Model(
        image_uri=image_uri,
        model_data=f"s3://{s3_bucket_name}/{model_name}.tar.gz",
        role=role,
    )
    return model

def deploy_model(model, endpoint_name):
    print(f"{datetime.now()} Deploying Model...")
    predictor = model.deploy(
        endpoint_name=endpoint_name,
        initial_instance_count=1,
        instance_type="ml.p2.xlarge",                   # ml.p2.xlarge 0.90, ml.p3.2xlarge 3.02
        container_startup_health_check_timeout=600,     # 10min
    )
    return predictor

def main():
    model = create_model("pygmalion-6b")
    predictor = deploy_model(model, "pygmalion-6b")


print(f"{datetime.now()} Creating sagemaker session...")
sagemaker_session = sagemaker.Session()
sage_bucket = sagemaker_session.default_bucket()

#sagemaker_session.delete_endpoint(EndpointName=endpoint_name)
#sagemaker_session.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
#sagemaker_session.delete_model(ModelName=model_name)