import json
import requests

import sagemaker, boto3
from sagemaker.huggingface import get_huggingface_llm_image_uri
from sagemaker.huggingface import HuggingFaceModel

def sage_deploy_model():
    sage_context = sagemaker.Session()
    sage_bucket = sage_context.default_bucket()

    sage_context = sagemaker.Session(default_bucket=sage_bucket)

    llm_image = get_huggingface_llm_image_uri(
        "huggingface",
        version="0.9.3"
    )
    print(f"llm image uri: {llm_image}")

    env_config = {
        #'HF_MODEL_ID': "PygmalionAI/pygmalion-6b", # model_id from hf.co/models
        'HF_TASK': 'text-generation',
        'SM_NUM_GPUS': "1",
        'MAX_INPUT_LENGTH': "96",
        'MAX_TOTAL_TOKENS': "1024",
    }

    llm_model = HuggingFaceModel(
        model_data="s3://llms_sep2023/pygmalion-6b.tar.gz",

        role="SageMakerFullAccessRole", # Name of my AWS role
        #image_uri=llm_image,
        env=env_config,
        transformers_version="4.26",                           # Transformers version used
        pytorch_version="1.13",                                # PyTorch version used
        py_version='py39'
    )

    predictor = llm_model.deploy(
        initial_instance_count=1,
        #instance_type="p3.2xlarge",        # 3.02 usd
        instance_type="ml.p2.xlarge",       # 0.90 usd
        container_startup_health_check_timeout=360, # 6min
    )

    return predictor

#predictor = sage_deploy_model()