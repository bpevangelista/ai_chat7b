import json
import requests
import time

import sagemaker, boto3
from sagemaker.huggingface import get_huggingface_llm_image_uri
from sagemaker.huggingface import HuggingFaceModel

def deploy_sage_model():
    sage_context = sagemaker.Session()
    sage_bucket = sage_context.default_bucket()

    sage_context = sagemaker.Session(default_bucket=sage_bucket)
    '''
    llm_image = get_huggingface_llm_image_uri(
        "huggingface",
        version="0.9.3"
    )
    print(f"llm image uri: {llm_image}")
    '''

    env_config = {
        #'HF_MODEL_ID': "PygmalionAI/pygmalion-6b", # model_id from hf.co/models
        'HF_TASK': 'text-generation',
        'SM_NUM_GPUS': "1",
        'MAX_INPUT_LENGTH': "96",
        'MAX_TOTAL_TOKENS': "1024",
    }

    llm_model = HuggingFaceModel(
        model_data="s3://sagemaker-us-west-2-459678513027/pygmalion-6b.tar.gz",
        role="SageMakerFullAccessRole", # Name of my AWS role
        #image_uri=llm_image,
        #env=env_config,

        transformers_version="4.28",
        pytorch_version="2.0",
        py_version='py310'
    )

    llm = llm_model.deploy(
        initial_instance_count=1,
        instance_type="ml.p2.xlarge",                   # ml.p2.xlarge 0.90, ml.p3.2xlarge 3.02
        container_startup_health_check_timeout=600,     # 10min
    )
    return llm

llm = deploy_sage_model()
time.sleep(30)

userPrompt = f"I'm needing a girlfriend."

results = llm.predict({
    "inputs": userPrompt,
    "parameters": {
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 0.8,
        "top_k": 50,
        "min_new_tokens": 8*2,
        "max_new_tokens": 32*2,
        "repetition_penalty": 1.02,
    }
})
print( results[0]["generated_text"] )

llm.delete_model()
llm.delete_endpoint()