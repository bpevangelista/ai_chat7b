import json
import sagemaker, boto3
from sagemaker.huggingface import get_huggingface_llm_image_uri
from sagemaker.huggingface import HuggingFaceModel


sage_context = sagemaker.Session()
sage_bucket = sage_context.default_bucket()

#try:
#role = sagemaker.get_execution_role()
#except ValueError:
#    iam = boto3.client('iam')
#    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

sage_context = sagemaker.Session(default_bucket=sage_bucket)

#print(f"sagemaker role arn: {role}")
#print(f"sagemaker session region: {sage_context.boto_region_name}")

llm_image = get_huggingface_llm_image_uri(
    "huggingface",
    version="0.9.3"
)

print(f"llm image uri: {llm_image}")

config = {
    #'HF_MODEL_ID': "PygmalionAI/pygmalion-6b", # model_id from hf.co/models
    'model_data': 's3:{location_of_your_model}',
    
    'HF_TASK': 'text-generation',
    'SM_NUM_GPUS': "1",
    'MAX_INPUT_LENGTH': "96",
    'MAX_TOTAL_TOKENS': "1024",
}

llm_model = HuggingFaceModel(
    role="SageMakerFullAccessRole",
    image_uri=llm_image,
    env=config
)

llm = llm_model.deploy(
    initial_instance_count=1,
    #instance_type="p3.2xlarge",    # 3.02 usd
    instance_type="ml.p2.xlarge",    # 0.90 usd
    container_startup_health_check_timeout=300,
)
