from datetime import datetime
import os, shutil, tarfile

import boto3
import torch
from transformers import GPTJForCausalLM, AutoTokenizer

s3 = boto3.resource("s3")
s3_bucket_name = 'sagemaker-hf-inference'
s3_bucket = s3.Bucket(s3_bucket_name)

#model_url = 'EleutherAI/gpt-j-6B'
model_url = 'PygmalionAI/pygmalion-6b'
model_dst_folder = os.path.basename(model_url)

def load_model():
    print(f'{datetime.now()} Loading model...')
    model = GPTJForCausalLM.from_pretrained(model_url,
                                            #device_map="auto",
                                            offload_folder="offload",
                                            #revision="float16",
                                            torch_dtype=torch.float16,
                                            low_cpu_mem_usage=True
                                            )

    print(f'{datetime.now()} Converting to Better...')
    # Convert to BetterTransformer
    # https://huggingface.co/docs/transformers/perf_infer_gpu_many
    #model.to_bettertransformer()
    return model

def load_tokenizer():
    print(f'{datetime.now()} Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_url)
    return tokenizer

def write_pickled(model, tokenizer):
    print(f'{datetime.now()} Writing pytorch pickle...')
    os.makedirs(model_dst_folder, exist_ok=True)

    torch_model_path = os.path.join(model_dst_folder, 'pytorch_model.pt')
    if os.path.exists(torch_model_path):
        print('  already done')
    else:
        #model.save_pretrained(model_dst_folder)
        torch.save(model, torch_model_path)
    tokenizer.save_pretrained(model_dst_folder)

def copy_artifacts():
    print(f'{datetime.now()} Copying required artifacts...')
    print(f'  ../_sagemaker_artifacts-->{model_dst_folder}')
    shutil.copytree('../_sagemaker_artifacts', model_dst_folder, dirs_exist_ok=True)

def upload_to_s3_raw():
    print(f'{datetime.now()} Uploading to S3...')
    print(f'  s3://{os.path.join(s3_bucket_name, model_dst_folder)}/')
    for root, _, files in os.walk(model_dst_folder):
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(model_dst_folder, file)
            print(f'  {src_path}-->{dst_path}')
            s3_bucket.upload_file(src_path, dst_path)

def upload_to_s3_compressed():
    today_str = datetime.now().strftime('%b%d').lower() # e.g. Sep01, Aug28
    print(f'{datetime.now()} Compressing to .tar.gz...')
    tar_name = os.path.join('../../s3_packed_models/',
                            f'{model_dst_folder}-{today_str}.tar.gz')
    print(f'  {tar_name}')
    with tarfile.open(tar_name, "w:gz") as tar:
        tar.add(model_dst_folder, arcname=os.path.sep)

    print(f'{datetime.now()} Uploading to S3...')
    s3_bucket.upload_file(tar_name, os.path.basename(tar_name))

model = load_model()
tokenizer = load_tokenizer()
write_pickled(model, tokenizer)
copy_artifacts()
upload_to_s3_compressed()
#upload_to_s3_raw()