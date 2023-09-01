from datetime import datetime
import os, shutil, tarfile

import boto3
import torch
from transformers import GPTJForCausalLM, AutoTokenizer


s3 = boto3.resource("s3")
s3_bucket_name = 'sagemaker-hf-inference'
s3_bucket = s3.Bucket(s3_bucket_name)

from boto3.s3.transfer import TransferConfig
config = TransferConfig()

#model_url = 'EleutherAI/gpt-j-6B'
model_url = 'PygmalionAI/pygmalion-6b'
model_basename = os.path.basename(model_url) # e.g.pygmalion-6b

today_str = datetime.now().strftime('%b%d').lower()  # e.g. sep01, aug28
out_model_name = f'{model_basename}-{today_str}'

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
    model.to_bettertransformer()
    return model

def load_tokenizer():
    print(f'{datetime.now()} Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_url)
    return tokenizer

def write_pickled(model, tokenizer):
    print(f'{datetime.now()} Writing pytorch pickle...')
    os.makedirs(out_model_name, exist_ok=True)

    torch_model_path = os.path.join(out_model_name, 'pytorch_model.pt')
    #model.save_pretrained(model_dst_folder)
    torch.save(model, torch_model_path)
    tokenizer.save_pretrained(out_model_name)

def copy_artifacts():
    print(f'{datetime.now()} Copying required artifacts...')
    print(f'  ../../artifacts-->{out_model_name}')
    shutil.copytree('../../artifacts', out_model_name, dirs_exist_ok=True)

def upload_to_s3_raw():
    print(f'{datetime.now()} Uploading raw to S3...')
    print(f'  s3://{os.path.join(s3_bucket_name, out_model_name)}/')

    for root, _, files in os.walk(out_model_name):
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(out_model_name, file)
            print(f'  {src_path}-->{dst_path}')
            s3_bucket.upload_file(src_path, dst_path)

def upload_to_s3_compressed():
    print(f'{datetime.now()} Compressing to .tar.gz...')
    packed_model_path = os.path.join('../../s3_packed_models/',
                                     f'{out_model_name}.tar.gz')

    print(f'  {packed_model_path}')
    with tarfile.open(packed_model_path, "w:gz") as tar:
        tar.add(packed_model_path, arcname=os.path.sep)

    print(f'{datetime.now()} Uploading compressed to S3...')
    s3_bucket.upload_file(packed_model_path, os.path.basename(packed_model_path))

# TODO Support multi-part upload for speeding up
def main():
    write_pickled(load_model(), load_tokenizer())
    copy_artifacts()

    #upload_to_s3_compressed()
    upload_to_s3_raw()

main()