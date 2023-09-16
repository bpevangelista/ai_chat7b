from datetime import datetime
import gc, os, shutil, sys, tarfile

import boto3
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# default used TransferConfig() uses multi-part upload
#from boto3.s3.transfer import TransferConfig
#config = TransferConfig()

s3 = boto3.client("s3")
s3_bucket_name = "sagemaker-hf-inference"
hf_auth_token = None

def load_model(model_url):
    print(f"{datetime.now()} Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_url,
                                            #device_map="auto",
                                            offload_folder="offload",
                                            #revision="float16",
                                            torch_dtype=torch.float16,
                                            low_cpu_mem_usage=True,
                                            use_auth_token=hf_auth_token,
                                            )

    # move to GPU not needed to serialize
    #model = model.to("cuda")
    return model

def optimize_model_for_gpu(model):
    # Convert to BetterTransformer
    # https://huggingface.co/docs/transformers/perf_infer_gpu_many
    print(f"{datetime.now()} Converting to Better...")
    model = model.to_bettertransformer()
    return model

def load_tokenizer(model_url):
    print(f"{datetime.now()} Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_url,
        use_auth_token=hf_auth_token,
    )
    return tokenizer

def write_pickled_to_output(model, tokenizer, output_folder, out_model_file_name="pytorch_model.pt"):
    print(f"{datetime.now()} Writing pytorch pickle...")
    print(f"  -->{output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    if model != None:
        torch_model_path = os.path.join(output_folder, out_model_file_name)
        #model.save_pretrained(model_dst_folder)
        torch.save(model, torch_model_path, _use_new_zipfile_serialization=False)
    if tokenizer != None:
        tokenizer.save_pretrained(output_folder)

def copy_artifacts_to_output(output_folder):
    print(f"{datetime.now()} Copying required artifacts...")
    print(f"  artifacts/*.*-->{output_folder}")
    shutil.copytree("artifacts", output_folder, dirs_exist_ok=True)

def get_file_progress_bar(src_path):
    return tqdm(
        total=os.path.getsize(src_path),
        bar_format="{percentage:5.1f}%|{bar:40} | {rate_fmt}",
        unit='B', unit_scale=True, unit_divisor=1024)

def upload_to_s3_raw(output_folder):
    print(f"{datetime.now()} Uploading raw to S3...")
    print(f"  s3://{os.path.join(s3_bucket_name, output_folder)}/")

    for root, _, files in os.walk(output_folder):
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(os.path.basename(output_folder), file)
            print(f"  {src_path}-->{dst_path}")
            with get_file_progress_bar(src_path) as pbar:
                s3.upload_file(src_path, s3_bucket_name, dst_path, Callback=pbar.update)

def upload_to_s3_compressed(output_folder):
    print(f"{datetime.now()} Compressing .tar.gz...")
    tar_path = f"{os.path.dirname(output_folder)}/{os.path.basename(output_folder)}.tar.gz"
    dst_path = os.path.basename(tar_path)
    print(f"  {tar_path}")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(output_folder, arcname=os.path.sep)

    print(f"{datetime.now()} Uploading compressed to S3...")
    print(f"  s3://{os.path.join(s3_bucket_name, dst_path)}/")
    with get_file_progress_bar(tar_path) as pbar:
        s3.upload_file(tar_path, s3_bucket_name, dst_path, Callback=pbar.update)

def free_unused_memory():
    torch.cuda.empty_cache()
    gc.collect()

def process_argv():
    if len(sys.argv) < 3:
        print("usage: python pack_for_s3.py [hf_model_url] [version] [--gpu|--upload_tar]")
        print("usage: python pack_for_s3.py \"meta-llama/Llama-2-7b-chat-hf\" today")
        print("  hf_model_url:")
        print("    PygmalionAI/pygmalion-6b")
        print("    meta-llama/Llama-2-7b-chat-hf")
        print("  version: [today | custom_string]")
        print("  options list")
        print("    --gpu: upload additional gpu optimized model")
        print("    --upload_tar: upload additional compressed .tar.gz model")
        exit(1)

    model_url = sys.argv[1]
    version_str = sys.argv[2].lower()
    options_argv = sys.argv[3:]
    options = {arg[2:]: True for arg in options_argv if arg.startswith("--") and len(arg) > 2}
    return model_url, version_str, options

def write_and_upload_model(model, tokenizer, output_folder, options):
    write_pickled_to_output(model, tokenizer, output_folder)
    copy_artifacts_to_output(output_folder)
    free_unused_memory()
    upload_to_s3_raw(output_folder)
    free_unused_memory()
    if "upload_tar" in options:
        upload_to_s3_compressed(output_folder)
        free_unused_memory()

def main():
    model_url, version_str, options = process_argv()

    # handle optional access token
    if "HF_ACCESS_TOKEN" in os.environ:
        hf_auth_token = os.environ["HF_ACCESS_TOKEN"]
    else:
        print("  HF_ACCESS_TOKEN env not defined. Loading model without access token!")

    today_str = datetime.now().strftime("%b%d")  # e.g. sep01, aug28
    version_str = today_str if version_str == "today" else version_str
    output_folder = f"{os.path.basename(model_url)}-{version_str}".lower()
    base_output_folder = "s3_outputs"
    output_folder = f"{base_output_folder}/{output_folder}"

    # cpu/gpu non-optimized model
    model = load_model(model_url)
    tokenizer = load_tokenizer(model_url)
    write_and_upload_model(model, tokenizer, output_folder, options)

    # gpu-only model
    if "gpu" in options:
        model = optimize_model_for_gpu(model)
        write_and_upload_model(model, tokenizer, f"{output_folder}-gpu", options)
    
main()