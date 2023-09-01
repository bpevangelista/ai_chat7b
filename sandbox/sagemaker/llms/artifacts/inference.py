from datetime import datetime

print(f'{datetime.now()} BEBE pip install packages...')
import subprocess
import importlib
subprocess.run(["pip", "install", "-r", "requirements.txt"])
importlib.import_module("transformers")

import os
import torch
from transformers import AutoTokenizer, pipeline

def model_fn(model_dir):
    print(f'{datetime.now()} BEBE model_fn...')

    if torch.cuda.is_available():
        print(f'{datetime.now()} BEBE CUDA Available')
        device = 0
    else:
        print(f'{datetime.now()} BEBE CUDA NOT Available')
        device = -1

    print(f'{datetime.now()} BEBE torch.load...')
    model_path = os.path.join(model_dir, 'pytorch_model.pt')
    print(f'  {model_path}')
    model = torch.load(model_path) # req transformers==4.31.0
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print(f'{datetime.now()} BEBE pipeline...')
    generation = pipeline(
        task="text-generation", model=model, tokenizer=tokenizer, device=device
    )

    print(f'{datetime.now()} BEBE done!')
    return generation