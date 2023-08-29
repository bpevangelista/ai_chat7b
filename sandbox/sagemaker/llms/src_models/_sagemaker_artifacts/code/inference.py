from datetime import datetime

import os
import torch
from transformers import AutoTokenizer, pipeline

def model_fn(model_dir):
    print(f'{datetime.now()} BEBE model_fn...')
    device = "cuda"

    print(f'{datetime.now()} BEBE torch.load...')
    model = torch.load(os.path.join(model_dir, 'pytorch_model.pt'))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print(f'{datetime.now()} BEBE pipeline...')
    generation = pipeline(
        task="text-generation", model=model, tokenizer=tokenizer, device=device
    )

    print(f'{datetime.now()} BEBE done!')
    return generation