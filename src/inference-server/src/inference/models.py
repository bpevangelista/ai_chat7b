from . import log

import os, torch, transformers
from transformers import AutoTokenizer, GPTJForCausalLM, LlamaForCausalLM

def _free_unused_memory():
    torch.cuda.empty_cache()
    import gc
    gc.collect()

class InferenceModel():
    def __init__(self, model, model_type, model_device, tokenizer):
        self.model = model
        self.model_type = model_type
        self.model_device = model_device
        self.tokenizer = tokenizer
        
    def __str__(self):
        return f"InferenceModel(model_type: '{self.model_type}', model: {type(self.model)}, tokenizer: {type(self.tokenizer)})"

    @staticmethod
    def from_folder(folder_name, debug_skip_model=False):
        log.info(f"from_folder {folder_name} {debug_skip_model}")
        log.info(f"CUDA available: {torch.cuda.is_available()}")
        log.info(f"MLKDNN available: {torch.backends.mkldnn.is_available()}")

        model = None
        model_device = "cpu"
        cuda_device = "cuda:0"

        log.info("torch.load...")
        if not debug_skip_model:
            model = torch.load(f"{folder_name}/pytorch_model.pt")
        tokenizer = AutoTokenizer.from_pretrained(folder_name)
        tokenizer.pad_token = tokenizer.eos_token
        _free_unused_memory()

        if model and torch.cuda.is_available():
            log.info(f"model.to({cuda_device})")
            model_device = cuda_device
            model = model.to(model_device)
            _free_unused_memory()

        model_type = None
        if isinstance(model, transformers.LlamaForCausalLM):
            model_type = "llama2"
        elif isinstance(model, transformers.GPTJForCausalLM):
            model_type = "gptj"

        return InferenceModel(model, model_type, model_device, tokenizer)