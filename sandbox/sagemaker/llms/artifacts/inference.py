from datetime import datetime

# IMPORTANT
# Update packages
print(f"{datetime.now()} BEBE pip install packages...")
import subprocess
subprocess.run(["pip", "install", "-r", "requirements.txt"])

import os
import torch
from transformers import AutoTokenizer, pipeline

class ClassFromDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

def get_single_reply(text):
    # remove special tokens
    text = text.replace("<START>", "")
    text = text.replace(", <USER>.", ".")
    text = text.replace(", <USER>!", "!")
    text = text.replace(", <USER>?", "?")
    # fail safe
    text = text.replace("<USER>", "bud")
    text = text.replace("the user", "you")
    text = text.replace("the player", "you")
    text = text.replace("the man", "you")
    text = text.replace("the woman", "you")

    # stop before AI generates new prompt for user
    index = text.find("{user}:")
    if index != -1:
        text = text[:index]
    index = text.find("You:")
    if index != -1:
        text = text[:index]

    # remove incomplete generations at end (due to max length)
    end_phrase = {".", "!", "?", "*"}
    index = max(text.rfind(char) for char in end_phrase)
    if index != -1:
        text = text[:index + 1]

    # remove leading and trailing spaces
    return text.strip()

def model_fn(model_dir):
    print(f"{datetime.now()} BEBE model_fn...")

    file = open("yuki_hinashi.txt", "r")
    char_persona = file.read()
    file.close()

    char_blob = ClassFromDict({
        "name": "Yuki Hinashi",
        "description": char_persona,
        "first_message": "Yuki Hinashi: *Yuki, a blond-haired, brown-eyed girl, leans against a wall, confidently flickering through a book. As you approaches, she raises an eyebrow with curiosity and amusement.* Hey there, you new here? *Yuki smirks, stepping closer with a graceful stride.* I'm Yuki Hinashi.",
        "created_at": "",
    })

    if torch.cuda.is_available():
        print(f"{datetime.now()} BEBE CUDA Available")
        device = "cuda:0"
    else:
        print(f"{datetime.now()} BEBE CUDA NOT Available")

    print(f"{datetime.now()} BEBE torch.load...")
    model_path = os.path.join(model_dir, "pytorch_model.pt")
    print(f"  {model_path}")
    model = torch.load(model_path) # req transformers==4.31.0
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print(f"{datetime.now()} BEBE To CUDA")
    model = model.to("cuda:0")

    print(f"{datetime.now()} BEBE garbage collect")
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    print(f"{datetime.now()} BEBE done!")
    return {
        "character": char_blob,
        "model": model,
        "tokenizer": tokenizer
    }

def predict_fn(request_body, model_tokenizer):
    print(f"{datetime.now()} BEBE predict_fn", request_body)

    input_text = request_body["message"]
    chat_history = request_body["chat_history"]
    tokenizer = model_tokenizer["tokenizer"]
    model = model_tokenizer["model"]
    char = model_tokenizer["character"]

    # greeting shortcut
    if input_text == "" and chat_history == "":
        return {
            "reply": char.first_message,
            "new_history_entry": ""
        }

    # BEBE
    input_prompt = f"{char.description}\n{chat_history}\nYou: *{input_text}*\n"
    #print('BEBE', input_prompt)

    tokenizer.pad_token = tokenizer.eos_token
    input_tokens = tokenizer(input_prompt, return_tensors="pt")

    input_ids = input_tokens.input_ids.to("cuda:0")
    input_attention_mask = input_tokens.attention_mask.to("cuda:0")

    predict_params = {
        "do_sample": True,
        "temperature": 0.8,
        "min_length": len(input_ids[0]) + 8,
        "max_length": len(input_ids[0]) + 96,
        "repetition_penalty": 1.0,
        "top_k": 50,
        "top_p": 0.95,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "attention_mask": input_attention_mask
    }

    output_tokens = model.generate(input_ids, **predict_params)
    output_texts = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    print('BEBE decode', output_texts)

    # remove input from generation
    gen_text = output_texts[0][len(input_prompt):]
    # fix AI char name
    gen_text = gen_text.replace("<BOT>:", f"{char.name}:")

    gen_text = get_single_reply(gen_text)
    print('BEBE reply', gen_text)

    return {
        "reply": gen_text,
        "new_history_entry": f"You: *{input_text}*\n{gen_text}"
    }