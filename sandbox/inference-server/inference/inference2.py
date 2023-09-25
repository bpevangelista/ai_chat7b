from datetime import datetime

import os, re, requests, sys, yaml
import torch
from enum import Enum
from transformers import AutoTokenizer

cuda_device = "cuda:0"

class ClassFromDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
    def __setattr__(self, attr, value):
        self[attr] = value

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

def copy_over_gdoc_persona():
    gdoc_in_out_list = [
        ("1tRxEPX-b5nInMVdsr7hkIGrQ4h8Jm-x1-97yZx6dzwQ", "custom1"),
        ("18HHK9UrT-OoezSULAJjil8fzWvs8vu_SS2xq5yVyqtA", "custom2"),]

    for gdoc_in_out in gdoc_in_out_list:
        try:
            gdoc_in = f"https://docs.google.com/document/d/{gdoc_in_out[0]}/export?format=txt"
            gdoc_out = f"/tmp/{gdoc_in_out[1]}.yaml"
            gdown.download(gdoc_in, gdoc_out)
        except Exception as e:
            print(f"  error: {gdoc_in}-->{gdoc_out} {e}")


class MLModelType(Enum):
    GPTJ = "gptj"
    LLAMA2 = "llama2"

def clean_multi_spaces(str):
    new_str = re.sub(r"\n+", "\n", re.sub(r"[ \t]+", " ", str.strip()))
    return re.sub(r"\n ", "\n", new_str)

def update_persona_for_gptj(persona):
    new_history = persona.hidden_history
    new_history = re.sub(r"{{user}}", "You: ", new_history)
    new_history = re.sub(r"{{model}}", f"{persona.name}: ", new_history)
    if "hidden_history" in persona:
        del persona["hidden_history"]
    
    persona.description = clean_multi_spaces(f"""
    {persona.name}'s Persona: {persona.description}
    <START>
    {new_history}
    """)
    
    print(persona)
    return persona


def update_persona_for_llama2(persona):
    new_history = persona.hidden_history
    new_history = re.sub(r"{{user}}", "<|user|>", new_history)
    new_history = re.sub(r"{{model}}", "<|model|>", new_history)
    if "hidden_history" in persona:
        del persona["hidden_history"]

    persona.description = clean_multi_spaces(f"""
    <|system|>Enter RP mode. Pretend to be {{{{{persona.name}}}}} whose persona follows:
    {persona.description}
    You shall reply to the user while staying in character, and generate long responses.
    {new_history}
    """)

    print(persona)
    return persona

def get_persona_for_model(yaml_data, ml_model_type):
    persona = ClassFromDict(yaml.safe_load(yaml_data))
    for attr in dir(persona):
        attr_value = getattr(persona, attr)
        if isinstance(attr_value, str):
            setattr(persona, attr, clean_multi_spaces(attr_value))

    if ml_model_type == MLModelType.GPTJ:
        return update_persona_for_gptj(persona)
    elif ml_model_type == MLModelType.LLAMA2:
        return update_persona_for_llama2(persona)
    else:
        # error and default to llama2
        print(f"BEBE error! Invalid Model Type {ml_model_type}")
        return update_persona_for_llama2(persona)


def load_all_personas():
    personas = {}
    listdir = os.listdir()
    #listdir = os.listdir() + ["/tmp/" + file for file in os.listdir("/tmp/")] # current and /tmp/
    for yaml_file in [file for file in listdir if file.endswith(".yaml")]:
        try:
            with open(yaml_file, "r") as yaml_data:
                key = os.path.splitext(os.path.basename(yaml_file))[0]
                personas[key] = get_persona_for_model(yaml_data, MLModelType.LLAMA2)
        except Exception as e:
            print(f"  error reading: {yaml_file} {e}")
    if len(personas) < 1:
        print(f"  error no persona found!")
    return personas


def model_fn(model_dir, debug_skip_model=False):
    print(f"{datetime.now()} BEBE model_fn", model_dir)

    if torch.cuda.is_available():
        print(f"{datetime.now()} BEBE CUDA Available")
    else:
        print(f"{datetime.now()} BEBE CUDA NOT Available")

    print(f"{datetime.now()} BEBE torch.load...")
    model_path = os.path.join(model_dir, "pytorch_model.pt")
    print(f"  {model_path}")
    model = None
    if debug_skip_model == False:
        model = torch.load(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    if model != None and torch.cuda.is_available():
        print(f"{datetime.now()} BEBE To CUDA")
        model = model.to(cuda_device)

    print(f"{datetime.now()} BEBE garbage collect")
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    print(f"{datetime.now()} BEBE personas.load...")
    #copy_over_gdoc_persona()
    personas = load_all_personas()
    print(f"{datetime.now()} BEBE personas", list(personas.keys()))

    print(f"{datetime.now()} BEBE done!")
    return {
        "personas": personas,
        "model": model,
        "tokenizer": tokenizer
    }

def predict_fn(request_body, loaded_blob):
    print(f"{datetime.now()} BEBE predict_fn", request_body)

    input_message = request_body["message"]
    chat_history = request_body["chat_history"]
    persona_id = request_body["persona_id"] if "persona_id" in request_body else None
    tokenizer = loaded_blob["tokenizer"]
    model = loaded_blob["model"]
    personas = loaded_blob["personas"]

    # handle custom gdoc personas
    custom_persona_to_gdoc = {
        "custom1": "1tRxEPX-b5nInMVdsr7hkIGrQ4h8Jm-x1-97yZx6dzwQ",
        "custom2": "18HHK9UrT-OoezSULAJjil8fzWvs8vu_SS2xq5yVyqtA",
    }
    if persona_id in custom_persona_to_gdoc:
        gdoc_id = custom_persona_to_gdoc[persona_id]
        response = requests.get(f"https://docs.google.com/document/d/{gdoc_id}/export?format=txt")
        if response.status_code == 200:
            try:
                personas[persona_id] = get_persona_for_model(response.content, MLModelType.LLAMA2)
            except Exception as e:
                print(f"  error yaml load: {gdoc_id} {e}")
        else:
            print(f"  error response: {gdoc_id} {response.status_code}")

    # get persona_if or first persona
    first_persona = next(iter(personas.values()))
    persona = personas.get(persona_id, first_persona)

    # greeting shortcut
    if input_message == "" and chat_history == "":
        return {
            "reply": persona.first_message,
            "new_history_entry": ""
        }

    # build prompt correctly
    full_prompt = f"{persona.description}\n{chat_history}\nYou: *{input_message}*\n"

    tokenizer.pad_token = tokenizer.eos_token
    input_tokens = tokenizer(full_prompt, return_tensors="pt")

    if torch.cuda.is_available():
        input_ids = input_tokens.input_ids.to(cuda_device)
        input_attention_mask = input_tokens.attention_mask.to(cuda_device)

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

    # debug early exit
    if model == None:
        return None

    output_tokens = model.generate(input_ids, **predict_params)
    #output_texts = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    output_texts = tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
    # debug
    print('BEBE output_texts', output_texts)

    # remove input from generation
    gen_text = output_texts[0][len(full_prompt):]
    # fix AI char name
    gen_text = gen_text.replace("<BOT>:", f"{persona.name}:")
    gen_text = get_single_reply(gen_text)
    new_history_entry = f"You: *{input_message}*\n{gen_text}"
    print('BEBE prompt/reply', new_history_entry)

    torch.cuda.empty_cache()

    return {
        "reply": gen_text,
        "new_history_entry": new_history_entry,
    }

# local debug
if __name__ == "__main__" and len(sys.argv) == 2:
    if sys.argv[1] == "debug":
        loaded_blob = model_fn("./", True)
        request_body = {
            "message": "Ola Yuki!",
            "chat_history": [],
        }
        predict_fn(request_body, loaded_blob)

        request_body["persona_id"] = "wrong-key"
        predict_fn(request_body, loaded_blob)
        request_body["persona_id"] = "custom1"
        predict_fn(request_body, loaded_blob)
        request_body["persona_id"] = "custom2"
        predict_fn(request_body, loaded_blob)
        request_body["persona_id"] = "yuki_hinashi_en"
        predict_fn(request_body, loaded_blob)
        request_body["persona_id"] = "yuki_hinashi_en_v0"
        predict_fn(request_body, loaded_blob)

        #request_body["persona_id"] = "yuki_hinashi_en2"
        #predict_fn(request_body, loaded_blob)
    elif sys.argv[1] == "summary":
        loaded_blob = model_fn("./")
        print(loaded_blob["model"])