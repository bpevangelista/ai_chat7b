from startup import DefaultLogger
import torch

log = DefaultLogger("model")
cuda_device = "cuda:0"

class InferenceSpec():
    blah
    
class Inference():
    class ModelType(Enum):
        GPTJ = "gptj"
        LLAMA2 = "llama2"
    def __init__(model_type, model, tokenizer):
        self.model_type = model_type
        self.model = model
        self.tokenizer = tokenizer

def free_unused_memory():
    torch.cuda.empty_cache()
    import gc
    gc.collect()


def model_fn(model_dir, debug_skip_model=False):
    log.info(f"model_fn {model_dir} {debug_skip_model}")
    log.info(f"CUDA available: {torch.cuda.is_available()}")

    log.info("torch.load...")
    model = None
    if not debug_skip_model:
        torch.load(f"{model_dir}/pytorch_model.pt")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    if model and torch.cuda.is_available():
        log.info(f"model.to({cuda_device}")
        model = model.to(cuda_device)
    
    free_unused_memory()
    return ModelTokenizer(model, tokenizer)


def persona_hot_reloader():
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


def build_prompt(prompt, persona, model_type):
    blah()


def predict_fn(request_body, model_tokenizer, personas):
    log.info(f"predict_fn {request_body} {loaded_blob}")

    input_message = request_body["message"]
    chat_history = request_body["chat_history"]
    persona_id = request_body["persona_id"]

    persona = personas.get(persona_id)

    # greeting shortcut
    if input_message == "" and chat_history == "":
        return {
            "reply": persona.first_message,
            "new_history_entry": ""
        }

    # build prompt correctly
    #full_prompt = f"{persona.description}\n{chat_history}\nYou: *{input_message}*\n"
    full_prompt = build_prompt(input_message, persona, model_type)

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

model_fn("")