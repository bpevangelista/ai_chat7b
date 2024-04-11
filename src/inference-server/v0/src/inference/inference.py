from . import log

import torch

class PredictResult():
    def __init__(self, reply_message, new_history_entry):
        self.reply_message = reply_message
        self.new_history_entry = new_history_entry
    
    def __str__(self):
        return f"PredictResult({self.reply_message}, {self.new_history_entry})"

    def __json__(self):
        return {
            "reply_message": self.reply_message,
            "new_history_entry": self.new_history_entry
        }

class InferenceEngine():
    @staticmethod
    def predict(input_message, chat_history, inference_model, chat_persona):
        log.info(f"predict {input_message} {chat_history} {inference_model} {chat_persona.name}")

        # greeting shortcut
        if input_message == "" and chat_history == "":
            return PredictResult(chat_persona.first_message, "")

        model = inference_model.model
        tokenizer = inference_model.tokenizer
        full_prompt, prompt = chat_persona.build_prompt(input_message, chat_history)
        log.info(f"full_prompt {full_prompt}")

        input_tokens = tokenizer(full_prompt, return_tensors="pt")
        input_ids = input_tokens.input_ids
        input_attention_mask = input_tokens.attention_mask
        if inference_model.model_device != "cpu":
            input_ids.to(inference_model.model_device)
            input_attention_mask.to(inference_model.model_device)

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
            return PredictResult("PredictResult No Model", "PredictResult No Model")

        output_tokens = model.generate(input_ids, **predict_params)
        output_texts = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        #output_texts = tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
    
        # debug
        log.info(f"output_texts {output_texts[0]}")

        gen_text = output_texts[0]
        gen_reply = chat_persona.get_reply(gen_text, full_prompt)
        new_history_entry = f"{prompt}{gen_reply}"
        
        result = PredictResult(gen_reply, new_history_entry)
        log.info(f"result {result}")

        return result
