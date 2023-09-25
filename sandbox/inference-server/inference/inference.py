from . import log

import torch

class PredictResult():
    def __init__(self, reply_message, new_history_entry):
        self.reply_message = reply_message
        self.new_history_entry = new_history_entry

class InferenceEngine():
    @staticmethod
    def predict(input_message, chat_history, inference_model, chat_persona):
        log.info(f"predict {input_message} {chat_history} {inference_model} {chat_persona}")

        # greeting shortcut
        if input_message == "" and (chat_history == "" or chat_history == []):
            return PredictResult(chat_persona.first_message, "")

        model = inference_model.model
        tokenizer = inference_model.tokenizer
        full_prompt = chat_persona.build_prompt(input_message, chat_history)

        input_tokens = tokenizer(full_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            input_ids = input_tokens.input_ids.to(inference_model.device)
            input_attention_mask = input_tokens.attention_mask.to(inference_model.device)

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
        log.info('output_texts', output_texts[0])

        gen_text = output_texts[0]
        gen_reply = chat_persona.get_reply(gen_text, full_prompt)
        new_history_entry = f"{{user}}: {input_message}\n{{model}}: {gen_reply}"
        
        return PredictResult(gen_reply, new_history_entry)
