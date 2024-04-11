import torch
import re

from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BaseTextPostProcessors:
    def process(self, text: list[str]) -> list[str]:
        raise NotImplementedError


class SingleResponseProcessor(BaseTextPostProcessors):
    def process(self, inputs: list[str]) -> list[str]:
        new_inputs = []
        for input in inputs:
            index = input.find('### ')
            if index != -1:
                new_inputs.append(input[:index])
            else:
                new_inputs.append(input)
        return new_inputs


class CompleteSentenceProcessor(BaseTextPostProcessors):
    def process(self, inputs: list[str]) -> list[str]:
        outputs = []
        pattern = re.compile(pattern=r'.*[\.\?!*"](?=\s|$)', flags=re.DOTALL)
        for input in inputs:
            match = pattern.search(input.strip())
            outputs.append(match.group(0) if match else '')
        return outputs


class BestOfProcessor(BaseTextPostProcessors):
    def __init__(self, reward_model_uri: str):
        super().__init__()
        self.reward_model_uri = reward_model_uri
        self.reward_model = None
        self.reward_tokenizer = None
        self._try_load_model()

    def _try_load_model(self):
        self.reward_tokenizer = AutoTokenizer.from_pretrained(
            self.reward_model_uri,
        )
        self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token

        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.reward_model_uri,
            #torch_dtype=torch.float16,
        )
        self.reward_model.eval()

    def process(self, inputs: list[str]) -> list[str]:
        with torch.no_grad():
            tokens = self.reward_tokenizer(inputs, padding=True, return_tensors='pt')
            input_ids = tokens.input_ids
            attention_mask = tokens.attention_mask

            outputs = self.reward_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # [:, 0] Negative, [:, 1] Positive
            positive_logits = outputs.logits[:, 1]
            best_seq_idx = torch.argmax(positive_logits, dim=-1)

            return [ inputs[best_seq_idx] ]
