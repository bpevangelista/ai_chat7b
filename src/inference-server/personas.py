import os

import requests
import yaml

from log import log


class ChatPersona:
    def __init__(self,
                 uuid: str,
                 version: str,
                 language: str,
                 name: str,
                 # Search
                 gender: str,
                 tags: list[str],
                 # Prologue and First Message are shown in sequence
                 prologue: str,
                 first_message: str,
                 # Persona
                 hidden_description: str,
                 hidden_dialogue_style: str, # Flattened
                 ):
        self.uuid = uuid
        self.version = version
        self.language = language
        self.name = name

        self.gender = gender
        self.tags = tags

        self.prologue = prologue
        self.first_message = first_message
        self.hidden_description = hidden_description
        self.hidden_dialogue_style = hidden_dialogue_style


class ChatPersonas:
    def __init__(self):
        self.personas: dict[str, ChatPersona] = {}

    def load_from_gdoc(self, persona_id: str, gdoc_id: str):
        try:
            response = requests.get(f"https://docs.google.com/document/d/{gdoc_id}/export?format=txt")
            if response.status_code == 200:
                persona_dict = yaml.safe_load(response.content)
                new_personas = {
                    persona_id: ChatPersona(**persona_dict)
                }
                self.personas.update(new_personas)
                return new_personas
            else:
                log.error(f"  request error: {gdoc_id} {response.status_code}")
        except Exception as e:
            log.error(f"  request error: {gdoc_id} {e}")
        return None

    def load_all_from_folder(self, base_path_uri: str):
        yaml_files = [file for file in os.listdir(base_path_uri) if file.endswith(".yaml")]
        yaml_files = [os.path.join(base_path_uri, file) for file in yaml_files]

        new_personas: dict[str, ChatPersona] = {}
        for yaml_file in yaml_files:
            try:
                with (open(yaml_file, "rt") as yaml_data):
                    key = os.path.splitext(os.path.basename(yaml_file))[0]
                    persona_dict = yaml.safe_load(yaml_data)
                    new_personas[key] = ChatPersona(**persona_dict)
            except Exception as e:
                print(f"  error reading: {yaml_file} {e}")
        if not new_personas:
            print("  error no persona found!")
            return None

        self.personas.update(new_personas)
        return new_personas

    def _make_safe_prompt(self, prompt: str) -> str:
        return prompt

    def _make_safe_history(self, chat_history: list[str], max_history_words: int = 960) -> str:
        flatten_chat_history = ''
        total_word_count = 0

        reversed_chat = list(reversed(chat_history))
        for reply, prompt in zip(reversed_chat[::2], reversed_chat[1::2]):
            #prompt_reply = "\n".join([prompt, reply])
            word_count = len(prompt.split()) + len(reply.split())
            if word_count + total_word_count > max_history_words:
                break
            #flatten_chat_history = prompt_reply + flatten_chat_history
            #flatten_chat_history = f'### Instruction:\n{prompt}\n### Response:\n{reply}\n' + flatten_chat_history

            #flatten_chat_history = f'[INST] {prompt} [/INST]{reply}\n' + flatten_chat_history
            flatten_chat_history = f'\n{prompt} {reply}\n' + flatten_chat_history

            total_word_count = total_word_count + word_count

        if flatten_chat_history:
            flatten_chat_history = flatten_chat_history + '\n'

        return flatten_chat_history

    def get_first_message(self, persona_id: str):
        if persona_id in self.personas:
            persona = self.personas[persona_id]
            return persona.prologue, persona.first_message
        return '', ''

    def _get_prompt_preamble(self, persona_id: str):
        if persona_id in self.personas:
            persona = self.personas[persona_id]
            return f'{persona.hidden_description}\n{persona.hidden_dialogue_style}\n'
        return ''

    def build_prompt(self, prompt: str, chat_history: list[str], persona_id: str):
        preamble = self._get_prompt_preamble(persona_id)
        safe_prompt = self._make_safe_prompt(prompt)
        safe_history = self._make_safe_history(chat_history)
        return f'<s>[INST] <<SYS>>\n{preamble}{safe_history}\n<</SYS>>\n{safe_prompt}\n[/INST]'
        #return f'{preamble}\n{safe_history}### Instruction:\n{safe_prompt}\n### Response:\n'
        #return f'{preamble}{safe_history}### Instruction:\n{safe_prompt}\n### Response:\n'

