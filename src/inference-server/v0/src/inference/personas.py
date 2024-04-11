from . import log

import json, os, re, yaml

def _clean_spaces(str):
    new_str = re.sub(r"\n+", "\n", re.sub(r"[ \t]+", " ", str.strip()))
    return re.sub(r"\n ", "\n", new_str)

def _gdoc_request(gdoc_id):
    try:
        response = requests.get(f"https://docs.google.com/document/d/{gdoc_id}/export?format=txt")
        if response.status_code == 200:
            return response.content
        else:
            log.error(f"  request error: {gdoc_id} {response.status_code}")
    except Exception as e:
        log.error(f"  request error: {gdoc_id} {e}")
    return None


class ChatModelTokens():
    def __init__(self, user, model):
        self.user = user
        self.model = model
    def __str__(self):
        return f"ChatModelTokens(user: {self.user}, model: {self.model})"


class ChatModelTranslator():
    def to_model(persona):
        if persona.model_type == "llama2":
            ChatModelTranslator.to_llama2(persona)
        elif persona.model_type == "gptj":
            ChatModelTranslator.to_gptj(persona)
        elif persona.model_type != None:
            raise Exception(f"unknown model_type {persona.model_type}")

    def to_llama2(persona):
        new_history = re.sub(r"{{user}}", "<|user|>", persona.hidden_history)
        new_history = re.sub(r"{{model}}", "<|model|>", new_history)
        del persona.hidden_history

        persona.description = _clean_spaces(f"""
        <|system|>Enter RP mode. Pretend to be {{{{{persona.name}}}}} whose persona follows:
        {persona.description}
        You shall reply to the user while staying in character, and generate long responses.
        {new_history}
        """)
        #print(persona)

    def to_gptj(persona):
        new_history = re.sub(r"{{user}}", "You: ", persona.hidden_history)
        new_history = re.sub(r"{{model}}", f"{persona.name}: ", new_history)
        del persona.hidden_history
    
        persona.description = _clean_spaces(f"""
        {persona.name}'s Persona: {persona.description}
        <START>
        {new_history}
        """)
        #print(persona)

    def get_chat_tokens(persona):
        if persona.model_type == "llama2":
            return ChatModelTokens("<|user|>", "<|model|>")
        elif persona.model_type == "gptj":
            return ChatModelTokens("You:", f"{persona.name}:")
        elif persona.model_type != None:
            raise Exception(f"unknown model_type {model_type}")

    def make_safe_input(message):
        return message

    def build_prompt(message, chat_history, persona):
        message = ChatModelTranslator.make_safe_input(message)
        preamble = f"{persona.description}\n{chat_history}\n"
        prompt = f"{persona.tokens.user} {message}\n{persona.tokens.model}"
        full_prompt = f"{preamble}{prompt}"
        return full_prompt, prompt

    def get_single_reply(text):
        # remove additional prompt/reply generation
        index = text.find("{user}:")
        if index != -1:
            text = text[:index]
        index = text.find("You:")
        if index != -1:
            text = text[:index]

        # remove incomplete generations (due to max length)
        end_phrase = {".", "!", "?", "*"}
        index = max(text.rfind(char) for char in end_phrase)
        if index != -1:
            text = text[:index + 1]
        return text.strip()

    def get_reply(gen_text, builded_prompt, persona):
        # remove prompt
        text = gen_text[len(builded_prompt):]
        text = ChatModelTranslator.get_single_reply(text)
        
        # token fixes
        text = text.replace("<START>", "")
        text = text.replace(", <USER>.", ".")
        text = text.replace(", <USER>!", "!")
        text = text.replace(", <USER>?", "?")
        text = text.replace("<USER>", "you")
        text = text.replace("<BOT>:", f"{persona.name}:")
        # other fixes
        text = text.replace("the user", "you")
        text = text.replace("the player", "you")
        text = text.replace("the man", "you")
        text = text.replace("the woman", "you")
        return text


class ChatPersona():
    def __init__(self, yaml_data, model_type):
        self.model_type = model_type
        self.name = yaml_data["name"]
        self.language = yaml_data["language"]
        self.version = yaml_data["version"]
        self.description = yaml_data["description"]
        self.hidden_history = yaml_data["hidden_history"]
        self.first_message = yaml_data["first_message"]
        self.tokens = ChatModelTranslator.get_chat_tokens(self)
        ChatModelTranslator.to_model(self)

    @staticmethod
    def from_yaml(yaml_data, model_type):
        yaml_object = yaml.safe_load(yaml_data)
        return ChatPersona(yaml_object, model_type)

    def __str__(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
        """
        attributes = []
        for attr, value in vars(self).items():
            if isinstance(value, str):
                if len(value) > 64:
                    value = f"{value[:64]}..."
                value = f"'{value}'"
            attributes.append(f"{attr}={value}")
        attributes_str = ", ".join(attributes)
        return f"ChatPersona({attributes_str})"
        """

    def build_prompt(self, input_message, chat_history):
        return ChatModelTranslator.build_prompt(input_message, chat_history, self)

    def get_reply(self, gen_text, builded_prompt):
        return ChatModelTranslator.get_reply(gen_text, builded_prompt, self)


class ChatPersonas():
    gdoc_personas = {
        "custom1": "1tRxEPX-b5nInMVdsr7hkIGrQ4h8Jm-x1-97yZx6dzwQ",
        "custom2": "18HHK9UrT-OoezSULAJjil8fzWvs8vu_SS2xq5yVyqtA",
    }

    def __init__(self, personas, model_type):
        self.personas = personas
        self.model_type = model_type
    
    def __str__(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
        """
        personas_str = ", ".join([f"'{key}': {str(value)}" for key, value in self.personas.items()])
        return f"ChatPersonas(personas: {{{personas_str}}}, model_type: {self.model_type})"
        #return f"ChatPersonas(\n{_str_format(self)})"
        """
    
    @staticmethod
    def from_folder(folder=None, model_type=None):
        personas = {}
        base_folder = folder or ""
        yaml_files = [file for file in os.listdir(folder) if file.endswith(".yaml")]
        yaml_files = [os.path.join(base_folder, file) for file in yaml_files]

        for yaml_file in yaml_files:
            try:
                with open(yaml_file, "r") as yaml_data:
                    key = os.path.splitext(os.path.basename(yaml_file))[0]
                    personas[key] = ChatPersona.from_yaml(yaml_data, model_type)
            except Exception as e:
                print(f"  error reading: {yaml_file} {e}")
        if not personas:
            print("  error no persona found!")

        return ChatPersonas(personas, model_type)
    
    def get(self, persona_id):
        # always hot-reload from google docs
        if persona_id in ChatPersonas.gdoc_personas:
            gdoc_id = ChatPersonas.gdoc_personas[persona_id]
            gdoc_yaml_data = _gdoc_request(gdoc_id)
            return ChatPersona.from_yaml(gdoc_yaml_data, self.model_type)

        return self.personas[persona_id]
