from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTJForCausalLM

print( f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Starting')

#device = torch.device("mps")
#device = "mps"
#modelsUrl = 'EleutherAI/gpt-j-6B';
#modelsUrl = 'hakurei/lit-6B';
modelsUrl = 'PygmalionAI/pygmalion-6b';

print( f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Loading models')

tokenizer = AutoTokenizer.from_pretrained(modelsUrl, device_map="auto")

model = AutoModelForCausalLM.from_pretrained(modelsUrl,
                                        device_map="auto",
                                        offload_folder="offload",
                                        #revision="float16",
                                        torch_dtype=torch.float16,
                                        #torch_dtype=torch.bfloat16,
                                        #low_cpu_mem_usage=True
                                        )

# convert to BetterTransformer
# https://huggingface.co/docs/transformers/perf_infer_gpu_many
model = model.to_bettertransformer()

def keepOneLineAfterIndex(text, pos):
  index = text.find("\n", pos)
  if index != -1:
    return text[0 : index +1]
  else:
    return f'{text}.\n'

def getLastReply(text, replyStartIndex, printReply):
    #reply = text[replyStartIndex + 1:]
    reply = text[replyStartIndex:]

    # remove some tokens
    reply = reply.replace("<START>", "")
    reply = reply.replace(", <USER>.", ".")
    reply = reply.replace(", <USER>!", "!")
    reply = reply.replace(", <USER>?", "?")
    # fail safe
    reply = reply.replace("<USER>", "bud")

    index = reply.find("{user}:")
    if index != -1:
        reply = reply[:index]
    index = reply.find("{{user}}:")
    if index != -1:
        reply = reply[:index]
    index = reply.find("You:")
    if index != -1:
        reply = reply[:index]

    # remove any garbage on end of string
    end_phrase = {'.', '!', '?', '*'}
    index = max(reply.rfind(char) for char in end_phrase)
    if index != -1:
        reply = reply[:index+1]

    # remove all breaklines, leading and trailing spaces
    reply = reply.replace("\n", " ").strip()

    if (printReply):
        print(reply)

    return reply

def model_generate(prompt, printPerf=True):
    #print("<START>" + prompt + "<END>\n\n")

    if (printPerf):
        print( f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Token Encoding')

    tokenizer.pad_token = tokenizer.eos_token
    tokensOut = tokenizer(prompt, return_tensors="pt") #pt == pytorch

    #input_ids = tokensOut.input_ids.to("mps")
    #attention_mask = tokensOut.attention_mask.to('mps')
    input_ids = tokensOut.input_ids.to("cuda")
    attention_mask = tokensOut.attention_mask.to('cuda')

    if (printPerf):
        print( f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Model Generation')

    gen_tokens = model.generate(input_ids,
                                do_sample=True, # For Quality & Perf, must be true
                                temperature=0.8, # higher == more random/non-sense

                                min_length=len(input_ids[0]) + 8,
                                max_length=len(input_ids[0]) + 96,

                                repetition_penalty=1.0,
                                top_k=50,
                                top_p=0.95,
                                #top_k=30,
                                #top_p=1.0,

                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                attention_mask=attention_mask)

    if (printPerf):
        print( f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Token Decoding')

    gen_text = tokenizer.batch_decode(gen_tokens, 
        skip_special_tokens=True
        )

    if (printPerf):
        print( f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Done')

    print(f'\n\n<START>{gen_text[0]}<END>\n\n')
    return gen_text[0]


# Persona 1
file = open('yuki_persona1.txt', 'r') # need to add bot-name, greeting (or scenario), 
file_contents = file.read()
file.close()

# Prompt
chat_log = file_contents
charLongName = 'Yuki Nagato'
charShortName = 'Yuki'

# ---- 
userPrompt = f"hey I'm new here, I'm needing a girlfriend."
print(f'\nPrompt: {userPrompt}\n')


while (True):
    #chat_log = f'{chat_log}\nYou: *{userPrompt}*\n{charLongName}: '
    chat_log = f'{chat_log}\nYou: *{userPrompt}*\n'

    temp = model_generate(chat_log).replace('<BOT>:', f'{charLongName}:').replace('<BOT>', f'{charShortName}')
    
    reply = getLastReply(temp, len(chat_log), True)
    chat_log += reply

    userPrompt = input('\n\nPrompt: ')
    print('')
