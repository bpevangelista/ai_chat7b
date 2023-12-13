import transformers, torch
from transformers import GPTJModel, GPTJConfig


torch.set_default_dtype(torch.float16)

config = GPTJConfig()
model = GPTJModel(config)
print(model)
torch.save(model, "blah0.pt")

config.n_head = 1
config.n_layer = 1
config.rotary_dim = 1

model = GPTJModel(config)
print(model)
torch.save(model, "blah1.pt")

config.n_head = 16 # default
config.rotary_dim = 64 # default
model = GPTJModel(config)
print(model)
torch.save(model, "blah2.pt")

config.n_layer = 2
model = GPTJModel(config)
print(model)
torch.save(model, "blah3.pt")

