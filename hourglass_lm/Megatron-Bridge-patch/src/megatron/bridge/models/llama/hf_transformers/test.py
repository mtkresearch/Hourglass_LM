from modeling_llama import LlamaForCausalLM, LlamaModel
from configuration_llama import LlamaConfig

config = LlamaConfig()
config.num_mlp_layers = 2
config.mlp_intermediate_sizes = [11008, 11008]

model = LlamaModel(config)
print(model)