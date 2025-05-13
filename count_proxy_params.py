import torch
from models.gpt import GPT2Model
from utils.configs import ProxyConfig

# Load config and model
config = ProxyConfig()
model = GPT2Model(config)

# Load weights
model.load_state_dict(torch.load("trained_models/proxy_model.pt", map_location="cpu"))

# Count parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Proxy model has {num_params:,} trainable parameters ({num_params / 1e6:.2f}M)")
