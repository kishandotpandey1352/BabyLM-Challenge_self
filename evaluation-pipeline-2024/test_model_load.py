import torch
from models.gpt import GPT2Model
from utils.configs import ModelConfig

print("[DEBUG] Loading config and model...")
config = ModelConfig()
model = GPT2Model(config)

print("[DEBUG] Loading state_dict...")
state_dict = torch.load("trained_models/main_baseline_10Mmodel_weights.pt", map_location="cpu")

print("[DEBUG] Applying weights...")
model.load_state_dict(state_dict)

print("[DEBUG] Running dummy input...")
dummy = torch.randint(0, config.vocab_size, (1, config.block_size))
output = model(dummy)
print("✅ Model forward passed! Output shape:", output.shape)
