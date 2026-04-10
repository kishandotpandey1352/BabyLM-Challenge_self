import os
import torch
from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast
from models.gpt import GPT2Model
from utils.configs import ModelConfig

# === Limit threads (safe on login or batch nodes) ===
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# === Load config and model ===
config = ModelConfig()
custom_model = GPT2Model(config)
custom_model.load_state_dict(torch.load("trained_models/main_baseline_10Mmodel_weights.pt"))
custom_model.eval()

# === Build Hugging Face-compatible config ===
hf_config = GPT2Config(
    vocab_size=config.vocab_size,
    n_embd=config.n_embd,
    n_layer=config.n_layers,
    n_head=config.n_head,
    n_positions=config.block_size,
    bos_token_id=0,
    eos_token_id=1
)

hf_model = GPT2LMHeadModel(hf_config)

# === Transfer weights manually ===
with torch.no_grad():
    hf_model.transformer.wte.weight.copy_(custom_model.token_embd.weight)
    hf_model.transformer.wpe.weight.copy_(custom_model.posn_embd.weight)

    for i in range(config.n_layers):
        custom_block = custom_model.blocks[i]
        custom_state = custom_block.state_dict()
        mapped_state = {}

        for key in custom_state:
            new_key = key
            param = custom_state[key]

            # Rename keys to Hugging Face format
            new_key = new_key.replace("ln1", "ln_1")
            new_key = new_key.replace("ln2", "ln_2")

            if "c_attn" in new_key:
                new_key = "attn." + new_key
            elif "c_proj" in new_key:
                new_key = "attn." + new_key

            new_key = new_key.replace("mlp.0", "mlp.c_fc")
            new_key = new_key.replace("mlp.2", "mlp.c_proj")

            # Transpose if it's a weight matrix
            if "weight" in new_key and param.ndim == 2:
                param = param.T

            mapped_state[new_key] = param

        hf_model.transformer.h[i].load_state_dict(mapped_state)

    # Skip loading ln_f and lm_head since not present in your model
    # hf_model.transformer.ln_f.load_state_dict(...)
    # hf_model.lm_head.load_state_dict(...)

# === Save model and tokenizer
hf_model.save_pretrained("hf_main_baseline_10M")

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.model_max_length = config.block_size
tokenizer.save_pretrained("hf_main_baseline_10M")

print("✅ Export complete: model saved to hf_main_baseline_10M/")
