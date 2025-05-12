import torch
from torch.utils.data import TensorDataset, DataLoader
from models.proxy_model import ProxyTrain
from models.gpt import GPT2Model
from utils.configs import ProxyConfig

# --- Generate dummy data ---
batch_size = 2
seq_len = 32
vocab_size = 50256

x = torch.randint(0, vocab_size, (20, seq_len))  # 20 samples
y = torch.randint(0, vocab_size, (20, seq_len))

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=batch_size)

# --- Use minimal GPU-friendly ProxyConfig ---
config = ProxyConfig()
config.block_size = seq_len
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.T_steps = 10
config.t0 = 5

# --- Run the test ---
proxy = ProxyTrain(loader, loader, loader, config, GPT2Model)
proxy.train()
