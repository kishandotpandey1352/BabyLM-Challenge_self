from dataclasses import dataclass
import torch 

# ==== Configuration ==== #
@dataclass
class ModelConfig:
    def __init__(self):
        self.block_size = 32
        self.vocab_size = 50256
        self.max_iters = 100
        self.n_head = 6
        self.n_embd = 32
        self.n_layers = 4
        self.dropout = 0.0079737
        self.bias: bool = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 32
    max_len: int = 64
    pad: int = 0
    grad_accum_steps: int = 2
    max_lr: float = 3.35329e-05
    min_lr: float = 0.001739
    warmup_iters: int = 20
    n_steps: int = 0
    lr_decay_iters: int = 1000
    save_checkpoints: bool = False
    save_every: int = 1
    save_every_steps: int = 500

    alpha_update_steps:int = 500
    gamma: float = 0.3 # [non zero]
    total_steps:int = 1

@dataclass
class ProxyConfig:
    epochs: int = 5
    vocab_size: int = ModelConfig().vocab_size
    block_size: int = ModelConfig().block_size
    batch_size: int = 16
    n_embd: int = 64 // 4
    n_head: int = 2
    n_layers: int = 2
    dropout: float = 0.15
    bias: bool = False
    min_lr: float = 0.0001388403383951373
    max_lr: float = 0.0005018765292380106
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T_steps: int = 250
    t0: int = 25
    step: int = 0
    alpha_scale: float = 0.5

