from dataclasses import dataclass
import torch 

# ==== Configuration ==== #
@dataclass
class ModelConfig:
        block_size = 1024
        vocab_size = 50258
        max_iters = 100
        n_head = 12
        n_embd = 768
        n_layers = 12
        dropout = 0.1
        bias: bool = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 32
    max_len: int = 1024
    pad: int = 0
    grad_accum_steps: int = 2
    max_lr: float = 1e-3
    min_lr: float = 5e-4
    warmup_iters: int = 1000
    n_steps: int = 0
    lr_decay_iters: int = 1000
    save_checkpoints: bool = False
    save_every: int = 1
    save_every_steps: int = 1000

    alpha_update_steps:int = 500
    gamma: float = 0.3 # [non zero]
    total_steps:int = 1


@dataclass
class ProxyConfig:
    epochs: int = 10
    vocab_size: int = ModelConfig().vocab_size
    block_size: int = 1024
    batch_size: int = 32
    n_embd: int = 256
    n_head: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    bias: bool = False
    min_lr: float = 0.000255
    max_lr: float =  0.00092
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T_steps: int = 1000
    t0: int = 250
    step: int = 0
    alpha_scale: float = 0.73

