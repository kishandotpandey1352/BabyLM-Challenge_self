from dataclasses import dataclass
import torch 

# ==== Configuration ==== #
@dataclass
class ModelConfig:
    # def __init__(self):
    #     self.block_size = 1024
    #     self.vocab_size = 50256
    #     self.max_iters = 100
    #     self.n_head = 12
    #     self.n_embd = 768
    #     self.n_layers = 12
    #     self.dropout = 0.1
    #     self.bias: bool = True
    #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self):
        self.block_size = 64                
        self.vocab_size = 50256
        self.max_iters = 10
        self.n_head = 2
        self.n_embd = 64
        self.n_layers = 2
        self.dropout = 0.1
        self.bias: bool = True
        self.device = torch.device("cuda")

@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    max_len: int = 1024
    pad: int = 0
    grad_accum_steps: int = 2
    max_lr: float = 0.001739
    min_lr: float = 3.35329e-05
    warmup_iters: int = 1000
    n_steps: int = 0
    lr_decay_iters: int = 1000
    save_checkpoints: bool = False
    save_every: int = 1
    save_every_steps: int = 500

    alpha_update_steps:int = 500
    gamma: float = 0.3 # [non zero]
    total_steps:int = 1

@dataclass
# class ProxyConfig:
#     epochs: int = 15
#     vocab_size: int = ModelConfig().vocab_size
#     block_size: int = ModelConfig().block_size
#     batch_size: int = 16
#     n_embd: int = 768 // 4
#     n_head: int = 4
#     n_layers: int = 4
#     dropout: float = 0.15
#     bias: bool = False
#     min_lr: float = 0.0001388403383951373
#     max_lr: float = 0.0005018765292380106
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     T_steps: int = 5000
#     t0: int = 100
#     step: int = 0
#     alpha_scale: float = 0.5
class ProxyConfig:
    epochs: int = 1
    vocab_size: int = 50256
    block_size: int = 32          # very short input sequences
    batch_size: int = 2           # tiny batches
    n_embd: int = 32              # small embedding size
    n_head: int = 2               # low attention head count
    n_layers: int = 2             # shallow model
    dropout: float = 0.1
    bias: bool = False
    min_lr: float = 1e-5
    max_lr: float = 5e-4
    device = torch.device("cuda")
    T_steps: int = 10             # very short test run
    t0: int = 5                   # early checkpoint
    step: int = 0
    alpha_scale: float = 0.5