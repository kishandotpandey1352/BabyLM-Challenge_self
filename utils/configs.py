from dataclasses import dataclass
import torch 

# ==== Configuration ==== #
@dataclass
class ModelConfig:
    def __init__(self):
        self.block_size = 32 # context size (min num of tokens in sequence, how long model can see at once)
        self.vocab_size = 50256
        self.max_iters = 100
        self.n_head = 4
        self.n_embd = 64
        self.n_layers = 6
        self.dropout = 0.1
        self.bias: bool = True
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
                                          # switch out when transferring to github or stanage for cuda use

@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 32
    max_len: int = 64
    pad: int = 0
    grad_accum_steps: int = 1
    max_lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_iters: int = 100
    n_steps: int = 0
    lr_decay_iters: int = 1000
    save_checkpoints: bool = False
    save_every: int = 1
    save_every_steps:int = 2000

@dataclass
class ProxyConfig:
    epochs:int = 5
    vocab_size:int = ModelConfig().vocab_size    # keep same vocab
    block_size:int = ModelConfig().block_size    # same context length
    batch_size:int = 32
    n_embd:int = 128                        # e.g. 1/4 or 1/8 of main
    n_head :int = 4                          # just enough heads
    n_layers:int = 2                          # really small depth
    dropout:float = 0.1
    bias: bool = False
    min_lr: float = 1e-4
    max_lr: float = 3e-4
    device  = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Proxy steps
    T_steps:int = 1_00
    t0:int = 5_0
    step:int = 0

    alpha_scale:float = 0.7 # keep [0,1] range
