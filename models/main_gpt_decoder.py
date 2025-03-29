# Build 2-4 layer GPT2 Decoder 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import numpy as np

# ==== CONFIG ==== 
class Config:
    def __init__(self, 
                vocab_size = 5000,
                batch_size = 64, # how many sequences independently processed
                block_size = 100, # maximum context lenght for prediciton
                device = 'cuda' if torch.cuda.is_available() else 'cpu',
                n_embd = 384,
                n_heads = 6,
                n_layer = 6,
                n_state = 512
                ):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_heads
        self.n_layer = n_layer
        self.device = device
        self.n_state = n_state

# ==== ATTENTION ====

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_heads, block_size):
        super(MultiHeadAttention, self).__init__()


    #def perHeadAttention(self, x):




# ==== GPT2 ====

#class GPT2(nn.Module):
    #def __init__(self, config):


if __name__ == "__main__":

    batch_size = 4
    seq_len = 10
    n_embd = 32
    num_heads = 4

    mha = MultiHeadAttention(d_model=n_embd, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, n_embd)

    out = mha(x, x, x)  # Q=K=V for self-attention
    print(out.shape)  # should be (batch_size, seq_len, d_model)
