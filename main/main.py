from scheduler import Scheduler
from toy_data import ToySequenceDataset

import torch
import torch.nn as nn
from scipy.stats import entropy
import numpy as np

if __name__ == "__main__":

    batch_size = 4
    seq_len = 10
    n_embd = 32
    num_heads = 4

    mha = MultiHeadAttention(d_model=n_embd, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, n_embd)

    out = mha(x, x, x)  # Q=K=V for self-attention
    print(out.shape)  # should be (batch_size, seq_len, d_model)
