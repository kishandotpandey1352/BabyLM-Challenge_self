"""
Main pipeline for training GPT2 using curriculum learning with proxy models and entropy-guided scheduling.
Accepts tokenized data (10M or 100M) and trains a GPT2 model with learnability-aware scheduling.
"""

import os
import argparse
import pickle
import random
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from proxy_model import ProxyTrain
from gpt import GPTTrainer, GPT2Model
from scheduler import Scheduler

# ------------------------ Config Classes ------------------------ #

class ModelConfig:
    def __init__(self):
        self.block_size = 32
        self.vocab_size = 50256
        self.max_iters = 100
        self.n_head = 4
        self.n_embd = 64
        self.n_layers = 6
        self.dropout = 0.1
        self.bias = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainConfig:
    def __init__(self):
        self.epochs = 10
        self.batch_size = 32
        self.max_len = 64
        self.max_lr = 3e-4
        self.grad_accum_steps = 1
        self.warmup_iters = 100
        self.save_every_steps = 5000
        self.save_every = 1
        self.T_steps = 1000
        self.t0 = 200
        self.total_steps = 0  # To be updated later

# ------------------------ Utility Functions ------------------------ #

def set_seed(seed=88):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_pickle_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def split_data(data, split_ratio=0.8):
    split_idx = int(len(data) * split_ratio)
    train = data[:split_idx]
    test = data[split_idx:]
    return train, test

def build_loader(data, batch_size):
    x = [torch.tensor(seq[:-1]) for seq in data]
    y = [torch.tensor(seq[1:]) for seq in data]
    x_padded = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    y_padded = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)
    dataset = TensorDataset(x_padded, y_padded)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ------------------------ Main Pipeline ------------------------ #

def main(args):
    set_seed()
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    def log(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

    # === Load Tokenized Data === #
    data = load_pickle_data(args.data_path)
    log(f"Loaded {len(data)} sequences from {args.data_path}")

    holdout_data, score_data = split_data(data, 0.5)
    holdout_loader = build_loader(holdout_data, args.batch_size)
    score_loader = build_loader(score_data, args.batch_size)

    # === Proxy Model Phase === #
    log("Starting proxy model training...")
    model_config = ModelConfig()
    train_config = TrainConfig()
    proxy = ProxyTrain(holdout_loader, score_loader, train_config, GPT2Model)
    proxy.train()

    log("Calculating learnability scores...")
    learnability = proxy.LearnabilityScore()
    log(f"Computed {len(learnability)} learnability scores")

    # Plot histogram
    plt.hist(learnability.cpu().numpy(), bins=100)
    plt.title("Learnability Score Distribution")
    plt.xlabel("Delta Loss")
    plt.ylabel("Count")
    plt.savefig(os.path.join(args.output_dir, "learnability_histogram.png"))
    plt.close()

    # === Main Training Phase === #
    log("Initializing Scheduler and GPTTrainer...")
    scheduler = Scheduler(score_loader, learnability, args.batch_size)
    val_loader = build_loader(score_data[:len(score_data)//10], args.batch_size)
    trainer = GPTTrainer(score_data, score_data, scheduler.seqentialBatch(0, 1.0), val_loader, train_config, model_config, scheduler, check_points_dir=args.output_dir)

    log("Starting GPT2 training...")
    train_losses, val_losses, alphas, lambdas = trainer.train()

    # Plot losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(args.output_dir, "loss_plot.png"))
    plt.close()

    log("Training completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train GPT2 with curriculum learning on tokenized data")
    parser.add_argument('--data_path', type=str, required=True, help='Path to tokenized .pkl file (10M or 100M)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to store checkpoints and plots')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for all loaders')
    args = parser.parse_args()
    main(args)