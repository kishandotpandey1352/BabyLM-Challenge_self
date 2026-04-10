import pickle 
import numpy as np
from sklearn.model_selection import train_test_split 
from utils.configs import TrainConfig, ModelConfig
import torch
from torch.utils.data import DataLoader, TensorDataset

class LazyTokenDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.block_size]
        y = self.tokens[idx + 1:idx + 1 + self.block_size]
        assert len(x) == self.block_size and len(y) == self.block_size, f"Truncated sequence: x={len(x)}, y={len(y)}, block_size={self.block_size}"
        return x, y

def get_loaders(path, n_tokens, split_type='tune'):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    tokens = [token for sublist in data for token in sublist]
    if n_tokens is not None:
        tokens = tokens[:n_tokens]

    tokens_tensor = torch.tensor(tokens, dtype=torch.long)

    model_configs = ModelConfig()
    train_configs = TrainConfig()

    block_size = model_configs.block_size
    batch_size = train_configs.batch_size

    dataset = LazyTokenDataset(tokens_tensor, block_size)
    total_len = len(dataset)

    if split_type == 'tune':
        train_main_size = int(0.64 * total_len)
        val_size  = int(0.16 * total_len)
        hold_size = int(0.16 * total_len)
        hold_val_size = total_len - (train_main_size + val_size + hold_size)

        splits = [train_main_size, val_size, hold_size, hold_val_size]

        train_dataset, val_dataset, hold_dataset, holdout_val_dataset = torch.utils.data.random_split(
            dataset, splits, generator=torch.Generator().manual_seed(88)
        )

        return {
            'train_loader': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True),
            'val_loader': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1, pin_memory=True),
            'holdout_loader': DataLoader(hold_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True),
            'holdout_val_loader': DataLoader(holdout_val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1, pin_memory=True),
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'hold_dataset': hold_dataset,
            'holdout_val_dataset': holdout_val_dataset,
            'score_loader': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True),
            'score_dataset': train_dataset
        }

    elif split_type == 'final':
        val_size = int(0.05 * total_len)
        train_size = total_len - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(88)
        )

        return {
            'train_loader': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True),
            'val_loader': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1, pin_memory=True),
            'train_dataset': train_dataset,
            'val_dataset': val_dataset
        }

    else:
        raise ValueError(f"Unknown split_mode: {split_type}")
