import pickle 
import numpy as np
from sklearn.model_selection import train_test_split 
from utils.configs import TrainConfig, ModelConfig
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_loaders(path, n_tokens):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    tokens = [token for sublist in data for token in sublist] # flatten tokens
    if n_tokens is not None:
        tokens = tokens[:n_tokens]
    else:
        tokens = tokens[:len(tokens)]

    tokens_np = np.array(tokens, dtype=np.int32) # too much precision?

    model_configs = ModelConfig()
    train_configs = TrainConfig()

    block_size = model_configs.block_size
    batch_size = train_configs.batch_size

    X = np.lib.stride_tricks.sliding_window_view(tokens_np, block_size)[:-1]
    Y = np.lib.stride_tricks.sliding_window_view(tokens_np[1:], block_size)

    # Split into train + val (80%) and proxy-holdout (20%)
    X_train_val, X_hold, y_train_val, y_hold = train_test_split(
    X, Y, 
    test_size=0.20, 
    random_state=88
    )

    # From the train+val set, create validation set (20% of 80% = 16% total)
    X_train_main, X_val, y_train_main, y_val = train_test_split(
    X_train_val, 
    y_train_val, 
    test_size=0.20, 
    random_state=88
    )
    X_hold, X_val_hold, y_hold, y_val_hold = train_test_split(
        X_hold, y_hold,
        test_size=0.1, 
        random_state=88
    )

    # Tensors
    X_train_tensor = torch.tensor(X_train_main, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train_main, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    X_hold_tensor = torch.tensor(X_hold, dtype=torch.long)
    y_hold_tensor = torch.tensor(y_hold, dtype=torch.long)

    X_hold_val_tensor = torch.tensor(X_val_hold, dtype=torch.long)
    y_hold_val_tensor = torch.tensor(y_val_hold, dtype=torch.long)

    # datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    hold_dataset = TensorDataset(X_hold_tensor, y_hold_tensor) # for proxy model
    hold_val_dataset = TensorDataset(X_hold_val_tensor, y_hold_val_tensor)
    score_dataset = TensorDataset(X_train_tensor, y_train_tensor) # for irreducible 

    # loaders 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    holdout_loader = DataLoader(hold_dataset, batch_size=batch_size, shuffle=True)
    holdout_val_loader = DataLoader(hold_val_dataset, batch_size=batch_size, shuffle=True)
    score_loader = DataLoader(score_dataset, batch_size=batch_size, shuffle=True)

    return {
        'X_train_tensor': X_train_tensor,
        'y_train_tensor':y_train_tensor,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'holdout_loader': holdout_loader,
        'holdout_val_loader': holdout_val_loader,
        'score_loader': score_loader,
        'train_dataset':train_dataset,
        'val_dataset': val_dataset,
        'hold_dataset':hold_dataset,
        'holdout_val_dataset':hold_val_dataset,
        'score_dataset': score_dataset
    }
    


    

