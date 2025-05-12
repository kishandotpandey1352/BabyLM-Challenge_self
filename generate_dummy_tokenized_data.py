import torch
import pickle
import os
from torch.utils.data import TensorDataset, DataLoader

# Parameters
num_samples = 1000
seq_len = 32
vocab_size = 50256
batch_size = 2

# Generate dummy tokenized data
x = torch.randint(0, vocab_size, (num_samples, seq_len))
y = torch.randint(0, vocab_size, (num_samples, seq_len))

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=batch_size)

# Wrap like get_loaders() expects
data_dict = {
    'holdout_loader': loader,
    'holdout_val_loader': loader,
    'score_loader': loader
}

# Save to file
os.makedirs("tokenizers", exist_ok=True)
with open("tokenizers/1k_data_token.pkl", "wb") as f:
    pickle.dump(data_dict, f)

print("Dummy tokenized dataset saved as tokenizers/1k_data_token.pkl")
