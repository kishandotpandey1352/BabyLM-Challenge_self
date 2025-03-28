from transformers import GPT2Tokenizer
import torch
import torch.nn as nn
import torch.optim as optim

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load and tokenize a sample text file
with open('simple_wiki.train', 'r') as f:
    text = f.read()

tokens = tokenizer.encode(text, return_tensors='pt')

# Define proxy GPT model (small)
class SmallGPT(nn.Module):
    def __init__(self, vocab_size, emb_size=128, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.transformer_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=4), 
            num_layers=n_layers
        )
        self.fc_out = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out = self.transformer_blocks(emb)
        return self.fc_out(out)

model = SmallGPT(vocab_size=tokenizer.vocab_size)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Train proxy model and track per-example loss
seq_len = 128
losses = []
for i in range(0, tokens.size(1) - seq_len - 1):
    input_seq = tokens[:, i:i+seq_len]
    target_seq = tokens[:, i+1:i+seq_len+1]

    output = model(input_seq)
    loss = criterion(output.view(-1, tokenizer.vocab_size), target_seq.view(-1))
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
