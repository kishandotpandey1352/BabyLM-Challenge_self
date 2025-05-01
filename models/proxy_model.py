import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
from gpt import GPT2Model
from configs import ProxyConfig, TrainConfig

class ProxyTrain(torch.nn.Module):
    def __init__(self, holdout_loader: DataLoader, score_loader: DataLoader, configs, model_cls):
        super().__init__()
        # Loader for training proxy
        self.holdout_loader = holdout_loader
        # Loader for computing learnability on unseen data
        self.score_loader = score_loader
        self.configs = configs
        self.device = configs.device

        # Build and move proxy model
        self.train_model = model_cls(configs).to(self.device)
        self.optim = torch.optim.AdamW(self.train_model.parameters(), lr=configs.max_lr)

        # Loss for training uses default (mean)
        self.Loss = torch.nn.CrossEntropyLoss()
        # Model class for fresh instances
        self.model_cls   = model_cls

    def train(self):
        hold_iter = iter(self.holdout_loader)
        for step in trange(1, self.configs.T_steps + 1, desc="Proxy training"):
            try:
                x, y = next(hold_iter)
            except StopIteration:
                hold_iter = iter(self.holdout_loader)
                x, y = next(hold_iter)

            self.train_model.train()
            x, y = x.to(self.device), y.to(self.device)
            logits = self.train_model(x)  # [B, L, V]
            B, L, V = logits.shape

            loss = self.Loss(logits.view(B * L, V), y.view(-1))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if step % 1000 == 0:
                print(f"Proxy step {step}/{self.configs.T_steps}, loss={loss.item():.4f}")

            if step == self.configs.t0:
                torch.save(self.train_model.state_dict(), "proxy_early.pt")
                print(f"Saved proxy early checkpoint at step {step}")

        # final checkpoint
        torch.save(self.train_model.state_dict(), "proxy_late.pt")
        print(f"Saved proxy final checkpoint at step {self.configs.T_steps}")


    def reductionLoss(self, model, inputs, targets):
        logits = model(inputs) # [B, L, V]
        B, L, V = logits.shape
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fn(logits.view(B * L, V), targets.view(-1))  # [B*L]
        seq_loss = token_loss.view(B, L).mean(dim=1) # [B]
        return seq_loss

    def LearnabilityScore(self):
        # Load early proxy
        early = self.model_cls(self.configs).to(self.device)
        early.load_state_dict(torch.load("proxy_early.pt", weights_only=True))
        early.eval()
        # Load late proxy
        late = self.model_cls(self.configs).to(self.device)
        late.load_state_dict(torch.load("proxy_late.pt", weights_only=True))
        late.eval()

        all_deltas = []
        with torch.no_grad():
            for x, y in tqdm(self.score_loader):
                x, y = x.to(self.device), y.to(self.device)
                loss_early = self.reductionLoss(early, x, y)
                loss_late  = self.reductionLoss(late,  x, y)
                delta = loss_early - loss_late
                all_deltas.append(delta)

        # return flat tensor of learnability scores
        return torch.cat(all_deltas, dim=0)
