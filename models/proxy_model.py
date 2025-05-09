import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
from models.gpt import GPT2Model
from utils.configs import ProxyConfig, TrainConfig
from torch.nn import functional as F


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
        self.model_cls = model_cls

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
                torch.save(self.train_model.state_dict(), f"proxy_early_{self.configs.block_size}.pt")
                print(f"Saved proxy early checkpoint at step {step}")

        # final checkpoint
        torch.save(self.train_model.state_dict(), f"proxy_late_{self.configs.block_size}.pt")
        print(f"Saved proxy final checkpoint at step {self.configs.T_steps}")
        


    def reductionLoss(self, model, inputs, targets):
        logits = model(inputs)         # [B, L, V]
        B, L, V = logits.shape
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fn(logits.view(B * L, V), targets.view(-1))  # [B*L]
        seq_loss = token_loss.view(B, L).mean(dim=1)                   # [B]
        return seq_loss
    
    def sequenceEntropy(self, model, inputs):
        logits = model(inputs)
        B, L, V = logits.shape
        token_probs = F.softmax(logits, dim=-1) # [B, L, V]
        log_probs = torch.log(token_probs + 1e-8)
        token_entropy = -torch.sum(token_probs * log_probs, dim=-1) # [B, L]
        seq_entropy = token_entropy.mean(dim=1) # [B]
        return seq_entropy


    def LearnabilityScore(self, type='composite'):

        # Load early proxy
        early = self.model_cls(self.configs).to(self.device)
        early.load_state_dict(torch.load(f"proxy_early_{self.configs.block_size}.pt", weights_only=True))
        early.eval()
        # Load late proxy
        late = self.model_cls(self.configs).to(self.device)
        late.load_state_dict(torch.load(f"proxy_late_{self.configs.block_size}.pt", weights_only=True))
        late.eval()

        all_deltas = []
        abs_entropys = []
        with torch.no_grad():
            for x, y in tqdm(self.score_loader):
                x, y = x.to(self.device), y.to(self.device)
                loss_early = self.reductionLoss(early, x, y)
                loss_late  = self.reductionLoss(late,  x, y)
                delta = loss_early - loss_late
                all_deltas.append(delta)

                entropy_late = self.sequenceEntropy(late, x)
                abs_entropys.append(entropy_late)
            
        irr_loss = torch.cat(all_deltas, dim=0)
        entropy = torch.cat(abs_entropys, dim=0)

        eps = 1e-8

        # normalise for composite score
        norm_irr = (irr_loss - torch.min(irr_loss)) / (torch.max(irr_loss) - torch.min(irr_loss) + eps)
        norm_entropy = (entropy - torch.min(entropy)) / (torch.max(entropy) - torch.min(entropy) + eps)

        # return flat tensor of learnability scores
        if type == 'Loss':
            return irr_loss
        if type == 'Entropy':
            return entropy
        elif type == 'composite':
            comp_score = self.configs.alpha_scale * norm_irr + (1 - self.configs.alpha_scale)*norm_entropy
            return comp_score

        
        
        
        
