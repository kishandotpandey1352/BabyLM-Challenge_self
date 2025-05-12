import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
from models.gpt import GPT2Model
from utils.configs import ProxyConfig, TrainConfig
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler


class ProxyTrain(torch.nn.Module):
    def __init__(self, holdout_loader: DataLoader, holdout_val_loader: DataLoader, score_loader: DataLoader, configs, model_cls):
        super().__init__()
        # Loader for training proxy
        self.holdout_loader = holdout_loader
        # Loader for computing learnability on unseen data
        self.score_loader = score_loader
        self.val_loader = holdout_val_loader
        self.configs = configs
        self.device = configs.device

        # Build and move proxy model
        self.train_model = model_cls(configs).to(self.device)
        self.optim = torch.optim.AdamW(self.train_model.parameters(), lr=configs.max_lr)

        # Loss for training uses default (mean)
        self.Loss = torch.nn.CrossEntropyLoss(ignore_index=self.configs.pad if hasattr(self.configs, 'pad') else 0)
        # Model class for fresh instances
        self.model_cls = model_cls
        self.criterion = nn.CrossEntropyLoss()
        print(f"[DEBUG] Proxy model on: {next(self.train_model.parameters()).device}")


    def validate(self, val_loader):
        self.train_model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                print("max token ID:", x.max().item())
                print("vocab size:", self.configs.vocab_size)
                if torch.any(x >= self.configs.vocab_size):
                    print("ERROR: Found token ID >= vocab_size!")
                    print(f"Max token in batch: {x.max().item()}, Vocab size: {self.configs.vocab_size}")
                    exit(1)
                logits = self.train_model(x)
                b,t,v = logits.shape
                logits=logits.view(b*t, v)
                y = y.view(b*t)

                loss = self.criterion(logits, y)
                total_loss += loss.item()
        return total_loss /len(self.val_loader)

    def train(self):
        print("Model device:", next(self.train_model.parameters()).device)
        hold_iter = iter(self.holdout_loader)
        total_loss = 0
        log_every=100
        loss_arr = []
        val_arr = []

        scaler = GradScaler()
        
        for step in trange(1, self.configs.T_steps + 1, desc="Proxy training", miniters=1):
            try:
                x, y = next(hold_iter)
            except StopIteration:
                hold_iter = iter(self.holdout_loader)
                x, y = next(hold_iter)

            self.train_model.train()
            x, y = x.to(self.device), y.to(self.device)

            self.optim.zero_grad()
            with autocast():
                logits = self.train_model(x)  # [B, L, V]
                B, L, V = logits.shape
                loss = self.Loss(logits.view(B * L, V), y.view(-1))
                total_loss += loss.item()

            assert y.dtype==torch.long, f"y dtype must be long tensor, got {y.dtype}"
            assert y.min() >= 0 and y.max() < self.configs.vocab_size, f"label out range, min = {y.min()}\tmax = {y.max()}\t vocab size = {self.configs.vocab_size}"
            assert logits.shape[0] * logits.shape[1] == y.numel(), f"shape mismatch: logits = {logits.shape} vs target = {y.shape}"

    
            scaler.scale(loss).backward()
            scaler.step(self.optim)
            scaler.update()

   
            if step % 100 == 0:
                torch.cuda.empty_cache()

            if step % log_every == 0 or step == self.configs.T_steps:
                val_loss = self.validate(self.val_loader)
                val_arr.append(val_loss)
                avg_loss = total_loss/log_every
                loss_arr.append(avg_loss)
                print(f"Step {step} | Training Loss: {avg_loss:.3f} | Validation Loss: {val_loss:.3f}")
                total_loss=0

            if step % 1000 == 0:
                print(f"Proxy step {step}/{self.configs.T_steps}, loss={loss.item():.4f}")

            if step == self.configs.t0:
                torch.save(self.train_model.state_dict(), f"proxy_early_{self.configs.block_size}.pt")
                print(f"Saved proxy early checkpoint at step {step}")

        # final checkpoint
        torch.save(self.train_model.state_dict(), f"proxy_late_{self.configs.block_size}.pt")
        print(f"Saved proxy final checkpoint at step {self.configs.T_steps}")
        return loss_arr ,val_arr
        


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


    def LearnabilityScore(self, type='composite', alpha_scale=None):

        # Load early proxy
        early = self.model_cls(self.configs).to(self.device)
        early.load_state_dict(torch.load(f"proxy_early_{self.configs.block_size}.pt", map_location=self.configs.device))
        early.eval()
        # Load late proxy
        late = self.model_cls(self.configs).to(self.device)
        late.load_state_dict(torch.load(f"proxy_late_{self.configs.block_size}.pt", map_location=self.configs.device))
        late.eval()

        print(f"[DEBUG] Early model on: {next(early.parameters()).device}")
        print(f"[DEBUG] Late model on: {next(late.parameters()).device}")


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
            alpha = alpha_scale if alpha_scale is not None else self.configs.alpha_scale
            comp_score = alpha * norm_irr + (1 - alpha)*norm_entropy
            return comp_score

        
        
        
        
