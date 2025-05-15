import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
from models.gpt import GPT2Model
from utils.configs import ProxyConfig, TrainConfig
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
import time

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


    def validate(self, val_loader, max_batches):

        self.train_model.eval()
        total_loss = 0
        n=0

        print(f"[debug] length of proxy validation loader = {len(val_loader)}")

        
    def validate(self, val_loader, max_bathces=10):
        self.train_model.eval()
        total_loss = 0
        n=0
        
        print(f"[debug] length of proxy validation loader = {len(val_loader)}")

        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if max_bathces is not None and i >= max_bathces:
                    break
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.train_model(x)
                b,t,v = logits.shape
                logits=logits.view(b*t, v)
                y = y.view(b*t)

                loss = self.criterion(logits, y)
                total_loss += loss.item()
                n+=1
        return total_loss /n


    def train(self, holdout_loader, val_loader):
        print("Model device:", next(self.train_model.parameters()).device)
        hold_iter = iter(holdout_loader)
        total_loss = 0
        loss_arr = []
        val_arr = []

        scaler = GradScaler()
        
        for step in trange(1, self.configs.T_steps + 1, desc="Proxy training", miniters=1):
            try:
                x, y = next(hold_iter)
            except StopIteration:
                hold_iter = iter(holdout_loader)
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

   
            if step % 1000 == 0:
                print(f"[GPU] memory allocated = {torch.cuda.memory_allocated()/1e6:.2f}")
                print(f"[GPU] memory reserved = {torch.cuda.memory_reserved()/1e6:.2f}")

            if step == self.configs.t0:
                torch.save(self.train_model.state_dict(), f"proxy_early_{self.configs.block_size}.pt")
                print(f"Saved proxy early checkpoint at step {step}")

                t_early = time.time()
                early_val = self.validate(val_loader, max_bathces=10)
                print(f"=====> \t[debug] time to perform early validte = {time.time() - t_early:.3}")
                val_arr.append(early_val)

                loss_arr.append(total_loss/step)
                print(f"[snapshot t0={step}] train_loss={loss_arr[-1]:.3f}  val_loss={early_val:.3f}")

            
            if  step == self.configs.T_steps:
                torch.save(self.train_model.state_dict(), f"proxy_late_{self.configs.block_size}.pt")
                print(f"Saved proxy final checkpoint at step {self.configs.T_steps}")

                t_late = time.time()
                late_val = self.validate(val_loader, max_bathces=20)
                print(f"=====>\t[debug] time to perform late validte = {time.time() - t_late:.3}")

                val_arr.append(late_val)
                loss_arr.append(total_loss/step)

                loss_arr.append(total_loss/step)
                print(f"[final T={step}] train_loss={loss_arr[-1]:.3f}  val_loss={late_val:.3f}")


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

        # pre-allocate the big tensors
        N = len(self.score_loader.dataset)
        irr_loss = torch.empty(N, device=self.device)
        entropy  = torch.empty(N, device=self.device)


        idx = 0
        V = self.configs.vocab_size
        with torch.no_grad():
            for x, y in self.score_loader:
                B, L = x.size(0), x.size(1)
                x, y = x.to(self.device), y.to(self.device)

                #  forward both proxies batch
                logits_e = early(x)  
                logits_l = late(x)  

                # per-token losses
                flat_e = logits_e.view(-1, V)
                flat_l = logits_l.view(-1, V)
                y_flat = y.view(-1)

                tok_e = F.cross_entropy(flat_e, y_flat, reduction='none')
                tok_l = F.cross_entropy(flat_l, y_flat, reduction='none')
                seq_e = tok_e.view(B, L).mean(dim=1)  
                seq_l = tok_l.view(B, L).mean(dim=1)  
                delta = seq_e - seq_l                

                # compute ent on late model
                p = F.softmax(logits_l, dim=-1)           
                logp = torch.log(p + 1e-8)
                tok_ent = -(p * logp).sum(dim=-1)            
                seq_ent = tok_ent.mean(dim=1)             

                # write into big tensors
                irr_loss[idx:idx+B] = delta
                entropy [idx:idx+B] = seq_ent
                idx += B
            
        eps = 1e-8

        # normalise for composite score
        norm_irr = (irr_loss - irr_loss.min()) / (irr_loss.max() - irr_loss.min() + eps)
        norm_entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + eps)

        # return flat tensor of learnability score
        alpha = alpha_scale if alpha_scale is not None else self.configs.alpha_scale
        comp_score = alpha * norm_irr + (1 - alpha)*norm_entropy
        return {
            'comp_score':comp_score,
            'loss': irr_loss,
            'entropy':entropy
        }

        
        
        
        
