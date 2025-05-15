import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from curriculum.scheduler import Scheduler
from utils.configs import ModelConfig, TrainConfig
from torch.cuda.amp import autocast, GradScaler
import csv
import os

class MetricLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.headers = ["epoch", "train_loss", "val_loss", "alpha", "beta", "lambda_val", "num_samples"]
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, epoch, train_loss, val_loss, alpha,  beta, lambda_val, num_samples):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, alpha,  beta,lambda_val, num_samples])

# ==== GPT2 Decoder ==== #

### UPDATE FOR TORCH USE ###

class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias) # create W attn embds
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # layer norm
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias), # expand dims (4 * from Attention all u need paper)
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd), # project back 
            nn.Dropout(config.dropout)
        )
        
        self.attn_dropout = nn.Dropout(config.dropout)

    def Mask(self, t):
        mask = torch.tril(torch.ones((t,t), device=self.config.device)).unsqueeze(0).unsqueeze(0)
        return mask

    def SelfAttention(self,x, mask=None):
        b,t,c = x.shape
        q, k, v = self.c_attn(x).split(self.config.n_embd, dim=2)
        
        q =q.reshape(b, t, self.config.n_head, c//self.config.n_head).permute(0,2,1,3)
        k = k.reshape(b, t, self.config.n_head, c//self.config.n_head).permute(0,2,1,3)
        v = v.reshape(b, t, self.config.n_head, c//self.config.n_head).permute(0,2,1,3)

        score = (q @ k.transpose(-2,-1)) / math.sqrt(k.shape[-1])

        if mask is None:
            mask = self.Mask(t)
        score = score.masked_fill(mask==0, float("-inf"))

        a = nn.functional.softmax(score, dim=-1)
        a = self.attn_dropout(a)
        head = a @ v
        out = head.permute(0,2,1,3).reshape(b,t,c) # back to input dims
        out = self.c_proj(out)
        #print("Type of self.SelfAttention(x):", type(out))
        return out
    
    def forward(self, x, mask=None):
        # self attention ad layer norm
        x = x + self.SelfAttention(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x)) # resudual connections
       
        return x
    
class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embd = nn.Embedding(config.vocab_size, config.n_embd)
        self.posn_embd = nn.Embedding(config.block_size, config.n_embd)

        self.proj = nn.Linear(config.n_embd, config.vocab_size)

        self.blocks = nn.Sequential(*[GPT2Block(config) for _ in range(config.n_layers)])
        self.fln = nn.LayerNorm(config.n_embd)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        b, t = x.size()
        tokn_embd = self.token_embd(x)
        posn_idcs = torch.arange(t, device=x.device).unsqueeze(0).expand(b,t)
        assert x.size(1) <= self.posn_embd.num_embeddings, (
            f"Seq length {x.size(1)} exceeds block size {self.posn_embd.num_embeddings}"
        )
        posn_embd = self.posn_embd(posn_idcs)

        x = tokn_embd + posn_embd

        x = self.blocks(x)
        x = self.fln(x) # final linear layer
        return self.proj(x) # project back to (enbedding size, vocab size)


class GPTTrainer:
    def lr_schedule(self, optim, train_config):
        def lr_lambda(current_step):
            if current_step < train_config.warmup_iters:
                return float(current_step) / float(max(1, train_config.warmup_iters))
            return max(0.0, float(train_config.total_steps - current_step) / float(max(1, train_config.total_steps - train_config.warmup_iters)))
        return LambdaLR(optim, lr_lambda)

    def __init__(self, x, y, train_loader, val_loader, train_config, model_config, scheduler=None, check_points_dir=None):
        self.x, self.y = x, y
        self.train_config = train_config
        self.save_checkpoints = train_config.save_checkpoints
        self.check_points_dir = check_points_dir

        self.train_loader = train_loader
        self.val_loader = val_loader
        #self.scheduler = scheduler

        self.device = model_config.device

        self.train_config.total_steps = self.train_config.epochs * len(self.train_loader)

        self.model = GPT2Model(model_config).to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=train_config.max_lr)
        self.schedule = self.lr_schedule(self.optim, self.train_config)
        self.criterion = nn.CrossEntropyLoss()

        self.step_count = 0
        print(f"Using device: {self.device}")

        # lyponav regularisation
        self.L0 = None # get initial val loss
        self.prev_val_loss = None # get prev val loss
        self.lambdas = [] # track lambda of leraning over time

        self.scaler = GradScaler()


    def step(self, inputs, targets):
        self.model.train()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        with autocast():
            logits = self.model(inputs)
            b, t, v = logits.shape
            logits = logits.view(b * t, v)
            targets = targets.view(b * t)

            loss = self.criterion(logits, targets) / self.train_config.grad_accum_steps

        self.scaler.scale(loss).backward()
    

        if (self.step_count + 1) % self.train_config.grad_accum_steps == 0:
            self.scaler.step(self.optim)
            self.scaler.update()
            self.schedule.step()
            self.optim.zero_grad()

        self.step_count += 1
        if self.step_count % 1000 == 0:
            print(f"[GPU] memory allocated = {torch.cuda.memory_allocated()/1e6:.2f}")
            print(f"[GPU] memory reserved = {torch.cuda.memory_reserved()/1e6:.2f}")


        if self.train_config.save_checkpoints and (self.check_points_dir and self.step_count % self.train_config.save_every_steps == 0):
            checkpoint = {
                "step": self.step_count,
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
                "schedule": self.schedule.state_dict()
            }
            torch.save(checkpoint, f"{self.check_points_dir}/checkpoint_step_{self.step_count}.pt")
            print(f" ------> Saved Checkpoint at {self.step_count}")

        if self.step_count % 1000 ==0:
            print(f"Step {self.step_count:04d} | Loss: {loss.item():.4f}")
        return loss.item()

    def validate(self, val_loader, max_batches=50):
        self.model.eval()
        n=0

        print(f"[debug] Lenght of validation loader = {len(val_loader)}")

        total_loss = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if max_batches is not None and i>=max_batches:
                    break
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                b, t, v = logits.shape
                logits = logits.view(b * t, v)
                y = y.view(b * t)

                loss = self.criterion(logits, y)
                total_loss += loss.item()
                n+=1
        return total_loss / n

    def train(self, scheduler=None):
        print("Model device:", next(self.model.parameters()).device)

        if scheduler is None:
            print("Running Baseline Model")
        else:
            print("Running Model with Scheduler")
        logger = MetricLogger("logs/train_metrics.csv")

        alpha = 1.
        train_loss_arr = []
        val_loss_arr = []
        alpha_arr = []
        lambdas_arr = []


        for epoch in range(self.train_config.epochs):
            total_loss = 0.
            n_batches = 0
            if self.lambdas:
                lambdas_arr.append(self.lambdas[-1])

            if scheduler is not None:
                alpha *= scheduler.lyapunovReguliser(self.lambdas)
                alpha_arr.append(alpha)
                train_loader = scheduler.seqentialBatch(alpha)
            else:
                train_loader = self.train_loader

            if self.step_count > 0:
                current_lr = self.optim.param_groups[0]['lr']
                print(f"Epoch {epoch + 1} | LR: {current_lr:.6f}")
            else:
                print(f"Epoch {epoch + 1} | Starting warmup...")

            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}, alpha {alpha}", miniters=100):
                loss_val = self.step(x, y)
                total_loss += loss_val
                n_batches += 1

                if scheduler is not None and self.step_count % self.train_config.alpha_update_steps == 0:
                    alpha *= scheduler.lyapunovReguliser(self.lambdas)
                    print(f"[Step {self.step_count}] updated alpha: {alpha:.4f}")

            avg_loss = total_loss / n_batches
            train_loss_arr.append(avg_loss)
    
            val_loss = self.validate(self.val_loader)
            val_loss_arr.append(val_loss)

            # regularise sample scaling 
            if self.L0 is None:
                self.L0 = val_loss # get initial validation loss
            if self.prev_val_loss is not None:
                delta_L = val_loss - self.prev_val_loss
                if delta_L != 0:
                    lambda_n = (1/(1+epoch)) * math.log(abs(delta_L)/self.L0)  # measure divergence
                    self.lambdas.append(lambda_n)
                    print(lambda_n)

            self.prev_val_loss = val_loss # update prev

            print(f"Epoch {epoch + 1} | Training Loss: {avg_loss:.2f} | Validation Loss: {val_loss:.4f}")
            
            # logging for analysis
            lambda_val = self.lambdas[-1] if self.lambdas else 0.0
            beta_t = scheduler.current_beta if scheduler else 1.0
            prct_samples = scheduler.prct_seen if scheduler else 100
            # add epoch avg scores + mean and std, min and max

            logger.log(
                epoch=epoch + 1,
                train_loss=avg_loss,
                val_loss=val_loss,
                alpha=alpha,
                beta=beta_t,
                lambda_val=lambda_val,
                num_samples=prct_samples
                )

            if self.check_points_dir and (epoch + 1) % self.train_config.save_every == 0:
                torch.save(self.model.state_dict(), f"{self.check_points_dir}/model_epoch_{epoch + 1}.pt")
        return train_loss_arr, val_loss_arr, alpha_arr, lambdas_arr



# Helper function for filtering logits using top-k and/or nucleus (top-p) sampling
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):

    # top-K filtering: keep only k highest tokens
    if top_k > 0:
        values, indices = torch.topk(logits, top_k)
        threshold = values[-1]
        logits[logits < threshold] = filter_value

    # top-p (nucleus) filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # remove tokens with cumulative prob above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # shift the idcs to kep the 1st token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


class GenerateGPT:
    def __init__(self, model, tokenizer, prompt, max_len=50):

        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_len = max_len

    def generate(self, temperature=1.0, top_k=0, top_p=0.0):
        # Set model to evaluation mode
        self.model.eval()

        # Encode the prompt into token IDs (assumes tokenizer provides a numerical encoding)
        # Make sure the prompt is encoded as a list of token IDs.
        input_ids = self.tokenizer.encode(self.prompt)
        input_tensor = torch.tensor([input_ids], device=next(self.model.parameters()).device)

        # Autoregressive gen loop
        with torch.no_grad():
            for _ in range(self.max_len):
                logits = self.model(input_tensor)
                next_token_logits = logits[0, -1, :]

                # ccale logits w/ temp
                next_token_logits = next_token_logits / temperature

                # apply top-k/top-p filtering if required
                filtered_logits = top_k_top_p_filtering(next_token_logits.clone(), top_k=top_k, top_p=top_p)
                probabilities = F.softmax(filtered_logits, dim=-1)

                # Sample the next token
                next_token = torch.multinomial(probabilities, num_samples=1).item()

                # add sampled token to input ids tensor
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=input_tensor.device)], dim=1)

                if next_token == self.tokenizer.eos_token_id: #end of sequence token
                    break

        # Decode the tokens back to text
        generated_text = self.tokenizer.decode(input_tensor[0].tolist())
        return generated_text

 
