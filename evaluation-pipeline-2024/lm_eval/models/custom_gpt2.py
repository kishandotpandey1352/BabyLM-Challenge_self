from lm_eval.base import BaseLM
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import GPT2Tokenizer

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.gpt import GPT2Model
from utils.configs import ModelConfig

class CustomGPT2(BaseLM):
    def __init__(self, pretrained, device="cuda", dtype="float32", **kwargs):
        super().__init__()
        print("[DEBUG] Initializing CustomGPT2")

        self.device = torch.device(device)
        config = ModelConfig()

        print("[DEBUG] Loading model from:", pretrained)
        self.model = GPT2Model(config).to(self.device)
        self.model.load_state_dict(torch.load(pretrained, map_location=self.device))
        self.model.eval()
        print("[DEBUG] Model loaded and moved to", self.device)

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.vocab_size = config.vocab_size
        self.eot_token_id = self.tokenizer.eos_token_id

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, return_tensors="pt").squeeze(0).tolist()

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def generate_until(self, requests):
        raise NotImplementedError

    def loglikelihood(self, requests):
        """
        Compute log-likelihoods for (context, continuation) pairs.
        Returns: List of tuples (logprob, is_greedy)
        """
        results = []
        for context, continuation in tqdm(requests, desc="Evaluating loglikelihood"):
            # Encode input
            full_text = context + continuation
            full_tokens = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)
            context_tokens = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
            continuation_tokens = self.tokenizer.encode(continuation, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(full_tokens[:, :-1])
                logits = outputs.logits  # shape: (1, seq_len-1, vocab_size)

                # Get the target logits for continuation part
                target_start = context_tokens.shape[1] - 1  # index where continuation starts
                target_logits = logits[:, target_start : target_start + continuation_tokens.shape[1], :]

                # Get actual continuation token ids (as labels)
                target_ids = full_tokens[:, target_start + 1 : target_start + 1 + continuation_tokens.shape[1]]

                # Cross entropy: log softmax + NLL
                log_probs = F.log_softmax(target_logits, dim=-1)
                selected_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)

                log_likelihood = selected_log_probs.sum().item()
                results.append((log_likelihood, True))  # "True" = greedy decoding match

        return results

    @property
    def max_length(self):
        return 1024

    @property
    def batch_size(self):
        return 1
