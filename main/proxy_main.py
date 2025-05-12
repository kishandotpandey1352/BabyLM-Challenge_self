# proxy_main.py

from curriculum.scheduler import Scheduler
from models.gpt import GenerateGPT, GPT2Block, GPT2Model, GPTTrainer
from models.proxy_model import ProxyTrain
from utils.configs import ModelConfig, TrainConfig, ProxyConfig
from hf_compat.compatibility import hfWrapper, hfConfig
from data.loader import get_loaders
import pickle
import argparse
import torch
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_size", type=str, required=True)

    args = parser.parse_args()
    path = args.data_path
    data_size = args.data_size

    # Load tokenized data
    data = get_loaders(path, None, split_type='tune')
    start = time.time()

    # Train the proxy model
    proxy = ProxyTrain(
        data['holdout_loader'],
        data['holdout_val_loader'],
        data['score_loader'],
        ProxyConfig(),
        GPT2Model,

    )
    
    proxy.train()
    end = start - time.time()
    print(f"Running time: {end}s")
    print(f"[info] Proxy{data_size} training complete. Time: {time.time() - start:.3f}s")

    # Save model weights
    torch.save(proxy.train_model.state_dict(), "trained_models/proxy_model.pt")

    # Save Hugging Face compatible version
    configs = hfConfig(ProxyConfig())
    model = hfWrapper(configs)
    model.save_pretrained(f"trained_models/hf_proxy_{data_size}")
    configs.save_pretrained(f"trained_models/hf_proxy_{data_size}")

    num_params = sum(p.numel() for p in proxy.train_model.parameters() if p.requires_grad)
    print(f"[info] Proxy{data_size} parameters: {num_params / 1e6:.2f}M")



    
    
    
    


