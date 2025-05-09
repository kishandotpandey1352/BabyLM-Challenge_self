# main

from curriculum.scheduler import Scheduler
from models.gpt import GenerateGPT, GPT2Block, GPT2Model, GPTTrainer
from models.proxy_model import ProxyTrain
from utils.configs import ModelConfig, TrainConfig, ProxyConfig
from data.loader import get_loaders
import pickle
import argparse
import torch

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    
    args = parser.parse_args()
    path = args.data_path

    data = get_loaders(path)

    proxy = ProxyTrain(data['holdout_loader'], data['score_loader'], ProxyConfig(), GPT2Model)
    proxy.train()

    torch.save(proxy.state_dict(), f"trained_models/proxy_model.pt")


    
    
    
    


