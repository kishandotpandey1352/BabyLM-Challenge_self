# proxy_main.py

from curriculum.scheduler import Scheduler
from models.gpt import GenerateGPT, GPT2Block, GPT2Model, GPTTrainer
from models.proxy_model import ProxyTrain
from utils.configs import ModelConfig, TrainConfig, ProxyConfig
from data.loader import get_loaders
import pickle
import argparse
import torch
import time
import numpy as np
import joblib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_size", type=str, required=True)
    parser.add_argument("--n_tokens", type=int)


    args = parser.parse_args()
    path = args.data_path
    data_size = args.data_size
    n_tokens = args.n_tokens

    if n_tokens is not None:
        data = get_loaders(path, n_tokens, split_type='tune')
    else:
        data = get_loaders(path, None, split_type='tune')

    start = time.time()

    proxy_study = joblib.load("trained_models/proxy_study.pkl")
    best_proxy_params = proxy_study.best_trial.params
    print(f"=====> [info] Best Proxy Params = {best_proxy_params}")


    # Train the proxy model
    proxy = ProxyTrain(
        data['holdout_loader'],
        data['holdout_val_loader'],
        data['score_loader'],
        ProxyConfig(**best_proxy_params),
        GPT2Model,

    )
    
    #train_loss, val_loss = proxy.train(data['holdout_loader'], data['holdout_val_loader'])
    train_loss, val_loss = proxy.train(holdout_loader=data['holdout_loader'], val_loader=data['holdout_val_loader'])

    np.save(f"trained_models/proxy_loss_{data_size}.npy", np.array(train_loss))
    np.save(f"trained_models/proxy_val_loss_{data_size}.npy", np.array(val_loss))

    # Compute and save all scores
    alpha = proxy.configs.alpha_scale if hasattr(proxy.configs, "alpha_scale") else 0.5
    torch.cuda.empty_cache()
    start_scores = time.time()
    scores_dict = proxy.LearnabilityScore(type='comp_score', alpha_scale=alpha)  # type doesn't matter since all are returned
    print(f"=====> [info] Scores took {time.time() - start_scores}")

    torch.save(scores_dict, f"trained_models/proxy_scores_{data_size}.pt")

    # Save model weights
    torch.save(proxy.train_model.state_dict(), f"trained_models/proxy_model_{data_size}.pt")

    end = time.time() - start
    print(f"Proxy {data_size} Running time: {end:.2}s")
    print(f"[info] Proxy{data_size} training complete. Time: {time.time() - start:.3f}s")

    # Save model weights
    torch.save(proxy.train_model.state_dict(), f"trained_models/proxy_model_{data_size}.pt")




    
    
    
    


