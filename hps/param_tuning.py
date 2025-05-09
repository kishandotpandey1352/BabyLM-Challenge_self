
from utils.configs import ProxyConfig, TrainConfig, ModelConfig
from models.proxy_model import ProxyTrain
from models.gpt import GPTTrainer, GPT2Model
from curriculum.scheduler import Scheduler
import numpy as np
import torch
import optuna
from data.loader import get_loaders
import argparse
import random
import joblib

def repeat(seed=88):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True

def proxy_objective(trial, holdout_loader, score_loader, model):
    min_lr_rate = trial.suggest_float("min_lr", 1e-5, 3e-4)
    max_lr_rate = trial.suggest_float("max_lr", 3e-4, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.2, step=0.05)
    seq_len = trial.suggest_categorical("block_size", [32, 64, 128])

    T_steps = trial.suggest_categorical("T_steps", [500, 1000])
    t0 = trial.suggest_categorical("t0", [25, 50, 100])
    alpha_config = trial.suggest_float("alpha_scale", 0.0, 1.0)

    proxy_config = ProxyConfig(
        min_lr=min_lr_rate,
        max_lr=max_lr_rate,
        batch_size=batch_size,
        dropout=dropout,
        block_size=seq_len, 
        T_steps=T_steps,
        t0=t0,
        alpha_scale=alpha_config
    )
    
    proxy = ProxyTrain(holdout_loader, score_loader, proxy_config, model)
    proxy.train()

    scores = proxy.LearnabilityScore(type='composite')
    return scores.std().item()

def main_objective(trial, x, y, train_loader, val_loader, train_dataset, use_scheduler, scores):
    # in train configs
    min_lr = trial.suggest_float("min_lr", 1e-5, 1e-4)
    max_lr = trial.suggest_float("max_lr", 1e-4, 3e-3)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    max_len = trial.suggest_categorical("max_len", [100, 500, 1000])
    grad_accum_steps = trial.suggest_categorical("grad_accum_steps", [1, 2, 3])

    # in model block
    dropout = trial.suggest_float("dropout",0.0, 0.5)

    # in sceduler
    if use_scheduler=='on':
        gamma = trial.suggest_float("gamma",0.01, 0.5)
        schedule_type = trial.suggest_categorical("schedule_type", ['linear','sigmoid','tanh','log','exp'])

        scheduler = Scheduler(    
            train_data=train_dataset,
            scores=scores,
            configs=ProxyConfig(),
            schedule_type=schedule_type,
            gamma=gamma,
            shuffle=True
        )
    else:
        scheduler=None

    model_config = ModelConfig()
    model_config.dropout=dropout
    model_config.block_size = max_len

    main_configs = TrainConfig(
        min_lr=min_lr,
        max_lr=max_lr,
        batch_size=batch_size,
        max_len=max_len,
        grad_accum_steps=grad_accum_steps,
    )
    
    main = GPTTrainer(
        x, y, 
        train_loader,
        val_loader, 
        main_configs,
        model_config,
        scheduler
    )

    _, val_loss, _, _ = main.train()
    return np.mean(val_loss)

if __name__=='__main__':

    repeat(seed=88)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--toggle_scheduler", required=True, choices=['on', 'off'])
    parser.add_argument("--score_type", required=True, choices=['composite','Entropy','Loss'])
    parser.add_argument("--n_tokens", type=int, required=True)
    parser.add_argument("--proxy_n_trials", type=int, required=True)
    parser.add_argument("--main_n_trials", type=int, required=True)

    args = parser.parse_args()

    curricula = args.toggle_scheduler 
    scores_type = args.score_type
    path = args.data_path
    n_tokens = args.n_tokens
    proxy_n_trials = args.proxy_n_trials
    main_n_trials = args.main_n_trials


    if n_tokens:
        data = get_loaders(path, n_tokens)
    else:
        data = get_loaders(path)

    #print(data.items())
    
    proxy_study = optuna.create_study(direction='minimize', study_name=f"proxy_hps_{n_tokens}", load_if_exists=True)
    proxy_study.optimize(
    lambda trial: proxy_objective(trial, data['holdout_loader'], data['score_loader'], GPT2Model), 
    n_trials=proxy_n_trials, # for proxy 10-50
    )

    # best params
    print(f"Best Proxy Model Params: {proxy_study.best_trial}")
    best_proxy = proxy_study.best_trial.params
    print("Proxy best hyperparams:", best_proxy)

    # train proxy on best params
    proxy_trainer = ProxyTrain(data['holdout_loader'], data['score_loader'], ProxyConfig(**best_proxy), GPT2Model)
    proxy_trainer.train()
    scores = proxy_trainer.LearnabilityScore(type=scores_type)
    np.save("proxy_scores.npy", arr=scores.cpu().numpy()) # for later 
    torch.save(proxy_trainer, "trained_models/proxy_model_hps.pt")

    # main model w/ best proxy
    main_study = optuna.create_study(direction='minimize', study_name=f"main_hps_{n_tokens}", load_if_exists=True)
    main_study.optimize(
    lambda trial: main_objective(trial, data['X_train_tensor'], data['y_train_tensor'], data['train_loader'], data['val_loader'], data['train_dataset'], curricula, scores),
    n_trials=main_n_trials, # for main model 100 +
    )

    print(f"Best Model Params: {main_study.best_trial}")
    print("Main best hyperparams:", main_study.best_trial.params)

    # save 
    joblib.dump(proxy_study, "proxy_study.pkl")
    joblib.dump(main_study, "main_study.pkl")

