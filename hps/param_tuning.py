
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

def proxy_objective(trial, data, model, scheduler:str, score_type:str):
    min_lr_rate = trial.suggest_float("min_lr", 1e-5, 3e-4)
    max_lr_rate = trial.suggest_float("max_lr", 3e-4, 1e-3)
    dropout = trial.suggest_float("dropout", 0.0, 0.2, step=0.05)

    t0 = trial.suggest_categorical("t0", [25, 50, 100])

    if scheduler=='on' and score_type=='comp_score':
        alpha_config = trial.suggest_float("alpha_scale", 0.0, 1.0)
    else:
        alpha_config=.5

    proxy_config = ProxyConfig(
        min_lr=min_lr_rate,
        max_lr=max_lr_rate,
        batch_size=32,
        dropout=dropout,
        block_size=1024, 
        t0=t0,
        alpha_scale=alpha_config,

    )
    # lower model configs for speed
    proxy_config.n_head = 4
    proxy_config.n_embd = 256 //2
    proxy_config.n_layers = 4
    
    proxy = ProxyTrain(data['holdout_loader'], data['holdout_val_loader'], data['score_loader'], proxy_config, model)
    
    train_loss, val_loss = proxy.train(holdout_loader=data['holdout_loader'], val_loader=data['holdout_val_loader'])
    print(f"Validation Loss: {val_loss[-1]:.2}")
    return np.min(val_loss)

def main_objective(trial,x,y, data, use_scheduler, scores):

    # lower for speed
    model_config = ModelConfig()
    model_config.dropout=0.1
    model_config.block_size = 1024
    model_config.n_embd = 256
    model_config.n_head = 4
    model_config.n_layers = 4

    main_configs = TrainConfig(
        min_lr=3.35329e-05,
        max_lr=0.001739,
        batch_size=128,
        max_len=100,
        grad_accum_steps=1,
    )

    # in sceduler
    if use_scheduler=='on':
        gamma = trial.suggest_float("gamma",0.01, 0.5)
        schedule_type = trial.suggest_categorical("schedule_type", ['linear','sigmoid','tanh','log','exp'])

        scheduler = Scheduler(    
            train_data=data['train_dataset'],
            scores=scores,
            configs=main_configs,
            schedule_type=schedule_type,
            gamma=gamma,
            shuffle=True
        )
    else:
        scheduler=None

    main = GPTTrainer(
        x, y, 
        data['train_loader'],
        data['val_loader'], 
        main_configs,
        model_config,
        scheduler
    )

    _, val_loss, _, _ = main.train(scheduler=scheduler)
    return np.min(val_loss)


    # main model w/ best proxy
if __name__=='__main__':
    print("[debug] CUDA available:", torch.cuda.is_available())
    repeat(seed=88)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--toggle_scheduler", required=True, choices=['on', 'off'])
    parser.add_argument("--score_type", required=True, choices=['comp_score','loss','entropy'])
    parser.add_argument("--n_tokens", type=int, required=True)
    parser.add_argument("--proxy_n_trials", type=int, required=True)
    parser.add_argument("--main_n_trials", type=int, required=True)
    parser.add_argument("--proxy_train", type=str, required=True, choices=['on', 'off'])
    parser.add_argument("--main_train", type=str, required=True, choices=['on', 'off'])
   

    args = parser.parse_args()

    curricula = args.toggle_scheduler 
    scores_type = args.score_type
    path = args.data_path
    n_tokens = args.n_tokens
    main_train = args.main_train
    proxy_train = args.proxy_train
    proxy_n_trials = args.proxy_n_trials
    main_n_trials = args.main_n_trials


    if n_tokens is not None:
        data = get_loaders(path, n_tokens, split_type='tune')
    else:
        data = get_loaders(path, n_tokens, split_type='tune')

    scores = None

    if proxy_train == 'on':
        proxy_study = optuna.create_study(direction='minimize', study_name=f"proxy_hps_{n_tokens}", load_if_exists=True)
        proxy_study.optimize(lambda trial: proxy_objective(trial, data, GPT2Model, scheduler=curricula, score_type=scores_type), n_trials=proxy_n_trials)

        best_proxy = proxy_study.best_trial.params
        proxy_trainer = ProxyTrain(data['holdout_loader'], data['holdout_val_loader'], data['score_loader'], ProxyConfig(**best_proxy), GPT2Model)
        proxy_trainer.train(holdout_loader=data['holdout_loader'], val_loader=data['holdout_val_loader'])

        alpha_scale = best_proxy['alpha_scale'] if 'alpha_scale' in best_proxy else 0.5
        all_scores = proxy_trainer.LearnabilityScore(type=scores_type, alpha_scale=alpha_scale)
        joblib.dump(all_scores,f"trained_models/proxy_scores.pkl")
        torch.save(proxy_trainer.train_model.state_dict(), f"trained_models/proxy_model_hps.pt")
        joblib.dump(proxy_study, f"trained_models/proxy_study.pkl")

    if main_train == 'on':
        if scores is None:
            try:
                scores = joblib.load(f"trained_models/proxy_scores.pkl")
                print(f"[info] loaded proxy scores, keys={list(scores.keys())}")
            except FileNotFoundError:
                raise RuntimeError("Proxy scores not found. You must run proxy training first.")

        if curricula == 'on':
            print(f"\n[info] Running model w/ scheduler score type = {curricula}")
            main_study = optuna.create_study(direction='minimize', study_name=f"{scores_type}_study", load_if_exists=True)
            main_study.optimize(lambda trial: main_objective(trial,  None, None, data, 'on', scores[scores_type]), n_trials=main_n_trials)
            joblib.dump(
                {"score_type": scores_type, "params": main_study.best_trial.params},
                f"trained_models/main_{scores_type}_fixed_result.pkl"
            )
        else:
            print("\n[info] Running model w/out curriculum for baseline")
            main_study = optuna.create_study(direction='minimize', study_name=f"main_hps_{n_tokens}", load_if_exists=True)
            main_study.optimize(
                lambda trial: main_objective(trial, None, None, data, 'off', scores),
                n_trials=main_n_trials
            )

            print(f"Best Model Params: {main_study.best_trial}")
            print("Main best hyperparams:", main_study.best_trial.params)
            joblib.dump(main_study, "trained_models/main_baseline_study.pkl")
