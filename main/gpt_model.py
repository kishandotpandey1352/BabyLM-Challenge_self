from models.gpt import GenerateGPT, GPT2Block, GPT2Model, GPTTrainer
from models.proxy_model import ProxyTrain
from utils.configs import ModelConfig, TrainConfig, ProxyConfig
from data.loader import get_loaders
from curriculum.scheduler import Scheduler
import pickle
import argparse
import torch


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--scoring", type=str, required=True, choices=['composite','Entropy','Loss'])
    parser.add_argument("--schedule_type", type=str, required=True, choices=['linear','sigmoid','tanh','log','exp'])
    

    args = parser.parse_args()
    path = args.data_path
    score_type = args.scoring
    schedule_type = args.schedule_type

    data = get_loaders(path)

    proxy = ProxyTrain(data['holdout_loader'], data['score_loader'], ProxyConfig(), GPT2Model)
    scores = proxy.LearnabilityScore(type=score_type)

    proxy = torch.load_state_dict(
        "trained_models/proxy.pt",
        map_location=torch.device("cpu")
        )

    scheduler = Scheduler(
        train_data=data['train_data'], 
        scores=scores,
        configs=ProxyConfig(),
        schedule_type=schedule_type,
        shuffle=True,      
    )

    parser.add_argument("--curriculum", required=True, choices=['on', 'off'])
    curricula = args.curriculum

    main = GPTTrainer(
        data['X_train_tensor'], data['y_train_tensor'], 
        data['train_loader'], data['val_loader'], 
        data['train_config'], data['model_config'], 
        scheduler
        )
    
    train_loss, val_loss, alphas, lambdas = main.train(scheduler=curricula)

    torch.save(main, "trained_models/main_model.pt")
