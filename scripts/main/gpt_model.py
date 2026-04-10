from models.gpt import GenerateGPT, GPT2Block, GPT2Model, GPTTrainer
from models.proxy_model import ProxyTrain
from utils.configs import ModelConfig, TrainConfig, ProxyConfig
from data.loader import get_loaders
from curriculum.scheduler import Scheduler
import argparse
import torch
import time
import joblib
import pandas as pd

if __name__=='__main__':

    print("cuda available:", torch.cuda.is_available())
    print("device being used:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--scoring", type=str, required=True, choices=['comp_score','entropy','loss'])
    parser.add_argument("--curriculum", required=True, choices=['on', 'off'])
    parser.add_argument("--data_size", type=str, required=True)
    parser.add_argument("--n_tokens", type=int)

    args = parser.parse_args()
    path = args.data_path
    score_type = args.scoring
    curricula = args.curriculum
    data_size = args.data_size
    n_tokens = args.n_tokens

    if n_tokens is not None:
        data = get_loaders(path,n_tokens, split_type='final')
    else:
        data = get_loaders(path,None, split_type='final')

    start = time.time()
    if curricula=='on':
        proxy_model = GPT2Model(ProxyConfig())
        proxy_model.load_state_dict(torch.load(f"trained_models/proxy_model_{data_size}.pt", map_location='cpu'))
        
        proxy_model.eval()

        main_study = joblib.load(f"trained_models/main_{score_type}_fixed_result.pkl")
        print(main_study)
        scores = torch.load(f"trained_models/proxy_scores_{data_size}.pt")
        score = scores[score_type]

        train_config = TrainConfig()
        train_config.total_steps = train_config.epochs * len(data['train_loader'])

        print(f"\n __________________________\n")
        print(f"=====> [info] Trainin on scheduler params:\nScore Type = {score_type}\t\nSchedule Type = {main_study['params']['schedule_type']}\t\nGamma value = {main_study['params']['gamma']}")
        print(f"\n __________________________\n")
        scheduler = Scheduler(
            train_data=data['train_dataset'], 
            scores=score,
            configs=train_config,
            schedule_type=main_study['params']['schedule_type'],
            shuffle=True,  
            gamma=main_study['params']['gamma']    
            )
            
        main = GPTTrainer(
            None, None,
            data['train_loader'], data['val_loader'], 
            TrainConfig(), ModelConfig(),
            scheduler=scheduler,
            )
        
        train_loss, val_loss, alphas, lambdas = main.train(scheduler=scheduler)
        end =time.time() - start
        print(f"Main{data_size} training complete. Time: {time.time() - start:.3f}s")

        train_metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'alphas':alphas,
            'lambdas':lambdas,
        }
        df = pd.DataFrame(train_metrics)
        df.to_csv(f"trained_models/train_metrics_{data_size}_{score_type}.csv")

        torch.save(main.model.state_dict(), f"trained_models/main_{score_type}_{data_size}model_weights.pt")

    else:
        main = GPTTrainer(
            None, None,
            data['train_loader'], data['val_loader'], 
            TrainConfig(), ModelConfig(),
            scheduler=None
            )
        
        train_loss, val_loss, alphas, lambdas = main.train(scheduler=None)
        end =time.time() - start
        print(f"Main{data_size} training complete. Time: {time.time() - start:.3f}s")
        
        torch.save(main.model.state_dict(), f"trained_models/main_baseline_{data_size}model_weights.pt")
        #torch.save(main, f"trained_models/full_baseline_{data_size}main_model.pt")

    num_params = sum(p.numel() for p in main.model.parameters() if p.requires_grad)
    print(f"[info] Main{data_size} parameters: {num_params / 1e6:.2f}M")

