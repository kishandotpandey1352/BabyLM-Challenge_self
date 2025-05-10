




from models.gpt import GenerateGPT, GPT2Block, GPT2Model, GPTTrainer
from models.proxy_model import ProxyTrain
from utils.configs import ModelConfig, TrainConfig, ProxyConfig
from data.loader import get_loaders
from curriculum.scheduler import Scheduler
from hf_compat.compatibility import hfWrapper, hfConfig
import argparse
import torch
import time
if __name__=='__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--scoring", type=str, required=True, choices=['composite','entropy','loss'])
    parser.add_argument("--schedule_type", type=str, required=True, choices=['linear','sigmoid','tanh','log','exp'])   
    parser.add_argument("--curriculum", required=True, choices=['on', 'off'])
    parser.add_argument("--data_size", type=str, required=True)

    args = parser.parse_args()
    path = args.data_path
    score_type = args.scoring
    schedule_type = args.schedule_type
    curricula = args.curriculum
    data_size = args.data_size

    data = get_loaders(path, None)
    start = time.time()
    if curricula=='on':
        proxy_model = GPT2Model(ProxyConfig())
        proxy_model.load_state_dict(torch.load("trained_models/proxy_model.pt", map_location='cpu'))
        proxy_model.eval()

        proxy = ProxyTrain(data['holdout_loader'], data['score_loader'], ProxyConfig(), GPT2Model)
        proxy.train_model = proxy_model

        all_scores = proxy.LearnabilityScore(type=score_type)
        torch.save(all_scores,f"logs/proxy_scores_{score_type}.pt")
        score = all_scores[score_type]

        train_config = TrainConfig()
        train_config.total_steps = train_config.epochs * len(data['train_loader'])

        scheduler = Scheduler(
            train_data=data['train_dataset'], 
            scores=score,
            configs=train_config,
            schedule_type=schedule_type,
            shuffle=True,      
            )
    else:
        scheduler=None


    main = GPTTrainer(
        data['X_train_tensor'], data['y_train_tensor'], 
        data['train_loader'], data['val_loader'], 
        data['train_config'], data['model_config'], 
        scheduler=scheduler
    )


    train_loss, val_loss, alphas, lambdas = main.train(scheduler=scheduler)
    end =time.time() - start
    print(f"Main{data_size} training complete. Time: {time.time() - start:.3f}s")
    torch.save(main.model.state_dict(), f"trained_models/main_{score_type}_{data_size}model_weights.pt")
    torch.save(main, f"trained_models/full_{score_type}_{data_size}main_model.pt")

    #compatibility w/ for hugging face 
    configs = hfConfig(ModelConfig())
    model = hfWrapper(configs)

    model.save_pretrained(f"trained_models/hf_main_{data_size}_{score_type}") # model save + config json
    configs.save_pretrained(f"trained_models/hf_main_{data_size}_{score_type}")    

    num_params = sum(p.numel() for p in proxy.train_model.parameters() if p.requires_grad)
    print(f"[info] Main{data_size} parameters: {num_params / 1e6:.2f}M")