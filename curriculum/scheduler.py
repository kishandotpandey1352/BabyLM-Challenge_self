
import math 
import random 
import torch
from torch.utils.data import DataLoader, TensorDataset

class Scheduler:
    def __init__(self, train_data, scores, configs, gamma, schedule_type:str, shuffle:bool ):
        super().__init__()

        self.train_data = train_data
        self.scores = scores
        self.configs = configs

        self.schedule_type = schedule_type
        self.shuffle = shuffle

        self.sorted_idcs = self.scoreSort()

        self.steps=0
        self.gamma=gamma

    def scoreSort(self):
        score = self.scores.cpu().numpy() # make sure still a tensor when passed

        pairs = list(enumerate(score))
        sorted_pairs = sorted(pairs, key=lambda pair: pair[1])
        sorted_idcs = [idc for idc, _ in sorted_pairs]

        return sorted_idcs
    
    def lyapunovReguliser(self,lambda_n):
        if len(lambda_n)<2:
            return 1.
        
        delta_lambda = lambda_n[-1] - lambda_n[-2]
        alpha = 1.

        if delta_lambda >= 0: # unstable condition => reduce speed
            alpha *= (1 - self.gamma*lambda_n[-1]/(1 + self.steps))

        else: # stable condition => increase speed
            alpha *= (1 + self.gamma*lambda_n[-1]/(1 + self.steps))

        return alpha

    def betaSchedule(self, alpha): 

        print(f"[{self.schedule_type}]")
        E_n = self.steps / self.configs.total_steps # current epoch ratio
        eps = 1e-8
        if self.schedule_type == 'linear': # schedules the sampling linearly (default)
            beta_t = min(1., alpha * E_n)
        if self.schedule_type == 'sigmoid':
            beta_t = (1 + math.exp(-alpha * E_n))**-1 
        if self.schedule_type == 'tanh':
            beta_t = 0.5*math.tanh(alpha * E_n) + 0.5
        if self.schedule_type == 'log':
            beta_t = min(1., math.log(alpha * E_n + eps))
        if self.schedule_type == 'exp':
            beta_t = min(1., math.exp(alpha * E_n) - 1 + eps)
        
        self.current_beta = beta_t
        self.prct_seen = beta_t * 100
        # sampling %
        cutoff = max(1, int(beta_t * len(self.sorted_idcs))) 
        sample_idcs = self.sorted_idcs[:cutoff] # sample from sorted indicies 

        if self.shuffle:
            random.shuffle(sample_idcs) # shuffle idcs in sample
        return sample_idcs

    def seqentialBatch(self, alpha):
        sampled_idcs = self.betaSchedule(alpha)
        subset = torch.utils.data.Subset(self.train_data, sampled_idcs)
        self.steps+=1
        train_loader = DataLoader(
            subset, 
            batch_size=self.configs.batch_size,
            )
        return train_loader
