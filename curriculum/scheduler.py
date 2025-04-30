
import math 
import random 
import torch
from torch.utils.data import DataLoader, TensorDataset

class Scheduler:
    def __init__(self, train_data, scores, configs, schedule_type:str, init_beta:float, shuffle:bool ):
        super().__init__()

        self.train_data = train_data
        self.scores = scores
        self.configs = configs

        self.schedule_type = schedule_type
        self.init_beta = init_beta
        self.shuffle = shuffle

        self.sorted_idcs = self.scoreSort()
        self.sorted_score = self.scoreSort()

        self.gamma = 0.1

    def scoreSort(self):
        score = self.scores.cpu().numpy() # make sure still a tensor when passed

        pairs = list(enumerate(score))
        sorted_pairs = sorted(pairs, key=lambda pair: pair[1])
        sorted_idcs = [idc for idc, _ in sorted_pairs]

        return sorted_idcs
    
    def lyapunovReguliser(self, epoch, lambda_n):
        if len(lambda_n)<2:
            return 1.
        
        delta_lambda = lambda_n[-1] - lambda_n[-2]
        alpha = 1.

        if delta_lambda >= 0: # unstable condition => reduce speed
            alpha *= (1 - self.gamma*lambda_n[-1]/(1 + epoch))

        else: # stable condition => increase speed
            alpha *= (1 + self.gamma*lambda_n[-1]/(1 + epoch))

        return alpha

    def betaSchedule(self,epoch, alpha): # alpha is inital sampling size. 
        
        # adaptive scaling. adapt during training based on validation performance?
        # if feedback (val) stronger than scaling type, will schedule type wash out?
        E_n = epoch / self.configs.epochs # current epoch ratio
        eps = 1e-8
        if self.schedule_type == 'linear': # schedules the sampling linearly (default)
            beta_t = min(1., self.init_beta + E_n)
        if self.schedule_type == 'sigmoid':
            beta_t = (1 + math.exp(-alpha * E_n))**-1 
        if self.schedule_type == 'tanh':
            beta_t = 0.5*math.tanh(alpha * E_n) + 0.5
        if self.schedule_type == 'log':
            beta_t = min(1., math.log(alpha * E_n + eps))
        if self.schedule_type == 'exp':
            beta_t = min(1., math.exp(alpha * E_n) - 1)

        # sampling %
        cutoff = int(beta_t * len(self.sorted_score))
        sample_idcs = self.sorted_idcs[:cutoff] # sample from sorted indicies 

        if self.shuffle:
            random.shuffle(sample_idcs) # shuffle idcs in sample
        return sample_idcs

    def seqentialBatch(self, epoch, alpha):
        sampled_idcs = self.betaSchedule(epoch, alpha)
        subset = torch.utils.data.Subset(self.train_data, sampled_idcs)
        
        train_loader = DataLoader(
            subset, 
            batch_size=self.configs.batch_size,
            )
        return train_loader