

import torch 
import torch.nn as nn
import scheduler

class Scheduler(nn.Module):
    def __init__(self, model, x_data, y_labels, criterion, device='cpu'):
        super(Scheduler, self).__init__()
        self.model = model # proxy model 
        self.x_data = x_data
        self.y_labels = y_labels
        self.criterion = criterion
        self.device = device

        self.loss_log = {} # sample id -> [loss t0, ..., loss tn]
        self.irreducible_scores = None


    def TrainProxy(self, epochs=10, lr=1e-3):
        """
        Train simple proxy model over epochs and log the loss per sample
        """

        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        for i in range(epochs):
            x = x.unsqueeze(0).to(self.device)
            y = y.unsqueeze(0).to(self.device)

            out = self.model(x)
            loss = self.criterion(out, y)

            # Log loss 
            if i not in self.loss_log:
                self.loss_log[i] = []
            self.loss_log[i].append(loss.item())

            # Back prop
            optim.zero_grad()
            loss.backward()
            optim.step()

    def calc_IrreducibleLoss(self):
        """
        Calculates the irreducible score as early_loss - late_loss per sample
        """

        self.irreducible_scores = {
            i: self.loss_log[i][0] - self.loss_log[i][-1] for i in self.loss_log
        }

    def  
    


