import torch
import torch.nn as nn

class Scheduler(nn.Module):
    def __init__(self, model, x_data, y_labels, criterion, entropy_scores=None, device='cpu'):
        super(Scheduler, self).__init__()
        self.model = model  # proxy model
        self.x_data = x_data
        self.y_labels = y_labels
        self.criterion = criterion
        self.device = device

        self.loss_log = {}  # sample_id → [loss_t0, ..., loss_tN]
        self.irreducible_scores = None
        self.entropy_scores = entropy_scores
        self.combined_scores = None

    def TrainProxy(self, epochs=10, lr=1e-3):

        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        for epoch in range(epochs):
            for i in range(len(self.x_data)):
                x = self.x_data[i].unsqueeze(0).to(self.device)
                y = self.y_labels[i].unsqueeze(0).to(self.device)

                out = self.model(x)
                loss = self.criterion(out, y)

                # Log loss
                if i not in self.loss_log:
                    self.loss_log[i] = []
                self.loss_log[i].append(loss.item())

                # Backprop
                optim.zero_grad()
                loss.backward()
                optim.step()

    def CalcIrreducibleLoss(self):

        self.irreducible_scores = {
            i: self.loss_log[i][0] - self.loss_log[i][-1] 
            for i in self.loss_log}
        
    def CompCombinedScore(self, alpha=0.5):

        if self.entropy_scores is None or self.irreducible_scores is None:
            raise ValueError("Need both entropy and irreducible scores")
        
                # Normalize both (min-max)
        irr_vals = torch.tensor(list(self.irreducible_scores.values()))
        ent_vals = torch.tensor(list(self.entropy_scores.values()))
        irr_norm = (irr_vals - irr_vals.min()) / (irr_vals.max() - irr_vals.min() + 1e-8)
        ent_norm = (ent_vals - ent_vals.min()) / (ent_vals.max() - ent_vals.min() + 1e-8)

        self.combined_scores = {
            i: alpha * ent_norm[i].item() + (1 - alpha) * irr_norm[i].item()
            for i in self.irreducible_scores
        }


    def get_curriculum_order(self, strategy='irreducible', ascending=True):

        if strategy == 'irreducible':
            scores = self.irreducible_scores
        elif strategy == 'entropy':
            scores = self.entropy_scores
        elif strategy == 'combined':
            scores = self.combined_scores
        else:
            raise ValueError("Unknown strategy. Use 'irreducible', 'entropy', or 'combined'.")

        return sorted(scores, key=scores.get, reverse=not ascending)

    def sample_batch(self, batch_size, step=None, strategy='irreducible'):

        order = self.get_curriculum_order(strategy=strategy)
        if step is not None:
            order = order[:step]  # narrow to easiest samples

        indices = torch.randperm(len(order))[:batch_size]
        chosen = [order[i] for i in indices]
        x_batch = torch.stack([self.x_data[i] for i in chosen])
        y_batch = torch.stack([self.y_labels[i] for i in chosen])
        return x_batch.to(self.device), y_batch.to(self.device)
        

