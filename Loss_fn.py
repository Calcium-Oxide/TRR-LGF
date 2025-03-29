import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.0, neg_weight=1.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = torch.tensor([pos_weight], device=device)  
        self.neg_weight = torch.tensor([neg_weight], device=device)  

    def forward(self, logits, targets):
        return nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(logits.to(device), targets.to(device))
    
