import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.0, neg_weight=5.0, lambda_ual=1.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = torch.tensor([pos_weight], device=device)  
        self.neg_weight = torch.tensor([neg_weight], device=device)  
        self.lambda_ual = lambda_ual

    def forward(self, logits, targets):
        # 计算 BCE 损失
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(logits.to(device), targets.to(device))
        
        # 计算 UAL
        probs = torch.sigmoid(logits)  # 将 logits 转换为概率
        uncertainty = 1 - torch.abs((2 * probs - 1)*(2 * probs - 1))  # 计算预测的不确定性
        ual_loss = torch.mean(uncertainty ** 2)  # 计算 UAL 损失
        
        # 总损失
        total_loss = bce_loss + self.lambda_ual * ual_loss
        return total_loss