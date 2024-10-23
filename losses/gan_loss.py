import torch
import torch.nn as nn

class GANLoss(nn.Module):
    """
    Real label is 1.0 and fake label is 0. 
    """
    def __init__(self):
        super(GANLoss, self).__init__()
        
        self.loss = nn.MSELoss()
            
    def get_target_tensor(self, prediciton, is_real):
        if is_real:
            return torch.ones_like(prediciton)
        else:
            return torch.zeros_like(prediciton)
        
    def forward(self, prediction, is_real):
        target = self.get_target_tensor(prediction, is_real)
        loss = self.loss(prediction, target)
        
        return loss