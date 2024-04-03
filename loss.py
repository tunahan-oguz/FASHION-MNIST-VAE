import torch
import torch.nn as nn
import numpy as np

class KLDivLoss(nn.Module):

    def __init__(self, scale : float = 784.) -> None:
        """
        Input shape: B x D
        """
        super().__init__()
        self.scale = scale
    
    def forward(self, m, gamma):
        kl_loss = -0.5 * torch.sum(1 + gamma - torch.exp(gamma) - m ** 2, dim=-1)
        kl_loss = torch.mean(kl_loss) / self.scale

        return kl_loss
    