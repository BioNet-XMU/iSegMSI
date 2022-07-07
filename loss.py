import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

class MaxEntropy(nn.Module):
    def __init__(self):
        super(MaxEntropy, self).__init__()

    def forward(self, c_i):

        p_i = F.softmax(c_i,dim=1)
        p_i = p_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = (p_i * torch.log(p_i)).sum()

        return ne_i