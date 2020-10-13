
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class Gelu(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def upsample_add(x, y):
    _,_,H,W = y.size()
    return F.upsample(x, size=(H,W), mode='bilinear') + y