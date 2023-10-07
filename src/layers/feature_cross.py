import torch
from torch import nn


class CrossProductNet(nn.Module):
    def __init__(self, device='cpu'):
        super(CrossProductNet, self).__init__()
        self.device = device

    def forward(self, emb_x):
        cross = torch.bmm(emb_x.unsqueeze(2), emb_x.unsqueeze(1))
        return torch.tensor([torch.tril(i, diagonal=-1).sum() for i in cross], device=self.device).unsqueeze(1)