import torch
from torch import nn


class CrossProductNet(nn.Module):
    def __init__(self, device='cpu'):
        super(CrossProductNet, self).__init__()
        self.device = device

    def forward(self, emb_x):
        print(emb_x.shape)
        raise Exception

        cross = torch.bmm(emb_x.unsqueeze(2), emb_x.unsqueeze(1))
        return torch.tensor([torch.tril(i, diagonal=-1).sum() for i in cross], device=self.device).unsqueeze(1)


class FM(nn.Module):
    def __init__(self, device='cpu'):
        super(FM, self).__init__()
        self.device = device

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        #cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)
        cross_term = 0.5 * cross_term

        return cross_term