import torch
from torch import nn


class MF(nn.Module):
    def __init__(self, n_users, n_items, emb_size):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.u = nn.Embedding(n_users, emb_size)
        self.i = nn.Embedding(n_items, emb_size)

    def forward(self, ux, ix):
        return torch.sum(self.u(ux) * self.i(ix), dim=1)


