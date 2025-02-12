import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1, batch_norm=True):
        super(MLP, self).__init__()

        _layers = []
        dims = [input_dim] + hidden_dim
        for in_dim, out_dim in zip(dims, dims[1:]):
            _layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                _layers.append(nn.BatchNorm1d(out_dim))
            _layers.append(nn.ReLU())
            _layers.append(nn.Dropout(p=dropout))
        _layers.append(nn.Linear(dims[-1], 1))

        self.layers = nn.Sequential(*_layers)

    def forward(self, emb_x):
        return self.layers(emb_x)