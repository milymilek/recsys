import torch
from torch_geometric import nn


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = nn.GCNConv((hidden_channels, hidden_channels), hidden_channels, normalize=True)
        self.conv2 = nn.GCNConv((hidden_channels, hidden_channels), out_channels, normalize=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
