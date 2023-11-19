import torch
from torch_geometric import nn


class GATConv(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = nn.GATConv((hidden_channels, hidden_channels), hidden_channels, add_self_loops=False, normalize=True)
        self.conv2 = nn.GATConv((hidden_channels, hidden_channels), out_channels, add_self_loops=False, normalize=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
