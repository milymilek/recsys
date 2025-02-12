import torch
from torch import nn

from layers import MLP, EmbeddingNet, FM


class DeepFM(nn.Module):
    def __init__(self, feature_store, hidden_dim, device='cpu'):
        super(DeepFM, self).__init__()
        self.feature_store = feature_store
        self.device = device

        self.V = EmbeddingNet(feature_store, device=self.device)
        self.fm = FM(device=self.device)
        self.dnn = MLP(feature_store.get_input_dim(), hidden_dim)

    def forward(self, x):
        x_dense = x[:, self.feature_store.dense_index]
        x_sparse = self.V(x)
        x_sparse_dense = torch.cat([x_sparse, x_dense], dim=1).to(torch.float)

        x = self.fm(x_sparse) + self.dnn(x_sparse_dense)
        return x

