import torch
from torch import nn

from layers import MLP, EmbeddingNet


class NCF(nn.Module):
    def __init__(self, feature_store, hidden_dim):
        super(NCF, self).__init__()
        self.feature_store = feature_store

        self.V = EmbeddingNet(feature_store)
        self.dnn = MLP(feature_store.get_input_dim(), hidden_dim)

    def forward(self, x):
        x_dense = x[:, self.feature_store.dense_index]
        x_sparse = self.V(x)
        x_sparse_dense = torch.cat([x_sparse, x_dense], dim=1).to(torch.float)

        x = self.dnn(x_sparse_dense)
        return x