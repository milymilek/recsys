import torch
from torch import nn

from layers import EmbeddingNet


class MF(nn.Module):
    def __init__(self, feature_store, device):
        super(MF, self).__init__()
        self.feature_store = feature_store
        self.device = device

        self.V = EmbeddingNet(feature_store, device=device)
        self.emb_dim = self.V.embeddings['user_id'].weight.shape[1]

    def forward(self, x):
        x = self.V(x).to(torch.float)
        x = torch.sum(x[:, :self.emb_dim] * x[:, self.emb_dim:], axis=1).unsqueeze(1)
        return x