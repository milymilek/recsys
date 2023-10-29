import torch
from torch import nn

from src.features.features import SparseFeat


class EmbeddingNet(nn.Module):
    def __init__(self, feature_store, device='cpu'):
        super(EmbeddingNet, self).__init__()
        self.device = device
        self._sparse_features = feature_store.get_feature_type(SparseFeat)

        _embeddings = {
            feat.name: nn.Embedding(
                feat.num_emb + 1,
                feat.emb_dim,
                padding_idx=0,
                device=self.device
            ) for feat in self._sparse_features
        }
        self.embeddings = nn.ModuleDict(_embeddings)

    def forward(self, x):
        x_emb = []
        for name, embed_matrix in self.embeddings.items():
            feat = [i for i in self._sparse_features if i.name == name][0]
            x_feat = x[:, feat.pad_index[0]:feat.pad_index[1]].to(torch.long)
            emb = embed_matrix(x_feat)
            emb_agg = torch.mean(emb, axis=1)
            x_emb.append(emb_agg)
        x_emb = torch.cat(x_emb, axis=1)
        return x_emb