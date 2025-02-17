import torch
from torch import nn

# from features.features import SparseFeat


# class EmbeddingNet(nn.Module):
#     def __init__(self, feature_store, device="cpu"):
#         super(EmbeddingNet, self).__init__()
#         self.device = device
#         self._sparse_features = feature_store.get_emb_features()

#         _embeddings = {
#             feat.name: nn.Embedding(feat.num_emb + 1, feature_store.emb_dims[feat.name], padding_idx=0, device=self.device)
#             for feat in self._sparse_features
#         }
#         self.embeddings = nn.ModuleDict(_embeddings)

#     def forward(self, x):
#         x_emb = []
#         for name, embed_matrix in self.embeddings.items():
#             feat = [i for i in self._sparse_features if i.name == name][0]
#             x_feat = x[:, feat.index[0] : feat.index[1]].to(torch.long)
#             emb = embed_matrix(x_feat)
#             emb_agg = torch.mean(emb, axis=1)
#             x_emb.append(emb_agg)
#         x_emb = torch.cat(x_emb, axis=1)
#         return x_emb


class EmbeddingNet(nn.Module):
    def __init__(self, feature_mapping, cat_dim, device="cpu"):
        super(EmbeddingNet, self).__init__()
        self.device = device
        self._sparse_features = feature_mapping.varlen

        _embeddings = {
            name: nn.Embedding(feat["num_emb"] + 1, cat_dim, padding_idx=0, device=self.device)
            for name, feat in self._sparse_features.items()
        }
        self.embeddings = nn.ModuleDict(_embeddings)

    def forward(self, x):
        x_emb = []
        for name, embed_matrix in self.embeddings.items():
            feat = self._sparse_features[name]
            x_feat = x[:, feat["index"]].to(torch.long)
            emb = embed_matrix(x_feat)
            emb_agg = torch.mean(emb, axis=1)
            x_emb.append(emb_agg)
        x_emb = torch.cat(x_emb, axis=1)
        return x_emb
