import torch
from torch import nn

from recsys.layers import FM, MLP, EmbeddingNet

# class DeepFM(nn.Module):
#     def __init__(self, feature_store, hidden_dim, device="cpu"):
#         super(DeepFM, self).__init__()
#         self.feature_store = feature_store
#         self.device = device

#         self.V = EmbeddingNet(feature_store, device=self.device)
#         self.fm = FM(device=self.device)
#         self.dnn = MLP(feature_store.get_input_dim(), hidden_dim)

#     def forward(self, x):
#         x_dense = x[:, self.feature_store.dense_index]
#         x_sparse = self.V(x)
#         x_sparse_dense = torch.cat([x_sparse, x_dense], dim=1).to(torch.float)

#         x = self.fm(x_sparse) + self.dnn(x_sparse_dense)
#         return x


class DeepFM(nn.Module):
    def __init__(self, feature_mapping, cat_dim, hidden_dim, device="cpu"):
        super().__init__()
        self.feature_mapping = feature_mapping
        self.device = device

        # self.V = EmbeddingNet(feature_mapping, device=self.device)
        feat = feature_mapping.varlen["user_item_categories"]
        self.embedding = nn.Embedding(feat["num_emb"] + 1, cat_dim, padding_idx=0, device=self.device)
        self.fm = FM(device=self.device)
        input_dim = len(feature_mapping.dense) + len(feature_mapping.varlen) * cat_dim
        self.dnn = MLP(input_dim, hidden_dim)

    def emb(self, x):
        bs = x.shape[0]
        w = x[:, self.feature_mapping.varlen["user_item_categories"]["index"]].unsqueeze(-1)
        x_arrange = torch.arange(1, 21).repeat(bs).reshape(bs, -1)
        # print(w.shape, x_arrange.shape, self.embedding(x_arrange).shape)
        user_item_categories_emb = (self.embedding(x_arrange) * w).mean(dim=1)

        w = x[:, self.feature_mapping.varlen["item_categories"]["index"]].unsqueeze(-1)
        x_arrange = torch.arange(1, 21).repeat(bs).reshape(bs, -1)
        # print(w.shape, x_arrange.shape, self.embedding(x_arrange).shape)
        item_categories_emb = (self.embedding(x_arrange) * w).mean(dim=1)

        # print(item_categories_emb.shape)

        return torch.cat([user_item_categories_emb, item_categories_emb], dim=1)

    def forward(self, x):
        # print([i["index"] for i in self.feature_mapping.dense.values()])
        x_dense = x[:, [i["index"] for i in self.feature_mapping.dense.values()]]
        x_sparse = self.emb(x)
        x_sparse_dense = torch.cat([x_sparse, x_dense], dim=1).to(torch.float)

        x = self.fm(x_sparse) + self.dnn(x_sparse_dense)
        return x
