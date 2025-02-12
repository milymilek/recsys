import torch
from torch import nn
import torch.nn.functional as F


class RippleNet(nn.Module):
    def __init__(self, emb_dim, n_relations, n_entities):
        super(RippleNet, self).__init__()
        self.relation_emb = nn.Embedding(n_relations, emb_dim * emb_dim, padding_idx=0)
        self.entity_emb = nn.Embedding(n_entities, emb_dim, padding_idx=0)

        self.emb_dim = emb_dim
        self.n_relations = n_relations
        self.n_items = n_entities

    def forward(self, edge_index, ripple_sets):
        item_emb = self.entity_emb(edge_index[:, 1]).unsqueeze(-1)
        u = torch.zeros_like(item_emb)
        for ripple_set in ripple_sets:
            R = self.relation_emb(ripple_set[:, :, 1]).view(ripple_set.shape[0], -1, self.emb_dim, self.emb_dim)
            h = self.entity_emb(ripple_set[:, :, 0]).unsqueeze(-1)
            Rh = torch.matmul(R, h).squeeze(-1)
            vRh = torch.matmul(Rh, item_emb).squeeze(-1)
            p = F.softmax(vRh, dim=1).unsqueeze(-1)
            t = self.entity_emb(ripple_set[:, :, 0])
            u += torch.sum(p * t, axis=1).unsqueeze(-1)

        return torch.sum(u * item_emb, axis=1)