from collections import namedtuple

import torch
from torch.nn.functional import pad


class SparseFeat(namedtuple('SparseFeat', ['name', 'map', 'num_emb', 'index', 'pad_index'])):
    def __new__(cls, name, map, num_emb, index, pad_index):
        return super(SparseFeat, cls).__new__(cls, name, map, num_emb, index, pad_index)

    def to_tensor(self, x):
        return torch.tensor([x.iloc[self.pad_index[0]:self.pad_index[1]].values]) + 1


class DenseFeat(namedtuple('DenseFeat', ['name', 'index', 'pad_index'])):
    def __new__(cls, name, index, pad_index):
        return super(DenseFeat, cls).__new__(cls, name, index, pad_index)

    def to_tensor(self, x):
        return torch.tensor([x.iloc[self.pad_index[0]:self.pad_index[1]].values])


class VarlenFeat(namedtuple('VarlenFeat', ['name', 'vals', 'num_emb', 'max_len', 'index', 'pad_index'])):
    def __new__(cls, name, vals, num_emb, max_len, index, pad_index):
        return super(VarlenFeat, cls).__new__(cls, name, vals, num_emb, max_len, index, pad_index)

    def to_tensor(self, x):
        start_idx, end_idx = self.pad_index
        nonzero_x = x.iloc[start_idx:end_idx].values.nonzero()[0]
        tensor_nonzero_x = torch.tensor(nonzero_x)
        return (pad(tensor_nonzero_x, (0, self.max_len - tensor_nonzero_x.shape[0]), value=-1) + 1).unsqueeze(1)