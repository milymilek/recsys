import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np


def collate_fn(batch):
    edge_index, ripple_sets = zip(*batch)
    edge_index = torch.tensor(np.array(edge_index))

    rs_pad_tensor = []
    for i in range(2):
        rs = [torch.tensor(rs[i]) for rs in ripple_sets]
        rs_pad_tensor.append(pad_sequence(rs, batch_first=True, padding_value=0))

    return edge_index, rs_pad_tensor


class RippleDataset(Dataset):
    def __init__(self, edge_index, edge_label, ripple_sets):
        super(RippleDataset).__init__()
        self.edge_index = edge_index
        self.edge_label = edge_label
        self.df_ripple_set1 = ripple_sets[0]
        self.df_ripple_set2 = ripple_sets[1]

        self.n_users = ripple_sets[0].index.get_level_values(0).nunique()

    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        el = self.edge_label.loc[idx].values
        ripple1 = self.df_ripple_set1.loc[el[0]]
        ripple2 = self.df_ripple_set2.loc[el[0]]

        ripple1 = ripple1.sample(n=min(750, ripple1.shape[0])).values
        ripple2 = ripple2.sample(n=min(750, ripple2.shape[0])).values

        return el, [ripple1, ripple2]