import torch
from torch.utils.data import Dataset


def collate_fn(batch):
    xs, ys = [], []
    for x, y in batch:
        xs.append(x)
        ys.append(y)
    xs = torch.cat(xs)
    ys = torch.cat(ys).to(torch.float)
    return xs, ys.unsqueeze(1)


def collate_fn_eval(batch):
    xs = []
    for x, _ in batch:
        xs.append(x)
    xs = torch.cat(xs)
    return xs


class DeepFMDataset(Dataset):
    def __init__(self, feature_store, edge_index, user_attr, item_attr, neg_sampl):
        self.edge_index = torch.tensor(edge_index) + 1
        self.user_attr = torch.tensor(user_attr.values)
        self.item_attr = feature_store.attr2tensor(item_attr, scheme='item_feat')

        self.users = self.edge_index[:, 0]
        self.items = self.edge_index[:, 1]

        self.n_users = self.user_attr.shape[0]
        self.n_items = self.item_attr.shape[0]

        self.neg_sampl = neg_sampl

    def __len__(self):
        return self.edge_index.shape[0]

    def __getitem__(self, idx):
        u_id = self.users[idx].repeat(self.neg_sampl + 1)
        i_id = torch.cat([self.items[idx].unsqueeze(0), self._approx_neg_sampl()])

        u_attr = self.user_attr[u_id - 1]
        i_attr = self.item_attr[i_id - 1]

        x = torch.column_stack((u_id, i_id, u_attr, i_attr))
        y = torch.tensor([1] + [0] * self.neg_sampl)

        return x, y

    def _approx_neg_sampl(self):
        neg_i_id = torch.randint(low=0, high=self.n_items, size=(self.neg_sampl,))
        return neg_i_id