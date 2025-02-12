import torch
from torch.utils.data import Dataset
from itertools import cycle

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.cat(xs)
    ys = torch.cat(ys)
    return xs, ys


def collate_fn_eval(batch):
    xs, _ = zip(*batch)
    xs = torch.cat(xs)
    return xs


def collate_fn_shuffle(batch):
    xs, ys = zip(*batch)
    xs = torch.cat(xs)
    ys = torch.cat(ys)

    perm = torch.randperm(xs.shape[0])
    xs = xs[perm]
    ys = ys[perm]
    return xs, ys


class DeepDataset(Dataset):
    def __init__(self, feature_store, edge_index, user_attr, item_attr, neg_sampl):
        super(DeepDataset).__init__()
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
        y = torch.tensor([1] + [0] * self.neg_sampl, dtype=torch.float).view(-1, 1)

        return x, y

    def _approx_neg_sampl(self):
        neg_i_id = torch.randint(low=0, high=self.n_items, size=(self.neg_sampl,))
        return neg_i_id


class DeepDatasetIterable(Dataset):
    def __init__(self, feature_store, edge_index, user_attr, item_attr, user_batch_size, neg_sampl, shuffle=True):
        super(DeepDatasetIterable).__init__()
        self.edge_index = torch.tensor(edge_index) + 1
        self.user_attr = torch.tensor(user_attr.values)
        self.item_attr = feature_store.attr2tensor(item_attr, scheme='item_feat')

        self.n_users = self.user_attr.shape[0]
        self.n_items = self.item_attr.shape[0]

        self.user_batch_size = user_batch_size
        self.neg_sampl = neg_sampl

        self.edge_index_batches = cycle(iter(range(0, edge_index.shape[0], user_batch_size)))
        self.edge_index_len = self.edge_index.shape[0]

        if shuffle:
            self._shuffle()

    def __len__(self):
        return self.edge_index_len // self.user_batch_size + 1

    def __getitem__(self, idx):
        edge_index_batch = next(self.edge_index_batches)
        ix_start, ix_stop = edge_index_batch, min(edge_index_batch + self.user_batch_size, self.edge_index_len)
        batch_len = ix_stop - ix_start

        u_id = self.edge_index[ix_start:ix_stop, 0].repeat(self.neg_sampl + 1)
        i_id = torch.cat([self.edge_index[ix_start:ix_stop, 1], self._approx_neg_sampl(batch_len * self.neg_sampl)])
        u_attr = self.user_attr[u_id - 1]
        i_attr = self.item_attr[i_id - 1]

        x = torch.column_stack((u_id, i_id, u_attr, i_attr))
        y = torch.cat([
            torch.tensor([1] * batch_len, dtype=torch.float),
            torch.tensor([0] * (batch_len * self.neg_sampl), dtype=torch.float)
        ]).view(-1, 1)

        return x, y

    def _approx_neg_sampl(self, size):
        neg_i_id = torch.randint(low=0, high=self.n_items, size=(size,))
        return neg_i_id

    def _shuffle(self):
        self.edge_index = self.edge_index[torch.randperm(self.edge_index_len)]


class FeaturelessDatasetIterable(Dataset):
    def __init__(self, edge_index, n_users, n_items, user_batch_size, neg_sampl, shuffle=True):
        super(DeepDatasetIterable).__init__()
        self.edge_index = torch.tensor(edge_index) + 1

        self.n_users = n_users
        self.n_items = n_items

        self.user_batch_size = user_batch_size
        self.neg_sampl = neg_sampl

        self.edge_index_batches = cycle(iter(range(0, edge_index.shape[0], user_batch_size)))
        self.edge_index_len = self.edge_index.shape[0]

        if shuffle:
            self._shuffle()

    def __len__(self):
        return self.edge_index_len // self.user_batch_size + 1

    def __getitem__(self, idx):
        edge_index_batch = next(self.edge_index_batches)
        ix_start, ix_stop = edge_index_batch, min(edge_index_batch + self.user_batch_size, self.edge_index_len)
        batch_len = ix_stop - ix_start

        u_id = self.edge_index[ix_start:ix_stop, 0].repeat(self.neg_sampl + 1)
        i_id = torch.cat([self.edge_index[ix_start:ix_stop, 1], self._approx_neg_sampl(batch_len * self.neg_sampl)])

        x = torch.column_stack((u_id, i_id))
        y = torch.cat([
            torch.tensor([1] * batch_len, dtype=torch.float),
            torch.tensor([0] * (batch_len * self.neg_sampl), dtype=torch.float)
        ]).view(-1, 1)

        return x, y

    def _approx_neg_sampl(self, size):
        neg_i_id = torch.randint(low=0, high=self.n_items, size=(size,))
        return neg_i_id

    def _shuffle(self):
        self.edge_index = self.edge_index[torch.randperm(self.edge_index_len)]