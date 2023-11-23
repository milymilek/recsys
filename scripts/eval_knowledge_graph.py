import argparse
import os
import pickle
import math

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.kg import RippleNet
from metrics import precision_k, recall_k, ndcg_k
from utils import write_scalars, load_model
from scripts.eval import recommend_k, recommendation_relevance


def collate_fn(batch):
    ei, rs = zip(*batch)
    return torch.cat(ei), rs[0]


class IterableRippleDataset(Dataset):
    def __init__(self, users, items, ripple_sets, user_batch_size):
        super(IterableRippleDataset).__init__()
        self.df_ripple_set1 = ripple_sets[0]
        self.df_ripple_set2 = ripple_sets[1]

        self.users = users
        self.items = items
        self.n_users = users.shape[0]
        self.n_items = items.shape[0]
        self.user_batch_size = user_batch_size

    def sample_ripple_set(self, ripple_set, batch_users):
        sample_fun = lambda x: x.sample(n=min(750, x.shape[0])).values

        i = ripple_set.index.isin(batch_users, level='user_id')
        ripple_set_samples = ripple_set[i].groupby('user_id')
        ripple_set_samples = ripple_set_samples.apply(sample_fun)
        ripple_set_samples = np.repeat(ripple_set_samples, 1231).values
        ripple_set_samples = pad_sequence([torch.tensor(i) for i in ripple_set_samples], batch_first=True,
                                          padding_value=0)

        return ripple_set_samples

    def get_batch_data(self, batch):
        u_start, u_end = batch, min(batch + self.user_batch_size, self.n_users)
        batch_users = self.users[u_start:u_end]

        u_id = torch.from_numpy(np.repeat(batch_users, self.n_items))
        i_id = torch.arange(self.n_items).repeat(u_id.shape[0] // self.n_items, 1).flatten() + 1

        ripple_set1 = self.sample_ripple_set(self.df_ripple_set1, batch_users)
        ripple_set2 = self.sample_ripple_set(self.df_ripple_set2, batch_users)

        return torch.column_stack((u_id, i_id)), [ripple_set1, ripple_set2]

    def __len__(self):
        return math.ceil(self.n_users / self.user_batch_size)

    # def __iter__(self):
    #     for batch in range(0, self.n_users, self.user_batch_size):
    #         yield self.get_batch_data(batch)

    def __getitem__(self, idx):
        batch = idx * self.user_batch_size
        return self.get_batch_data(batch)


@torch.no_grad()
def recommend_k_kg(model, dataloader, device, k=10):
    model.eval()
    preds = []
    for edge_index, ripple_sets in tqdm(dataloader):
        edge_index = edge_index.to(device)
        ripple_sets = [rs.to(device) for rs in ripple_sets]

        y_pred = model(edge_index, ripple_sets)
        preds.append(y_pred)
    pred = torch.cat(preds, dim=0).sigmoid().cpu()
    return pred


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description='Evaluation for graph-based recommendation system.')
    parser.add_argument('--artefact_directory', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--K', nargs="+", default=[1, 2, 5, 10, 20, 50, 100])
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


def evaluate():
    """Evaluation process."""
    args = get_args()

    dir_art = args.artefact_directory
    model_path = args.model_path
    log_dir = os.path.dirname(model_path)
    model_name = os.path.basename(os.path.dirname(log_dir))
    K = [int(k) for k in args.K]
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    num_workers = args.num_workers
    seed = args.seed

    torch.manual_seed(seed)

    with open(os.path.join(dir_art, 'knowledge_graph.pkl'), "rb") as f:
        knowledge_graph = pd.read_pickle(f)
    with open(os.path.join(dir_art, 'matrix.pkl'), "rb") as f:
        matrix = pd.read_pickle(f)

    valid_set = knowledge_graph["valid_set"]
    ripple_sets_valid = knowledge_graph['ripple_sets_valid']
    RELATIONS_MAP = knowledge_graph["relations_map"]
    ENTITY_MAP = knowledge_graph["entity_map"]

    train_csr = matrix['train_csr']
    valid_csr = matrix['valid_csr']
    relevance_mask = np.asarray((valid_csr.sum(axis=1) != 0)).ravel()
    valid_csr = valid_csr[relevance_mask]

    users = valid_set['user_id'].unique()
    items = np.arange(valid_csr.shape[1]) + 1

    users = users[:1000]
    valid_csr = valid_csr[:1000]

    if model_name == "RippleNet":
        model_cls = RippleNet
        model_kwargs = {
            "emb_dim": 16,
            "n_relations": 4,
            "n_entities": max(ENTITY_MAP.values())
        }
        eval_dataset = IterableRippleDataset(users, items, ripple_sets_valid, int(5))

    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False, num_workers=num_workers)
    model = load_model(
        cls=model_cls,
        model_path=model_path,
        model_kwargs=model_kwargs,
        device=device
    )
    prob = recommend_k_kg(model, eval_loader, device)
    prob_full = prob.reshape(-1, 1231)

    recommendations = recommend_k(
        prob_full=prob_full,
        past_interactions=train_csr,
        k=100,
        user_batch_size=10000
    )

    output_metrics = {"precision": [], "recall": [], "ndcg": []}
    for k in tqdm(K):
        reco_k = recommendations[:, :k]
        reco_rel, rel_mask = recommendation_relevance(reco_k, valid_csr)
        prec_k = precision_k(reco_rel, valid_csr, rel_mask, k)
        rec_k = recall_k(reco_rel, valid_csr, rel_mask, k)
        n_k = ndcg_k(reco_rel.getA(), valid_csr, rel_mask, k)
        output_metrics["precision"].append(prec_k)
        output_metrics["recall"].append(rec_k)
        output_metrics["ndcg"].append(n_k)

    writer = SummaryWriter(log_dir=log_dir)
    for i, k in enumerate(K):
        scalars = [v[i] for v in output_metrics.values()]
        write_scalars(writer=writer, names=output_metrics.keys(), scalars=scalars, step=k)

    with open(os.path.join(log_dir, 'output_metrics.pkl'), 'wb') as f:
        pickle.dump(output_metrics, f)


if __name__ == '__main__':
    evaluate()
