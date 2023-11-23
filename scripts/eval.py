import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.deep import DeepDataset, collate_fn_eval
from features.store import FeatureStore
from inference.inference import remove_past_interactions
from metrics import precision_k, recall_k, ndcg_k
from models import DeepFM, NCF, MF
from utils import load_model, write_scalars


def collate_fn(batch):
    return torch.cat(batch)


class IterableDatasetTest(IterableDataset):
    def __init__(self, feature_store, user_attr, item_attr, user_batch_size):
        super(IterableDatasetTest).__init__()
        self.user_attr = torch.tensor(user_attr.values)
        self.item_attr = feature_store.attr2tensor(item_attr, scheme='item_feat')

        self.n_users = user_attr.shape[0]
        self.n_items = item_attr.shape[0]

        self.user_batch_size = user_batch_size

    def get_batch_data(self, batch):
        u_id = torch.from_numpy(
            np.repeat(np.arange(batch, min((batch + self.user_batch_size), self.n_users)), self.n_items)) + 1
        i_id = torch.arange(self.n_items).repeat(u_id.shape[0] // self.n_items, 1).flatten() + 1
        u_attr = self.user_attr[u_id - 1]
        i_attr = self.item_attr[i_id - 1]

        return torch.column_stack((u_id, i_id, u_attr, i_attr))

    def __len__(self):
        return self.n_users // self.user_batch_size + 1

    def __iter__(self):
        for batch in range(0, self.n_users, self.user_batch_size):
            yield self.get_batch_data(batch)


class FeaturelessIterableDatasetTest(IterableDataset):
    def __init__(self, n_users, n_items, user_batch_size):
        super(IterableDatasetTest).__init__()
        self.n_users = n_users
        self.n_items = n_items

        self.user_batch_size = user_batch_size

    def get_batch_data(self, batch):
        u_id = torch.from_numpy(
            np.repeat(np.arange(batch, min((batch + self.user_batch_size), self.n_users)), self.n_items)) + 1
        i_id = torch.arange(self.n_items).repeat(u_id.shape[0] // self.n_items, 1).flatten() + 1

        return torch.column_stack((u_id, i_id))

    def __len__(self):
        return self.n_users // self.user_batch_size + 1

    def __iter__(self):
        for batch in range(0, self.n_users, self.user_batch_size):
            yield self.get_batch_data(batch)


@torch.no_grad()
def recommend_k_deep(model, dataloader, device, k=10):
    model.eval()
    preds = []
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        y_pred = model(batch)
        preds.append(y_pred)
    pred = torch.cat(preds, dim=0).sigmoid().cpu()

    return pred


def recommend_k(prob_full, past_interactions, k=10, user_batch_size=1000):
    user_batches = torch.arange(prob_full.shape[0]).split(user_batch_size)
    recommended_batches = []

    for user_batch in user_batches:
        prob = prob_full[user_batch]
        prob = remove_past_interactions(prob, user_batch, past_interactions)
        recommended_batches.append(prob.topk(k, 1)[1])

    recommendations = torch.cat(recommended_batches, 0)
    return recommendations


def recommendation_relevance(recommendations, ground_truth):
    n_users = ground_truth.shape[0]
    k = recommendations.shape[1]
    user_idx = np.repeat(np.arange(n_users), k)
    item_idx = list(recommendations.flatten().numpy())
    relevance = ground_truth[user_idx, item_idx].reshape(
        (n_users, k))

    return relevance, [True] * n_users


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description='Evaluation for graph-based recommendation system.')
    parser.add_argument('--artefact_directory', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--K', nargs="+", default=[1, 2, 5, 10, 20, 50, 100])
    parser.add_argument('--cuda', type=bool, default=False)

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

    with open(os.path.join(dir_art, 'data.pkl'), "rb") as f:
        data = pd.read_pickle(f)
    with open(os.path.join(dir_art, 'matrix.pkl'), "rb") as f:
        matrix = pd.read_pickle(f)

    item_attr = data['items_datastore'].dataframe.df
    user_attr = data['users_datastore'].dataframe.df
    scheme_relations = data['relations_datastore'].scheme
    scheme_items = data['items_datastore'].scheme
    scheme_users = data['users_datastore'].scheme

    train_csr = matrix['train_csr']
    valid_csr = matrix['valid_csr']

    relevance_mask = np.asarray((valid_csr.sum(axis=1) != 0)).ravel()
    user_attr = user_attr[relevance_mask]
    valid_csr = valid_csr[relevance_mask]


    if model_name == "DeepFM":
        feature_store = FeatureStore(scheme_relations, scheme_items, scheme_users,
                                     emb_dims={"sparse": 16, "varlen": 16})
        model_cls = DeepFM
        model_kwargs = {
            "feature_store": feature_store,
            "hidden_dim": [128, 64, 16],
            "device": device
        }
        eval_dataset = IterableDatasetTest(feature_store, user_attr, item_attr, user_batch_size=int(1e3))
    elif model_name == "NCF":
        feature_store = FeatureStore(scheme_relations, scheme_items, scheme_users,
                                     emb_dims={"sparse": 16, "varlen": 16})
        model_cls = NCF
        model_kwargs = {
            "feature_store": feature_store,
            "hidden_dim": [128, 64, 16],
            "device": device
        }
        eval_dataset = IterableDatasetTest(feature_store, user_attr, item_attr, user_batch_size=int(1e3))
    elif model_name == "MF":
        scheme_items.features, scheme_users.features = [], []
        feature_store = FeatureStore(scheme_relations, scheme_items, scheme_users,
                                     emb_dims={"sparse": 16, "varlen": 16})
        model_cls = MF
        model_kwargs = {
            "feature_store": feature_store,
            "device": device
        }
        eval_dataset = FeaturelessIterableDatasetTest(user_attr.shape[0], item_attr.shape[0], user_batch_size=int(1e3))

    eval_loader = DataLoader(eval_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn, drop_last=False)
    model = load_model(
        cls=model_cls,
        model_path=model_path,
        model_kwargs=model_kwargs,
        device=device
    )
    prob = recommend_k_deep(model, eval_loader, device)
    prob_full = prob.reshape(-1, 1231)

    recommendations = recommend_k(
        prob_full=prob_full,
        past_interactions=train_csr,
        k=max(K),
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