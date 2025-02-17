import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from metrics import precision_k, recall_k, ndcg_k
from models.gnn import GraphSAGE, GATConv, GNN
from inference import recommend_k, recommendation_relevance
from utils import write_scalars, load_model


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

    with open(os.path.join(dir_art, 'graph.pkl'), "rb") as f:
        graph = pd.read_pickle(f)
    with open(os.path.join(dir_art, 'matrix.pkl'), "rb") as f:
        matrix = pd.read_pickle(f)

    train_data = graph['train_data']
    valid_data = graph['valid_data']

    user_shape = train_data['user'].x.shape
    app_shape = train_data['app'].x.shape

    train_csr = matrix['train_csr']
    valid_csr = matrix['valid_csr']

    if model_name == "GraphSAGE":
        gnn_model = GraphSAGE(hidden_channels=32, out_channels=32)
    elif model_name == "GATConv":
        gnn_model = GATConv(hidden_channels=32, out_channels=32)

    model = load_model(
        cls=GNN,
        model_path=model_path,
        model_kwargs={
            "gnn_model": gnn_model,
            "entities_shapes": {"user": user_shape, "app": app_shape},
            "hidden_channels": 32,
            "metadata": train_data.metadata()
        },
        device=device
    )

    x_emb = model.evaluate(valid_data.to(device))
    recommendations = recommend_k(
        user_emb=x_emb['user'],
        item_emb=x_emb['app'],
        past_interactions=train_csr,
        k=max(K),
        user_batch_size=10000
    ).cpu().numpy()

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
