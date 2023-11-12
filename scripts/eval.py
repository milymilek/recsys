import argparse
import os

import numpy as np
import pandas as pd
import torch

from features.store import FeatureStore
from models import DeepFM


def load_model(model_path, model_kwargs):
    model = DeepFM(**model_kwargs)
    model.load_state_dict(torch.load(model_path))
    model = model.to(model_kwargs['device'])
    return model


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description='Evaluation for graph-based recommendation system.')
    parser.add_argument('--artefact_directory', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--cuda', type=bool, default=False)

    return parser.parse_args()


def evaluate():
    """Evaluation process."""
    args = get_args()

    dir_art = args.artefact_directory
    model_path = args.model_path
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    with open(os.path.join(dir_art, 'data.pkl'), "rb") as f:
        data = pd.read_pickle(f)
    with open(os.path.join(dir_art, 'martix.pkl'), "rb") as f:
        matrix = pd.read_pickle(f)

    train_set = data['relations_datastore'].dataframe.train.values.T
    supervision_set = data['relations_datastore'].dataframe.supervision.values.T
    valid_set = data['relations_datastore'].dataframe.valid.values.T
    item_attr = data['items_datastore'].dataframe.df
    user_attr = data['users_datastore'].dataframe.df
    scheme_relations = data['relations_datastore'].scheme
    scheme_items = data['items_datastore'].scheme
    scheme_users = data['users_datastore'].scheme

    train_csr = matrix['train_csr']
    valid_csr = matrix['valid_csr']

    train_set = np.concatenate((train_set, supervision_set), axis=1)

    feature_store = FeatureStore(scheme_relations, scheme_items, scheme_users, emb_dims={"sparse": 4, "varlen": 4})
    model = load_model(model_path, model_kwargs={"feature_store": feature_store, "hidden_dim": [128, 64], "device": device})



if __name__ == '__main__':
    evaluate()