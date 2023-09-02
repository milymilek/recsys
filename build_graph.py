import argparse
import os
import pickle
import numpy as np
import pandas as pd

from src.dataset import load_graph, transform_graph, init_edge_loader, get_sparse_adj_matr


def build_graph():
    """Create graph from preprocessing outputs."""
    args = get_args()

    dir_art = args.artefact_directory
    user_col = args.user_col
    item_col = args.item_col

    with open(os.path.join(dir_art, 'data.pkl'), "rb") as f:
        data = pickle.load(f)

    train_set = data['train_set'][[user_col, item_col]].values.T
    supervision_set = data['supervision_set'][[user_col, item_col]].values.T
    valid_set = data['valid_set'][[user_col, item_col]].values.T
    user_attr = data['user_attr']
    item_attr = data['item_attr']

    train_data = load_graph(train_ei=train_set, test_ei=supervision_set, user_attr=user_attr, item_attr=item_attr)
    valid_data = load_graph(train_ei=np.concatenate([train_set, supervision_set], axis=1),
                            test_ei=valid_set, user_attr=user_attr, item_attr=item_attr)
    print(train_data)
    print("> Graph loaded")

    train_data = transform_graph(train_data)
    valid_data = transform_graph(valid_data)
    print("> Graph Transformed")

    train_loader = init_edge_loader(train_data, num_neighbors=[20, 10], neg_sampl=2.0, bs=1024, shuffle=True,
                                    drop_last=True)
    valid_loader = init_edge_loader(valid_data, num_neighbors=[20, 10], neg_sampl=2.0, bs=1024, shuffle=True,
                                    drop_last=True)
    print("> Edge loader initialized")

    mp_matrix, val_matrix = get_sparse_adj_matr(valid_data)
    print("> Sparse validation matrices created")

    graph = {
        "train_data": train_data,
        "train_loader": train_loader,
        "valid_data": valid_data,
        "valid_loader": valid_loader,
        "mp_matrix": mp_matrix,
        "val_matrix": val_matrix
    }

    with open(os.path.join(dir_art, "graph.pkl"), "wb") as f:
        pickle.dump(graph, f)

def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--artefact_directory', type=str)
    parser.add_argument('--user_col', type=str, default='user_id')
    parser.add_argument('--item_col', type=str, default='app_id')

    return parser.parse_args()


if __name__ == '__main__':
    build_graph()
