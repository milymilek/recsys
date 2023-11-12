import argparse
import os
import pickle
import numpy as np
import pandas as pd

from data.utils import load_graph, transform_graph


def build_graph():
    """Create graph from preprocessing outputs."""
    args = get_args()

    dir_art = args.artefact_directory

    with open(os.path.join(dir_art, 'data.pkl'), "rb") as f:
        data = pd.read_pickle(f)

    train_set = data['relations_datastore'].dataframe.train.values.T
    supervision_set = data['relations_datastore'].dataframe.supervision.values.T
    valid_set = data['relations_datastore'].dataframe.valid.values.T
    item_attr = data['items_datastore'].dataframe.df.values
    user_attr = data['users_datastore'].dataframe.df.values

    train_data = load_graph(train_ei=train_set, test_ei=supervision_set, user_attr=user_attr, item_attr=item_attr)
    valid_data = load_graph(train_ei=np.concatenate([train_set, supervision_set], axis=1),
                            test_ei=valid_set, user_attr=user_attr, item_attr=item_attr)
    print("> Graph loaded")

    train_data = transform_graph(train_data)
    valid_data = transform_graph(valid_data)
    print("> Graph Transformed")

    graph = {
        "train_data": train_data,
        "valid_data": valid_data
    }

    with open(os.path.join(dir_art, "graph.pkl"), "wb") as f:
        pickle.dump(graph, f)
    print("> Graph Saved")


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--artefact_directory', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    build_graph()
