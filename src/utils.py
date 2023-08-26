import pandas as pd
from torch_geometric.data import HeteroData

from .dataset import load_data_from_csv, load_graph, transform_graph, split_graph, init_edge_loader, get_sparse_adj_matr


def create_graph_dataset(path_relations: str, path_user_attr: str = None, path_item_attr: str = None) -> dict:
    df_relations, user_attr, item_attr = load_data_from_csv(path_relations, path_user_attr, path_item_attr)
    return df_relations, user_attr, item_attr
    #n_users, n_items = df.user_id.nunique(), df.app_id.nunique()

    data = load_graph(df_relations, user_attr, item_attr)
    data = transform_graph(data)

    train_data, val_data, _ = split_graph(data)

    train_loader = init_edge_loader(train_data, num_neighbors=[20, 10], neg_sampl=2.0, bs=1024, shuffle=True,
                                    drop_last=True)
    val_loader = init_edge_loader(val_data, num_neighbors=[20, 10], neg_sampl=2.0, bs=256, shuffle=False,
                                  drop_last=True)

    mp_matrix, val_matrix = get_sparse_adj_matr(val_data)

    dct = {
        "train_data": train_data,
        "train_loader": train_loader,
        "val_data": val_data,
        "val_loader": val_loader,
        "message_passing_matrix": mp_matrix,
        "val_matrix": val_matrix
    }

    return dct
