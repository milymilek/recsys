import numpy as np
import torch
from torch.nn.functional import pad

#from .dataset import load_data_from_csv, load_graph, transform_graph, split_graph, init_edge_loader, get_sparse_adj_matr
from src.feature_store import SparseFeat, DenseFeat

# def create_graph_dataset(path_relations: str, path_user_attr: str = None, path_item_attr: str = None) -> dict:
#     df_relations, user_attr, item_attr = load_data_from_csv(path_relations, path_user_attr, path_item_attr)
#     return df_relations, user_attr, item_attr
#     #n_users, n_items = df.user_id.nunique(), df.app_id.nunique()
#
#     data = load_graph(df_relations, user_attr, item_attr)
#     data = transform_graph(data)
#
#     train_data, val_data, _ = split_graph(data)
#
#     train_loader = init_edge_loader(train_data, num_neighbors=[20, 10], neg_sampl=2.0, bs=1024, shuffle=True,
#                                     drop_last=True)
#     val_loader = init_edge_loader(val_data, num_neighbors=[20, 10], neg_sampl=2.0, bs=256, shuffle=False,
#                                   drop_last=True)
#
#     mp_matrix, val_matrix = get_sparse_adj_matr(val_data)
#
#     dct = {
#         "train_data": train_data,
#         "train_loader": train_loader,
#         "val_data": val_data,
#         "val_loader": val_loader,
#         "message_passing_matrix": mp_matrix,
#         "val_matrix": val_matrix
#     }
#
#     return dct


def attr2tensor(features: list, attr: np.ndarray) -> torch.tensor:
    x_tensor = []
    for elem in attr:
        elem_tensor = []
        for i, f in enumerate(features):
            if isinstance(f, SparseFeat):
                if f.max_len > 1:
                    elem_i = pad(torch.tensor(elem[i]), (0, f.max_len - len(elem[i])), value=-1) + 1
                else:
                    elem_i = torch.tensor([elem[i]]) + 1
            elif isinstance(f, DenseFeat):
                elem_i = torch.tensor([elem[i]])

            elem_tensor.append(elem_i)

        elem_tensor = torch.cat(elem_tensor)
        x_tensor.append(elem_tensor)

    x_tensor = torch.stack(x_tensor)
    return x_tensor
