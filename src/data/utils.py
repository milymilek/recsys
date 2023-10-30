from typing import List

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from scipy.sparse import csr_matrix

from data.dataframe import DataFrame, SplitDataFrame, IDataFrame
from data.datastore import DataStore


def split_by_time(ds: DataStore, cols: List[str], supervision_ratio: float=0.2, validation_ratio: float=0.3) -> (IDataFrame, List):
    """
    Sort and split dataframe into train and test sets on given split point.

    Args:
        df: Dataset containing `col`
        split_date: Split date in format `yyyy-mm-dd`
        col: Name of column with dates

    Returns:
        Tuple of (train_set, test_set) where train contains rows before `date`
    """
    cols = cols[0]
    df = ds.dataframe.repr_df()

    df_sorted = df.sort_values(by=cols)
    split_training = 1.0 - supervision_ratio - validation_ratio
    training_split_point = int(df_sorted.shape[0] * split_training)
    supervision_split_point = int(df_sorted.shape[0] * (split_training + supervision_ratio))

    df_train = df_sorted[:training_split_point]
    df_supervision = df_sorted[training_split_point:supervision_split_point]
    df_valid = df_sorted[supervision_split_point:]

    split_df = SplitDataFrame(train=df_train, supervision=df_supervision, valid=df_valid)
    return split_df, []


def filter_set(df, df_train, user_col, item_col):
    all_users, all_items = df_train[user_col].unique(), df_train[item_col].unique()
    df_filter = df[(df[user_col].isin(all_users)) & (df[item_col].isin(all_items))]

    return df_filter


def load_graph(train_ei: np.array, test_ei: np.array, user_attr: np.array, item_attr: np.array) -> HeteroData:
    """
    Loads a graph data structure from a pandas DataFrame.

    Parameters:
        - df (pd.DataFrame): The input DataFrame containing the graph data.

    Returns:
        - HeteroData: A heterogeneous graph data object representing the input graph.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'user_id': [1, 2, 3], 'app_id': [4, 5, 6], 'is_recommended': [1, 0 ,1]})
        >>> graph = load_graph(df)
    """
    data = HeteroData()

    data['user'].x = torch.from_numpy(user_attr).to(torch.float32)
    data['user'].n_id = torch.arange(user_attr.shape[0])

    data['app'].x = torch.from_numpy(item_attr).to(torch.float32)
    data['app'].n_id = torch.arange(item_attr.shape[0])

    data['user', 'recommends', 'app'].edge_index = torch.from_numpy(train_ei).to(torch.int64)
    data['user', 'recommends', 'app'].edge_label_index = torch.from_numpy(test_ei).to(torch.int64)
    data['user', 'recommends', 'app'].edge_label = torch.ones(test_ei.shape[1], dtype=torch.int64)

    return data


def transform_graph(data: HeteroData) -> HeteroData:
    """
    Applies a transformation to a heterogeneous graph data object.

    Parameters:
        data: The input graph data object to be transformed.

    Returns:
        HeteroData: A new heterogeneous graph data object resulting from the transformation.

    Example:
        >>> transformed_data = transform_graph(data)
    """
    transform = T.Compose([T.ToUndirected()])
    return transform(data)


def init_edge_loader(data: HeteroData, **kwargs) -> NeighborLoader:
    """
    Initializes a neighbor loader for edge-based data in a heterogeneous graph.
    Firstly we sample `batch_size` edges and then sample at most `num_neighbors[0]`
    neighboring edges at first hop and at most `num_neighbors[1]` at second hop.
    Value returned by next(iter(loader)) is a subgraph of `data` graph containing
    only sampled edges and congruent nodes.

    Args:
        data (HeteroData): The input heterogeneous graph data object.
        **kwargs: Additional keyword arguments for configuring the loader.

    Returns:
        NeighborLoader: A neighbor loader for the specified edge-based data.

    Example:
        >>> loader = init_edge_loader(data, num_neighbors=5, neg_sampl=0.2, bs=32, shuffle=True)
    """

    eli = data['user', 'recommends', 'app'].edge_label_index
    el = data['user', 'recommends', 'app'].edge_label

    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=kwargs['num_neighbors'],
        neg_sampling_ratio=kwargs['neg_sampl'],
        edge_label_index=(('user', 'recommends', 'app'), eli),
        edge_label=el,
        batch_size=kwargs['bs'],
        shuffle=kwargs['shuffle'],
    )
    return loader


def tabular2csr(train, valid, supervision=None):
    if supervision is not None:
        train = np.concatenate([train, supervision], axis=1)

    n_users, n_items = np.unique(train[0]).size, np.unique(train[1]).size

    train_csr = csr_matrix((np.ones_like(train[0]), (train[0], train[1])), shape=(n_users, n_items))
    valid_csr = csr_matrix((np.ones_like(valid[0]), (valid[0], valid[1])), shape=(n_users, n_items))

    return train_csr, valid_csr


