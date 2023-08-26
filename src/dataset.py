import numpy as np
import pandas as pd
import pickle
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.utils import to_scipy_sparse_matrix


def split_by_time(df: pd.DataFrame, col: str, split_point=None):
    """
    Sort and split dataframe into train and test sets on given split point.

    Args:
        df: Dataset containing `col`
        split_date: Split date in format `yyyy-mm-dd`
        col: Name of column with dates

    Returns:
        Tuple of (train_set, test_set) where train contains rows before `date`
    """
    df_sorted = df.sort_values(by=col)

    if isinstance(split_point, str):
        mask = df_sorted[col] <= split_point
    elif isinstance(split_point, float):
        mask = np.concatenate([np.ones(int(df_sorted.shape[0] * split_point)),
                               np.zeros(df_sorted.shape[0] - int(df_sorted.shape[0] * split_point))]).astype(bool)

    df_train = df_sorted[mask]
    df_test = df_sorted[~mask]

    return df_train, df_test


def filter_test(df_train, df_test, user_col, item_col):
    all_users, all_items = df_train[user_col].unique(), df_train[item_col].unique()
    df_test_filter = df_test[(df_test[user_col].isin(all_users)) & (df_test[item_col].isin(all_items))]

    return df_test_filter



def load_data_from_csv(path_relations: str, path_user_attr: str = None, path_item_attr: str = None):
    """
    Loads data from a CSV file into a Pandas DataFrame.
    Csv file requirements:
        - `user_id` - int
        - `app_id` - int
        - `is_recommended` - int [0/1]

    Parameters:
    - path (str): The file path of the CSV file to load.

    Returns:
    - df (pd.DataFrame): The loaded data as a Pandas DataFrame.
    """
    relations = pd.read_csv(path_relations, index_col=[0])[['user_id', 'app_id']].values

    if path_user_attr:
        with open(path_user_attr, 'rb') as f:
            user_attr = pickle.load(f)
    else:
        user_attr = np.zeros(relations[:, 0].nunique())

    if path_item_attr:
        with open(path_item_attr, 'rb') as f:
            item_attr = pickle.load(f)
    else:
        item_attr = np.zeros(relations[:, 1].nunique())

    return relations, user_attr, item_attr


def load_graph(relations: np.array, user_attr: np.array, item_attr: np.array) -> HeteroData:
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

    data['user'].x = torch.from_numpy(user_attr)
    data['user'].n_id = torch.arange(user_attr.shape[0])

    data['app'].x = torch.from_numpy(item_attr)
    data['app'].n_id = torch.arange(item_attr.shape[0])

    edge_index = torch.from_numpy(relations)
    edge_label = torch.ones(relations.shape[1], dtype=torch.long)

    data['user', 'recommends', 'app'].edge_index = edge_index
    data['user', 'recommends', 'app'].edge_label = edge_label

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


def split_graph(data: HeteroData) -> HeteroData:
    random_split = T.RandomLinkSplit(
        num_val=0.3,
        num_test=0.0,
        add_negative_train_samples=False,
        neg_sampling_ratio=2.0,
        disjoint_train_ratio=0.3,
        edge_types=('user', 'recommends', 'app'),
        rev_edge_types=('app', 'rev_recommends', 'user')
    )

    return random_split(data)


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


def get_sparse_adj_matr(data):
    # Extract sparse adjacency matrix with message passing edges
    mp_edges = data['user', 'recommends', 'app']['edge_index']
    mp_matrix = to_scipy_sparse_matrix(mp_edges, num_nodes=n_users).tocsr()

    # Extract sparse adjacency matrix with validation edges
    true_mask = data['user', 'recommends', 'app']['edge_label'].nonzero().flatten()
    val_edges = data['user', 'recommends', 'app']['edge_label_index'][:, true_mask]
    val_matrix = to_scipy_sparse_matrix(val_edges, num_nodes=n_users).tocsr()

    return mp_matrix, val_matrix


