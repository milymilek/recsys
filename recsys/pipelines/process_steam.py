import argparse
import os
import pickle
from typing import List

import numpy as np
import pandas as pd

from data.datastore import DataStore
from data.dataframe import DataFrame, SplitDataFrame, IDataFrame
from data.utils import split_by_time, filter_set, tabular2csr
from preprocessing.workflow import Workflow
from preprocessing.preprocess import sparse, no_feat, process_num, process_cat, process_multilabel

ITEM_FEATS = {
    "CAT": ['win', 'mac', 'linux', 'steam_deck', 'rating', ],
    "NUM": ["user_reviews", "price_final", "price_original", "discount",],
    "MULTILABEL": ['tags',]
}

USER_FEATS = {
    "NUM": ["products", "reviews",]
}


@sparse
def apply_map(ds: DataStore, cols: List[str], **kwargs) -> (IDataFrame, List):
    move_col_to_idx = kwargs.get('move_col_to_idx')
    cols = cols[0]
    mapping = ds.mapping[cols]
    df = ds.dataframe

    def apply_one(df: pd.DataFrame):
        df[cols] = df[cols].map(mapping)
        df = df.dropna()
        df[cols] = df[cols].astype('int64')
        df = df.sort_values(by=[cols])
        df = df.reset_index(drop=True)
        return df

    df.apply(apply_one)
    if move_col_to_idx:
        df.df.set_index(cols, drop=True, inplace=True)
    cols = list(zip([cols], [None] * 1))
    return df, cols


@no_feat
def apply_map_no_feat(ds: DataStore, cols: List[str], **kwargs) -> (IDataFrame, List):
    move_col_to_idx = kwargs.get('move_col_to_idx')
    cols = cols[0]
    mapping = ds.mapping[cols]
    df = ds.dataframe

    def apply_one(df: pd.DataFrame):
        df[cols] = df[cols].map(mapping)
        df = df.dropna(subset=cols)
        df[cols] = df[cols].astype('int64')
        df = df.sort_values(by=[cols])
        df = df.reset_index(drop=True)
        if move_col_to_idx:
            df = df.set_index(cols, drop=True)

        return df

    df.apply(apply_one)
    cols = list(zip([cols], [None] * 1))
    return df, cols


def filter_neg_relations(ds: DataStore, cols: List[str], **kwargs) -> (IDataFrame, List):
    """Filter negative relations to stick to PyG convention"""
    cols = cols[0]
    df = ds.dataframe.repr_df()

    df[cols] = df[cols].astype(float)
    df = df[df[cols] == 1.0]

    df = DataFrame(df)
    return df, []


def filter_non_train_relations(ds: DataStore) -> (IDataFrame, List):
    df = ds.dataframe
    df.supervision = filter_set(df=df.supervision, df_train=df.train, user_col="user_id", item_col="app_id")
    df.valid = filter_set(df=df.valid, df_train=df.train, user_col="user_id", item_col="app_id")

    df = SplitDataFrame(df.train, df.supervision, df.valid)
    return df, []


def create_map(ds: DataStore, cols: List[str], **kwargs) -> dict:
    def create_map_col(df: pd.DataFrame, col: str):
        idx = df[col].unique()
        new_idx = np.arange(idx.size)
        return {i: ni for i, ni in zip(idx, new_idx)}

    train_df = ds.dataframe.repr_df()
    dict = {c: create_map_col(train_df, c) for c in cols}
    return dict


def process():
    """Prepare Steam dataset to training and evaluation procedures."""
    args = get_args()

    dir = args.directory
    dir_art = dir + "/" + args.artefact_directory
    os.makedirs(dir_art, exist_ok=True)

    supervision_ratio = args.supervision
    validation_ratio = args.validation

    # Read data
    relations_ds = DataStore(os.path.join(dir, 'recommendations.csv'), read_method='csv')
    users_ds = DataStore(os.path.join(dir, 'users.csv'), read_method='csv')
    items_ds = DataStore(os.path.join(dir, 'items.csv'), read_method='csv')
    print("> Data read")

    workflow_relations = Workflow(pipe=[
        (filter_neg_relations, {"cols": ["is_recommended"]}),
        (split_by_time, {"cols": ["date"], "supervision_ratio": supervision_ratio, "validation_ratio": validation_ratio}),
        (filter_non_train_relations, {}),
        (create_map, {"cols": ["user_id", "app_id"], "map_func": True}),
        (apply_map, {"cols": ["user_id"]}),
        (apply_map, {"cols": ["app_id"]}),
    ])
    workflow_relations.fit(relations_ds)
    workflow_relations.transform()
    print("> Workflow Relations finished")

    items_ds.mapping = relations_ds.mapping

    workflow_items = Workflow(pipe=[
        (apply_map_no_feat, {"cols": ["app_id"], "move_col_to_idx": True}),
        (process_num, {"cols": ITEM_FEATS['NUM']}),
        (process_cat, {"cols": ITEM_FEATS['CAT']}),
        (process_multilabel, {"cols": ITEM_FEATS['MULTILABEL']})
    ])
    workflow_items.fit(items_ds)
    workflow_items.transform()
    print("> Workflow Items finished")

    users_ds.mapping = relations_ds.mapping

    workflow_users = Workflow(pipe=[
        (apply_map_no_feat, {"cols": ["user_id"], "move_col_to_idx": True}),
        (process_num, {"cols": USER_FEATS['NUM']})
    ])
    workflow_users.fit(users_ds)
    workflow_users.transform()
    print("> Workflow Users finished")

    # Create sparse csr matrix representation of tabular user-app relations
    train_csr, valid_csr = tabular2csr(
        train=workflow_relations.ds.dataframe.train.values.T,
        supervision=workflow_relations.ds.dataframe.supervision.values.T,
        valid=workflow_relations.ds.dataframe.valid.values.T
    )
    print("> Sparse adjacency matrices created")

    data = {
        "relations_datastore": workflow_relations.ds,
        "users_datastore": workflow_users.ds,
        "items_datastore": workflow_items.ds
    }
    matrix = {
        "train_csr": train_csr,
        "valid_csr": valid_csr
    }

    with open(os.path.join(dir_art, 'data.pkl'), 'wb') as f:
        pickle.dump(data, f)
    with open(os.path.join(dir_art, 'matrix.pkl'), 'wb') as f:
        pickle.dump(matrix, f)


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str)
    parser.add_argument('--artefact_directory', type=str)
    parser.add_argument('--supervision', type=float, default=0.2)
    parser.add_argument('--validation', type=float, default=0.3)

    return parser.parse_args()


if __name__ == '__main__':
    process()
