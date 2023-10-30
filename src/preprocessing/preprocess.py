import ast
from typing import List

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from data.dataframe import DataFrame, IDataFrame
from data.datastore import DataStore
from features.features import SparseFeat, DenseFeat, VarlenFeat


def sparse(func):
    def wrapper(*args, **kwargs):
        df, cols = func(*args, **kwargs)
        cols = [SparseFeat(name=c, map=map, num_emb=df.repr_df()[c].nunique(), index=None) for c, map in cols]
        return df, cols
    return wrapper


def dense(func):
    def wrapper(*args, **kwargs):
        df, cols = func(*args, **kwargs)
        cols = [DenseFeat(name=c, index=None) for c in cols]
        return df, cols
    return wrapper


def varlen(func):
    def wrapper(*args, **kwargs):
        df, cols = func(*args, **kwargs)
        cols = [VarlenFeat(name=name, vals=c, num_emb=len(c), max_len=df.repr_df()[c].sum(axis=1).max(), index=None) for name, c in cols]
        return df, cols
    return wrapper


@dense
def process_num(ds: DataStore, cols, **kwargs) -> (IDataFrame, List):
    df = ds.dataframe.repr_df()
    df[cols] = df[cols].astype('float32')
    df = DataFrame(df)
    return df, cols


@sparse
def process_cat(ds: DataStore, cols, **kwargs) -> (IDataFrame, List):
    df = ds.dataframe.repr_df()
    le = LabelEncoder()
    maps = []
    for c in cols:
        df[c] = le.fit_transform(df[c])
        maps.append({k: v for k,v in enumerate(le.classes_)})
    df = DataFrame(df)
    cols = list(zip(cols, maps))
    return df, cols


@varlen
def process_multilabel(ds: DataStore, cols, **kwargs) -> (IDataFrame, List):
    df = ds.dataframe.repr_df()
    cols = cols[0]

    mlb = MultiLabelBinarizer()
    mlb_arr = mlb.fit_transform(df[cols].apply(ast.literal_eval))
    mlb_df = pd.DataFrame(mlb_arr, columns=mlb.classes_, dtype='int64')
    df = pd.concat([df, mlb_df], axis=1)
    df = df.drop(columns=cols)

    df = DataFrame(df)
    return df, [(cols, list(mlb.classes_))]


