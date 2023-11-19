import argparse
import os
import pickle
import numpy as np
import pandas as pd
from ast import literal_eval
from tqdm import tqdm

from data.utils import load_graph, transform_graph


RELATIONS = {"developer": "developed", "publisher": "published", "OS": "system", "tags": "tag"}
RELATIONS_MAP = {k: v for v, k in enumerate(["developed", "published", "system", "tag"])}
ENTITY_MAP = None


def read_items_data(path: str, data: dict):
    df_items = pd.read_csv(path)
    df_items_additional_info = pd.read_csv("data/steam.csv")[['appid', 'developer', 'publisher']].rename(
        columns={"appid": "app_id"})
    df_items[['win', 'mac', 'linux']] = df_items[['win', 'mac', 'linux']].astype(int)
    df_items['OS'] = df_items[['win', 'mac', 'linux']].apply(lambda row: [row.index[i] for i, v in enumerate(row) if v],
                                                             axis=1)
    df_items['tags'] = df_items['tags'].apply(literal_eval)
    df_merged = df_items.merge(df_items_additional_info, on="app_id", how="left")[
        ['app_id', 'title', 'developer', 'publisher', 'OS', 'tags']]
    df_merged['app_id'] = df_merged['app_id'].map(data['relations_datastore'].mapping['app_id'])
    df_merged = df_merged.dropna(subset='app_id').sort_values(by='app_id')
    df_merged['developer'] = df_merged['developer'].str.split(';')
    df_merged['publisher'] = df_merged['publisher'].str.split(';')
    return df_merged


def create_exploded_df(df: pd.DataFrame):
    global ENTITY_MAP
    df_exploded = []
    for k, v in RELATIONS.items():
        df_temp = df[['app_id', k]].explode(k).dropna()
        df_temp['relation'] = v
        df_temp['tail'] = df_temp[k]
        df_temp['head'] = df_temp['app_id']
        df_temp = df_temp[['head', 'relation', 'tail']]
        df_exploded.append(df_temp)
    df_exploded = pd.concat(df_exploded)
    df_exploded['head'] = df_exploded['head'] + 1
    df_exploded['relation'] = df_exploded['relation'].map(RELATIONS_MAP)
    n_entities = df_exploded['head'].max()
    ENTITY_MAP = {k: v for v, k in enumerate(df_exploded['tail'].unique(), int(n_entities)+1)}
    df_exploded['tail'] = df_exploded['tail'].map(ENTITY_MAP)
    return df_exploded.astype(int)


def get_user_item_list(edge_index: pd.DataFrame):
    tqdm.pandas()
    user_apps = edge_index.groupby("user_id").progress_apply(lambda x: x['app_id'].values).values
    return user_apps


def sample_paths(x, K=0.3):
    unique, counts = np.unique(x['tail'].values, return_counts=True)
    probabilities = counts / counts.sum()
    random_tails = np.random.choice(np.arange(counts.shape[0]), size=min(max(int(counts.shape[0] * K), 1), 3), p=probabilities)
    heads = unique[random_tails]
    return heads


def get_ripple_set1(df_exploded, user_apps, users):
    ripple_set1 = []
    n_users = len(users)
    for u in tqdm(users):
        df_ripple_set1 = df_exploded[np.isin(df_exploded.values[:, 0], user_apps[u])] \
            .groupby("head") \
            .apply(lambda x: x.sample(n=min(10, len(x)))).values
        ripple_set1.append(df_ripple_set1)

    ripple_index = [i.shape[0] for i in ripple_set1]
    ripple_index_1 = np.repeat(users, ripple_index)
    ripple_index_2 = np.concatenate([np.arange(i) for i in ripple_index])
    multiindex = pd.MultiIndex.from_arrays([ripple_index_1, ripple_index_2], names=['user_id', 'id'])

    df_ripple_set1 = pd.DataFrame(np.concatenate(ripple_set1), index=multiindex,
                                  columns=['head', 'relation', 'tail'])

    return df_ripple_set1


def get_ripple_set2(df_exploded, df_ripple_set1, users):
    ripple_set2 = []
    n_users = df_ripple_set1.index.get_level_values(0).nunique()
    for u in tqdm(users):
        dff = df_ripple_set1.loc[u]
        heads = dff.groupby('relation').apply(sample_paths).explode().values
        df_ripple_set2 = df_exploded[df_exploded['head'].isin(heads)] \
            .groupby('head') \
            .apply(lambda x: x.sample(n=min(10, len(x))))
        df_ripple_set2 = df_ripple_set2[~df_ripple_set2['tail'].isin(dff['head'])].values
        ripple_set2.append(df_ripple_set2)

    ripple_index = [i.shape[0] for i in ripple_set2]
    ripple_index_1 = np.repeat(np.arange(n_users), ripple_index)
    ripple_index_2 = np.concatenate([np.arange(i) for i in ripple_index])
    multiindex = pd.MultiIndex.from_arrays([ripple_index_1, ripple_index_2], names=['user_id', 'id'])

    df_ripple_set2 = pd.DataFrame(np.concatenate(ripple_set2), index=multiindex, columns=['head', 'relation', 'tail'])

    return df_ripple_set2


def build_knowledge_graph():
    """Create knowledge graph from preprocess_steam.py outputs"""
    args = get_args()

    dir_art = args.artefact_directory

    with open(os.path.join(dir_art, 'data.pkl'), "rb") as f:
        data = pd.read_pickle(f)

    train_set = data['relations_datastore'].dataframe.train
    supervision_set = data['relations_datastore'].dataframe.supervision
    valid_set = data['relations_datastore'].dataframe.valid

    # Data adjustment - todo: data wrangling refactor
    train_set = train_set.sort_values(by=['user_id', 'app_id']).reset_index(drop=True)
    supervision_set = supervision_set.sort_values(by=['user_id', 'app_id']).reset_index(drop=True)
    valid_set = valid_set.sort_values(by=['user_id', 'app_id']).reset_index(drop=True)

    train_set['app_id'] = train_set['app_id'] + 1
    supervision_set['app_id'] = supervision_set['app_id'] + 1
    valid_set['app_id'] = valid_set['app_id'] + 1

    train_users = supervision_set['user_id'].unique()
    valid_users = valid_set['user_id'].unique()
    print("> Data adjustment done")

    df_items = read_items_data(path="data/items.csv", data=data)
    df_exploded = create_exploded_df(df_items)
    df_exploded2 = df_exploded.copy(deep=True)
    df_exploded2[['head', 'tail']] = df_exploded2[['tail', 'head']].values
    print("> Exploded DataFrame created")

    user_item_list_train = get_user_item_list(train_set)[train_users]
    user_item_list_valid = get_user_item_list(pd.concat([train_set, supervision_set]))[valid_users]
    print("> User-Items lists created")

    ripple_sets_train = []
    ripple_sets_train.append(
        get_ripple_set1(df_exploded=df_exploded, user_apps=user_item_list_train, users=train_users)
    )
    ripple_sets_train.append(
        get_ripple_set2(df_exploded=df_exploded2, df_ripple_set1=ripple_sets_train[0], users=train_users)
    )

    ripple_sets_valid = []
    ripple_sets_valid.append(
        get_ripple_set1(df_exploded=df_exploded, user_apps=user_item_list_valid, users=valid_users)
    )
    ripple_sets_valid.append(
        get_ripple_set2(df_exploded=df_exploded2, df_ripple_set1=ripple_sets_valid[0], users=valid_users)
    )
    print("> Ripple Sets created")


    knowledge_graph = {
        "train_set": train_set,
        "supervision_set": supervision_set,
        "valid_set": valid_set,
        "ripple_sets_train": ripple_sets_train,
        "ripple_sets_valid": ripple_sets_valid,
    }

    with open(os.path.join(dir_art, "knowledge_graph.pkl"), "wb") as f:
        pickle.dump(knowledge_graph, f)
    print("> Knowledge Graph Saved")


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--artefact_directory', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    build_knowledge_graph()
