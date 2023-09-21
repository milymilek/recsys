import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from src.dataset import split_by_time, filter_set


def remap(df, col):
    idx = df[col].unique()
    new_idx = np.arange(idx.size)
    return {i: ni for i, ni in zip(idx, new_idx)}


def listed_attr_to_ohc(df, col='tags'):
    mlb = MultiLabelBinarizer()
    mlb.fit_transform(df[col])
    tag_map = {cat: idx for idx, cat in enumerate(mlb.classes_)}
    return df[col].apply(lambda x: [tag_map[c] for c in x]), tag_map


def featurize_apps(df):
    RATING_COL = ['Mixed', 'Mostly Negative', 'Mostly Positive', 'Overwhelmingly Positive', 'Positive', 'Very Positive']
    CAT_COL = ['win', 'mac', 'linux', 'steam_deck']

    for c in CAT_COL:
        df[c] = df[c].astype(int)

    rating = pd.get_dummies(df['rating'], columns=RATING_COL)
    rating = rating.reindex(columns=RATING_COL, fill_value=0.0)
    df = pd.concat([df, rating], axis=1)

    cols = ['app_id'] +  CAT_COL + ['price_original', 'price_final', 'discount',
                                    'user_reviews', 'positive_ratio'] + RATING_COL

    return df[cols]


def app_attr_as_numpy(df_tags, df_meta):
    N_TAGS = 425
    N_CAT = 10
    N_CONT = 5

    TAGS_COL = ['tags_id']
    CAT_COL = ['win', 'mac', 'linux', 'steam_deck', 'Mixed', 'Mostly Negative', 'Mostly Positive',
               'Overwhelmingly Positive', 'Positive', 'Very Positive']
    CONT_COL = ['positive_ratio', 'user_reviews', 'price_original', 'price_final', 'discount']

    N_APPS = df_tags.shape[0]

    attr_matrix = []
    for (i, tags), (i, meta) in zip(df_tags.iterrows(), df_meta.iterrows()):
        # tag = np.zeros(N_TAGS)
        # mask = tags[TAGS_COL].values[0]
        # tag[mask] = 1
        tag = tags[TAGS_COL].values[0]

        cat = list(meta[CAT_COL].values)
        cont = list(meta[CONT_COL].values)

        #v = np.concatenate((tag, cat, cont))
        v = cat + cont
        v.append(tag)
        attr_matrix.append(np.array(v, dtype=object))

    #return np.vstack(attr_matrix)
    return np.array(attr_matrix, dtype=object)


def user_attr_as_numpy(df_user):
    N_CONT = 2
    CONT_COL = ['products', 'reviews']

    return df_user[CONT_COL].values


def process():
    """Prepare Steam dataset to training and evaluation process."""
    args = get_args()

    dir = args.directory
    dir_art = args.directory + "/" + args.artefact_directory
    os.makedirs(dir_art, exist_ok=True)

    supervision_ratio = args.supervision
    validation_ratio = args.validation

    # Read data
    relations = pd.read_csv(os.path.join(dir, 'recommendations.csv'))
    users = pd.read_csv(os.path.join(dir, 'users.csv'))
    items = pd.read_csv(os.path.join(dir, 'games.csv'))
    items_meta = pd.read_json(os.path.join(dir, 'games_metadata.json'), lines=True)
    print("> Data read")

    # Filter negative relations to stick to PyG convention
    relations['is_recommended'] = relations['is_recommended'].astype(float)
    relations = relations[relations['is_recommended'] == 1.0]

    # Fraction `split` as training set and 1-`split` as test set. Filter out users and items occurring only in test set.
    relations_train, relations_supervision, relations_valid = split_by_time(df=relations, col="date",
                                                                           supervision_ratio=supervision_ratio,
                                                                           validation_ratio=validation_ratio)
    relations_supervision = filter_set(df=relations_supervision, df_train=relations_train, user_col="user_id", item_col="app_id")
    relations_valid = filter_set(df=relations_valid, df_train=relations_train, user_col="user_id", item_col="app_id")
    print("> Splitted and filtered")

    # Normalize (remap) ids of entities and save mappings.
    user_dict = remap(relations_train, 'user_id')
    item_dict = remap(relations_train, 'app_id')
    for rel in [relations_train, relations_supervision, relations_valid]:
        rel['user_id'] = rel['user_id'].map(user_dict)
        rel['app_id'] = rel['app_id'].map(item_dict)
    pd.DataFrame.from_dict(user_dict, orient='index').to_csv(os.path.join(dir_art, 'user_dict.csv'))
    pd.DataFrame.from_dict(item_dict, orient='index').to_csv(os.path.join(dir_art, 'item_dict.csv'))
    print("> Remapped")

    # Extract `tags` from items included in training set by one-hot encoding
    items_meta['app_id'] = items_meta['app_id'].map(item_dict)
    items_meta = items_meta.dropna()
    items_meta['app_id'] = items_meta['app_id'].astype(int)
    items_meta = items_meta.sort_values(by=['app_id'])
    items_meta['tags_id'], tags_dict = listed_attr_to_ohc(df=items_meta, col='tags')
    pd.DataFrame.from_dict(tags_dict, orient='index').to_csv(os.path.join(dir_art, 'tags_dict.csv'))
    print("> Tags encoded")

    # Extract continuous and categorical features
    items['app_id'] = items['app_id'].map(item_dict)
    items = items.dropna()
    items['app_id'] = items['app_id'].astype(int)
    items = items.sort_values(by=['app_id'])
    items = featurize_apps(items)

    users['user_id'] = users['user_id'].map(user_dict)
    users = users.dropna()
    users['user_id'] = users['user_id'].astype(int)
    users = users.sort_values(by=['user_id'])
    print("> Features encoded")

    # Create and store numpy entities attribute matrices
    app_attr = app_attr_as_numpy(items_meta, items)
    user_attr = user_attr_as_numpy(users)
    with open(os.path.join(dir_art, 'app_attr.pkl'), 'wb') as f:
        pickle.dump(app_attr, f)
    with open(os.path.join(dir_art, 'user_attr.pkl'), 'wb') as f:
        pickle.dump(user_attr, f)
    print("> Attribute matrices created")

    data = {
        "train_set": relations_train,
        "supervision_set": relations_supervision,
        "valid_set": relations_valid,
        "user_attr": user_attr,
        "item_attr": app_attr,
    }

    with open(os.path.join(dir_art, 'data.pkl'), 'wb') as f:
        pickle.dump(data, f)


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
