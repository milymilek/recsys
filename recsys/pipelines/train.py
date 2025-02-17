import argparse
import os
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.deep import DeepDatasetIterable, FeaturelessDatasetIterable, collate_fn
from features.scheme import FeatureScheme
from features.store import FeatureStore
from models import DeepFM, NCF, MF
from utils import write_scalars


def train_epoch(model, criterion, optimizer, train_loader, device):
    model.train()

    running_loss = 0.
    preds, ground_truths = [], []
    for i_batch, (batch, y_true) in enumerate(tqdm(train_loader)):
        batch, y_true = batch.to(device), y_true.to(device)

        y_pred = model(batch)
        loss = criterion(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds.append(y_pred)
        ground_truths.append(y_true)
        running_loss += loss.item()

    pred = torch.cat(preds, dim=0).detach().sigmoid().cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
    train_loss = running_loss / len(train_loader)
    train_roc_auc = roc_auc_score(ground_truth, pred)

    return train_loss, train_roc_auc


@torch.no_grad()
def test(model, criterion, val_loader, device):
    model.eval()

    running_loss = 0.
    preds, ground_truths = [], []

    for i_batch, (batch, y_true) in enumerate(val_loader):
        batch, y_true = batch.to(device), y_true.to(device)

        y_pred = model(batch)
        loss = criterion(y_pred, y_true)

        preds.append(y_pred)
        ground_truths.append(y_true)
        running_loss += loss.item()

    pred = torch.cat(preds, dim=0).sigmoid().cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

    test_loss = running_loss / len(val_loader)
    test_roc_auc = roc_auc_score(ground_truth, pred)

    return test_loss, test_roc_auc


def train():
    args = get_args()
    dir_art = args.artefact_directory
    model_name = args.model
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    n_epochs = args.epochs
    seed = args.seed

    torch.manual_seed(seed)

    with open(os.path.join(dir_art, 'data.pkl'), "rb") as f:
        data = pd.read_pickle(f)

    train_set = data['relations_datastore'].dataframe.train.values
    supervision_set = data['relations_datastore'].dataframe.supervision.values
    valid_set = data['relations_datastore'].dataframe.valid.values
    item_attr = data['items_datastore'].dataframe.df
    user_attr = data['users_datastore'].dataframe.df
    scheme_relations = data['relations_datastore'].scheme
    scheme_items = data['items_datastore'].scheme
    scheme_users = data['users_datastore'].scheme

    train_set = np.concatenate((train_set, supervision_set), axis=0)
    n_users, n_items = user_attr.shape[0], item_attr.shape[0]

    feature_store = FeatureStore(scheme_relations, scheme_items, scheme_users, emb_dims={"sparse": 4, "varlen": 4})

    if model_name == "DeepFM":
        train_dataset = DeepDatasetIterable(feature_store, train_set, user_attr, item_attr, user_batch_size=int(1e4), neg_sampl=1)
        val_dataset = DeepDatasetIterable(feature_store, valid_set, user_attr, item_attr, user_batch_size=int(1e4), neg_sampl=1)
        model = DeepFM(feature_store, hidden_dim=[64, 16], device=device).to(device)
    elif model_name == "NCF":
        train_dataset = DeepDatasetIterable(feature_store, train_set, user_attr, item_attr, user_batch_size=int(1e4), neg_sampl=2)
        val_dataset = DeepDatasetIterable(feature_store, valid_set, user_attr, item_attr, user_batch_size=int(1e4), neg_sampl=2)
        model = NCF(feature_store, hidden_dim=[128, 64]).to(device)
    elif model_name == "MF":
        scheme_relations.features = [f for f in scheme_relations.features if f.name in ['user_id', 'app_id']]
        feature_store = FeatureStore(scheme_relations,
                                     FeatureScheme.from_feature_list([]),
                                     FeatureScheme.from_feature_list([]),
                                     emb_dims={"sparse": 16, "varlen": 16})

        train_dataset = FeaturelessDatasetIterable(train_set, n_users, n_items, user_batch_size=int(1e4), neg_sampl=5)
        val_dataset = FeaturelessDatasetIterable(valid_set, n_users, n_items, user_batch_size=int(1e4), neg_sampl=5)

        model = MF(feature_store, device=device).to(device)

    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn, drop_last=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=1e-4, momentum=0.9)

    log_dir = f"runs/{model.__class__.__name__}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    scalar_names = ['Loss/train', 'Loss/test', 'ROC_AUC/train', 'ROC_AUC/test']

    print(f"> Training model[{model.__class__.__name__}] on device[{device}] begins...")
    best_roc_auc = -1.0
    best_epoch = -1
    early_stop_thresh = 10
    for epoch in tqdm(range(n_epochs)):
        train_loss, train_roc_auc = train_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device
        )
        test_loss, test_roc_auc = test(
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            device=device
        )
        scalars = (train_loss, test_loss, train_roc_auc, test_roc_auc)
        write_scalars(writer=writer, names=scalar_names, scalars=scalars, step=epoch)

        if test_roc_auc > best_roc_auc:
            best_roc_auc = test_roc_auc
            best_epoch = epoch
            torch.save(model.state_dict(), f"{log_dir}/model.pth")
        elif epoch - best_epoch > early_stop_thresh:
            print("Early stopped training at epoch %d" % epoch)
            break

        print(f"""Epoch <{epoch}>\ntrain_loss: {train_loss} - train_roc_auc: {train_roc_auc}
test_loss: {test_loss} - test_roc_auc: {test_roc_auc}\n""")

    writer.close()


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--artefact_directory', type=str)
    parser.add_argument('--model', type=str, choices=['DeepFM', 'NCF', 'MF'], required=True)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    train()
