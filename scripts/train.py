import argparse
import pickle
import os
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.deep import DeepFMDataset, collate_fn
from features.store import FeatureStore
from models import DeepFM


def train_epoch(model, criterion, optimizer, train_loader, val_loader, device, print_loss=500):
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

        # if not ((i_batch + 1) % print_loss):
        #     pred = torch.cat(preds, dim=0).detach().sigmoid().cpu().numpy()
        #     ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        #     last_loss = running_loss / print_loss
        #
        #     train_roc_auc = roc_auc_score(ground_truth, pred)
        #     test_loss, test_roc_auc = test(model, criterion, val_loader, device)
        #
        #     preds, ground_truths = [], []
        #     running_loss = 0.
        #
        #     print(f"""batch <{i_batch}>\ntrain_loss: {last_loss} - train_roc_auc: {train_roc_auc}\n
        #     test_loss: {test_loss} - test_roc_auc: {test_roc_auc}\n""")

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


def write_progress(writer, scalars, epoch, batch=None):
    names = ['Loss/train', 'Loss/test', 'ROC_AUC/train', 'ROC_AUC/test']
    for name, scalar in zip(names, scalars):
        writer.add_scalar(name, scalar, epoch)


def train():
    args = get_args()
    dir_art = args.artefact_directory
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    n_epochs = args.epochs

    with open(os.path.join(dir_art, 'data.pkl'), "rb") as f:
        data = pd.read_pickle(f)

    train_set = data['relations_datastore'].dataframe.train.values.T
    supervision_set = data['relations_datastore'].dataframe.supervision.values.T
    valid_set = data['relations_datastore'].dataframe.valid.values.T
    item_attr = data['items_datastore'].dataframe.df
    user_attr = data['users_datastore'].dataframe.df
    scheme_relations = data['relations_datastore'].scheme
    scheme_items = data['items_datastore'].scheme
    scheme_users = data['users_datastore'].scheme

    train_set = np.concatenate((train_set, supervision_set), axis=1)

    feature_store = FeatureStore(scheme_relations, scheme_items, scheme_users, emb_dims={"sparse": 4, "varlen": 4})
    train_dataset = DeepFMDataset(feature_store, train_set.T, user_attr, item_attr, neg_sampl=2)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1024, collate_fn=collate_fn, drop_last=True)
    val_dataset = DeepFMDataset(feature_store, valid_set.T, user_attr, item_attr, neg_sampl=2)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1024, collate_fn=collate_fn, drop_last=True)

    model = DeepFM(feature_store, hidden_dim=[128, 64], device=device).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=1e-4, momentum=0.9)
    writer = SummaryWriter(log_dir=f"runs/{model.__class__.__name__}")

    best_roc_auc = -1.0
    best_epoch = -1
    early_stop_thresh = 2
    for epoch in tqdm(range(n_epochs)):
        train_loss, train_roc_auc = train_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
        test_loss, test_roc_auc = test(
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            device=device
        )
        scalars = (train_loss, test_loss, train_roc_auc, test_roc_auc)
        write_progress(writer=writer, scalars=scalars, epoch=epoch)

        if test_roc_auc > best_roc_auc:
            best_roc_auc = test_roc_auc
            best_epoch = epoch
            torch.save(model.state_dict(), f"models/{model.__class__.__name__}/{datetime.now().strftime('%Y%m%d%H%M%S')}.pth")
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
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--cuda', type=bool, default=False)

    return parser.parse_args()


if __name__ == '__main__':
    train()
