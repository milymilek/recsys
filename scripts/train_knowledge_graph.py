import argparse
import os

from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from models.kg import RippleNet
from dataset.kg import RippleDataset, collate_fn
from utils.utils import write_scalars


def train_epoch(model, criterion, optimizer, train_loader, device):
    model.train()

    running_loss = 0.
    preds, ground_truths = [], []
    for i_batch, (edge_index, ripple_sets) in enumerate(tqdm(train_loader)):
        neg_sampl = edge_index.clone().repeat(2, 1)
        neg_sampl[:, 1] = torch.randint(low=1, high=1232, size=(train_loader.batch_size * 2,))
        edge_index = torch.cat([edge_index, neg_sampl]).to(device)

        ripple_sets = [rs.repeat(1 + 2, 1, 1).to(device) for rs in ripple_sets]
        y_true = torch.cat([
            torch.tensor([1.0] * train_loader.batch_size),
            torch.tensor([0.0] * (train_loader.batch_size * 2))
        ]).unsqueeze(-1).to(device)

        y_pred = model(edge_index, ripple_sets)
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

    for i_batch, (edge_index, ripple_sets) in enumerate(tqdm(val_loader)):
        neg_sampl = edge_index.clone().repeat(2, 1)
        neg_sampl[:, 1] = torch.randint(low=1, high=1232, size=(val_loader.batch_size * 2,))
        edge_index = torch.cat([edge_index, neg_sampl]).to(device)

        ripple_sets = [rs.repeat(1 + 2, 1, 1).to(device) for rs in ripple_sets]
        y_true = torch.cat([
            torch.tensor([1.0] * val_loader.batch_size),
            torch.tensor([0.0] * (val_loader.batch_size * 2))
        ]).unsqueeze(-1).to(device)

        y_pred = model(edge_index, ripple_sets)
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

    with open(os.path.join(dir_art, 'knowledge_graph.pkl'), "rb") as f:
        knowledge_graph = pd.read_pickle(f)

    train_set = knowledge_graph["train_set"]
    supervision_set = knowledge_graph["supervision_set"]
    valid_set = knowledge_graph["valid_set"]
    ripple_sets_train = knowledge_graph['ripple_sets_train']
    ripple_sets_valid = knowledge_graph['ripple_sets_valid']
    entity_map = knowledge_graph['entity_map']

    # Data adjustment, todo: refactor
    train_rs0_users, train_rs1_users = (ripple_sets_train[i].index.get_level_values(0).unique() for i in range(2))
    valid_rs0_users, valid_rs1_users = (ripple_sets_valid[i].index.get_level_values(0).unique() for i in range(2))
    diff = np.concatenate([np.setdiff1d(train_rs0_users, train_rs1_users), np.setdiff1d(valid_rs0_users, valid_rs1_users)])
    ripple_sets_train[0] = ripple_sets_train[0].drop(diff)
    ripple_sets_valid[0] = ripple_sets_valid[0].drop(diff)
    ripple_sets_valid[1] = ripple_sets_valid[1].drop(diff)
    supervision_set = supervision_set[~supervision_set['user_id'].isin(diff)].reset_index(drop=True)
    valid_set = valid_set[~valid_set['user_id'].isin(diff)].reset_index(drop=True)
    # ===============================

    train_dataset = RippleDataset(train_set, supervision_set, ripple_sets_train)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn, drop_last=True)

    valid_dataset = RippleDataset(pd.concat([train_set, supervision_set]), valid_set, ripple_sets_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn, drop_last=True)

    if model_name == "RippleNet":
        model = RippleNet(emb_dim=16, n_relations=4, n_entities=max(entity_map.values())).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=1e-4, momentum=0.9)

    log_dir = f"runs/{model.__class__.__name__}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    scalar_names = ['Loss/train', 'Loss/test', 'ROC_AUC/train', 'ROC_AUC/test']

    print(f"> Training model[{model.__class__.__name__}] on device[{device}] begins...")
    best_roc_auc = -1.0
    best_epoch = -1
    early_stop_thresh = 2
    for epoch in tqdm(range(n_epochs)):
        train_loss, train_roc_auc = train_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=valid_loader,
            device=device
        )
        test_loss, test_roc_auc = test(
            model=model,
            criterion=criterion,
            val_loader=valid_loader,
            device=device
        )
        scalars = (train_loss, test_loss, train_roc_auc, test_roc_auc)
        write_scalars(writer=writer, names=scalar_names, scalars=scalars, step=epoch)

        if test_roc_auc > best_roc_auc:
            best_roc_auc = test_roc_auc
            best_epoch = epoch
            torch.save(model.state_dict(), f"{log_dir}/model.pth")
        elif epoch - best_epoch > early_stop_thresh:
            print(f"> Early stopped training at epoch {epoch}")
            break

        print(f"""Epoch <{epoch}>\ntrain_loss: {train_loss} - train_roc_auc: {train_roc_auc}
test_loss: {test_loss} - test_roc_auc: {test_roc_auc}\n""")

    writer.close()


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--artefact_directory', type=str)
    parser.add_argument('--model', type=str, choices=['RippleNet'], required=True)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    train()
