import argparse
import os

from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm
from datetime import datetime

import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from models.gnn import GraphSAGE, GCN, GNN
from utils.utils import write_scalars


def train_epoch(model, criterion, optimizer, train_loader, val_loader, device):
    model.train()

    running_loss = 0.
    preds, ground_truths = [], []
    for i_batch, batch in enumerate(tqdm(train_loader)):
        batch = batch.to(device)

        y_pred = model(batch)
        y_true = batch['user', 'recommends', 'app'].edge_label
        loss = criterion(y_pred, y_true.float())
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

    for i_batch, batch in enumerate(val_loader):
        batch = batch.to(device)

        y_pred = model(batch)
        y_true = batch['user', 'recommends', 'app'].edge_label
        loss = criterion(y_pred, y_true.float())

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
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    n_epochs = args.epochs

    with open(os.path.join(dir_art, 'graph.pkl'), "rb") as f:
        graph = pd.read_pickle(f)

    train_data = graph['train_data']
    valid_data = graph['valid_data']

    user_shape = train_data['user'].x.shape
    app_shape = train_data['app'].x.shape

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 15],
        neg_sampling_ratio=20.0,
        edge_label_index=(('user', 'recommends', 'app'), train_data['user', 'recommends', 'app'].edge_label_index),
        edge_label=train_data['user', 'recommends', 'app'].edge_label,
        batch_size=1024,
        shuffle=True,
        drop_last=True
    )
    valid_loader = LinkNeighborLoader(
        data=valid_data,
        num_neighbors=[20, 15],
        neg_sampling_ratio=20.0,
        edge_label_index=(('user', 'recommends', 'app'), valid_data['user', 'recommends', 'app'].edge_label_index),
        edge_label=valid_data['user', 'recommends', 'app'].edge_label,
        batch_size=1024,
        shuffle=True,
        drop_last=True
    )

    if model_name == "GraphSAGE":
        gnn_model = GraphSAGE(hidden_channels=32, out_channels=32)
    elif model_name == "GCN":
        gnn_model = GCN(hidden_channels=32, out_channels=32)

    model = GNN(
        gnn_model=gnn_model,
        entities_shapes={"user": user_shape, "app": app_shape},
        hidden_channels=32,
        metadata=train_data.metadata()
    ).to(device)
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
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--cuda', type=bool, default=False)

    return parser.parse_args()


if __name__ == '__main__':
    train()
