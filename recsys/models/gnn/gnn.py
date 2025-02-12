import torch
import torch_geometric
from torch_geometric import nn


class Classifier(torch.nn.Module):
    def forward(self, x_user, x_app, edge_label_index):
        x_user = x_user[edge_label_index[0]]
        x_app = x_app[edge_label_index[1]]
        return (x_user * x_app).sum(dim=-1)


class GNN(torch.nn.Module):
    def __init__(self, gnn_model, entities_shapes, hidden_channels, metadata):
        super().__init__()

        self.user_emb = torch.nn.Embedding(entities_shapes['user'][0], hidden_channels)
        self.user_lin = torch.nn.Linear(entities_shapes['user'][1], hidden_channels)
        self.app_emb = torch.nn.Embedding(entities_shapes['app'][0], hidden_channels)
        self.app_lin = torch.nn.Linear(entities_shapes['app'][1], hidden_channels)

        self.gnn = gnn_model
        self.gnn = nn.to_hetero(self.gnn, metadata=metadata, aggr='sum')

        self.clf = Classifier()

    def forward(self, batch):
        x_dict = {
            "user": self.user_emb(batch['user'].n_id) + self.user_lin(batch['user'].x),
            "app": self.app_emb(batch['app'].n_id) + self.app_lin(batch['app'].x),
        }

        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        pred = self.clf(
            x_dict["user"],
            x_dict["app"],
            batch['user', 'recommends', 'app'].edge_label_index,
        )
        return pred

    @torch.no_grad()
    def evaluate(self, batch):
        self.eval()

        x_dict = {
            "user": self.user_emb(batch['user'].n_id) + self.user_lin(batch['user'].x),
            "app": self.app_emb(batch['app'].n_id) + self.app_lin(batch['app'].x),
        }

        x_dict = self.gnn(x_dict, batch.edge_index_dict)

        return x_dict


def xavier_init(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch_geometric.nn.dense.linear.Linear):
        torch.nn.init.xavier_normal_(m.weight, gain=1.41)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)