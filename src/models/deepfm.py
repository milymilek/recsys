import torch
from torch import nn

from layers import MLP, EmbeddingNet, CrossProductNet

import time
import functools

from layers.feature_cross import FM


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

class DeepFM(nn.Module):
    def __init__(self, feature_store, hidden_dim, device='cpu'):
        super(DeepFM, self).__init__()
        self.feature_store = feature_store
        self.device = device

        self.V = EmbeddingNet(feature_store, device=self.device)
        #self.fm = CrossProductNet(device=self.device)
        self.fm = FM(device=self.device)
        self.dnn = MLP(feature_store.get_input_dim(), hidden_dim)

    def forward(self, x):
        x_dense = x[:, self.feature_store.dense_index]
        x_sparse = self.V(x)
        x_sparse_dense = torch.cat([x_sparse, x_dense], dim=1).to(torch.float)

        x = self.fm(x_sparse) + self.dnn(x_sparse_dense)
        return x

    # @timer
    # def dense_forward(self, x):
    #     return x[:, self.feature_store.dense_index]
    #
    # @timer
    # def v_forward(self, x):
    #     return self.V(x)
    #
    # @timer
    # def fm_forward(self, x_sparse):
    #     return self.fm(x_sparse)
    #
    # @timer
    # def dnn_forward(self, x_sparse_dense):
    #     return self.dnn(x_sparse_dense)
    #
    # def forward(self, x):
    #     x_dense = self.dense_forward(x)
    #     x_sparse = self.v_forward(x)
    #     x_sparse_dense = torch.cat([x_sparse, x_dense], dim=1).to(torch.float)
    #
    #     x = self.fm_forward(x_sparse) + self.dnn_forward(x_sparse_dense)
    #     return x

