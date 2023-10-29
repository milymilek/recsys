import numpy as np
import torch
from torch.nn.functional import pad

from src.features.features import SparseFeat, DenseFeat


def attr2tensor(features: list, attr: np.ndarray) -> torch.tensor:
    x_tensor = []
    for elem in attr:
        elem_tensor = []
        for i, f in enumerate(features):
            if isinstance(f, SparseFeat):
                if f.max_len > 1:
                    elem_i = pad(torch.tensor(elem[i]), (0, f.max_len - len(elem[i])), value=-1) + 1
                else:
                    elem_i = torch.tensor([elem[i]]) + 1
            elif isinstance(f, DenseFeat):
                elem_i = torch.tensor([elem[i]])

            elem_tensor.append(elem_i)

        elem_tensor = torch.cat(elem_tensor)
        x_tensor.append(elem_tensor)

    x_tensor = torch.stack(x_tensor)
    return x_tensor
