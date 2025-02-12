import functools
import time
from typing import Any

import torch


def write_scalars(writer, names, scalars, step):
    for name, scalar in zip(names, scalars):
        writer.add_scalar(name, scalar, step)


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def load_model(cls, model_path, model_kwargs, device="cpu"):
    model = cls(**model_kwargs)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    return model


def batch_dict_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) for k, v in batch.items()}
