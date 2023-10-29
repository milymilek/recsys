import pandas as pd
from typing import List, Tuple, Callable
from functools import partial

from src.data.datastore import DataStore
from src.features.scheme import FeatureScheme


class Workflow:
    def __init__(self, pipe: List[Tuple[Callable, dict]]) -> None:
        self.ds: DataStore = None
        self.pipe = [partial(fun, **kwargs) for fun, kwargs in pipe]

    def fit(self, ds: DataStore) -> None:
        self.ds = ds

    def transform(self) -> None:
        for func in self.pipe:
            self.ds.apply(func)
        self.ds.dataframe.cut_nonfeat(self.ds.scheme.get_colnames())
