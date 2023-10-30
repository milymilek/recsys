import pandas as pd

from data.dataframe import IDataFrame, DataFrame
from features.scheme import FeatureScheme


class DataStore:
    def __init__(self, data_path, read_method, **kwargs):
        _read_fun = self.__class__.get_read_method(read_method)
        self.dataframe: IDataFrame = DataFrame(_read_fun(data_path, **kwargs))
        self.scheme: FeatureScheme = FeatureScheme()
        self.mapping = None

    @staticmethod
    def get_read_method(method):
        match method:
            case 'csv': return pd.read_csv
            case 'parquet': return pd.read_parquet
            case 'json': return pd.read_json

    def apply(self, func):
        _kwargs = func.keywords
        if "map_func" in _kwargs and _kwargs['map_func']:
            self.mapping = func(self)
        else:
            df, cols = func(self)
            self.dataframe = df
            self.scheme.extend(cols)