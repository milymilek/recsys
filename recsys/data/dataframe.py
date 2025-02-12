from abc import ABC, abstractmethod


class IDataFrame(ABC):
    @abstractmethod
    def apply(self, func):
        ...

    @abstractmethod
    def repr_df(self):
        ...

    @abstractmethod
    def cut_nonfeat(self, feat_cols):
        ...


class DataFrame(IDataFrame):
    def __init__(self, df):
        self.df = df

    def apply(self, func):
        self.df = func(self.df)

    def repr_df(self):
        return self.df

    def cut_nonfeat(self, feat_cols):
        self.df = self.df[feat_cols]


class SplitDataFrame(IDataFrame):
    def __init__(self, train, supervision, valid):
        self.train = train
        self.supervision = supervision
        self.valid = valid

    def apply(self, func):
        self.train, self.supervision, self.valid = func(self.train), func(self.supervision), func(self.valid)

    def repr_df(self):
        return self.train

    def cut_nonfeat(self, feat_cols):
        self.train = self.train[feat_cols]
        self.supervision = self.supervision[feat_cols]
        self.valid = self.valid[feat_cols]
