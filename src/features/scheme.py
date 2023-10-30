import pickle

from features.features import VarlenFeat


class FeatureScheme:
    def __init__(self):
        self.features = []

    def extend(self, feat):
        self.features.extend(feat)

    def get_colnames(self):
        names = []
        for c in self.features:
            if isinstance(c, VarlenFeat):
                names.extend(c.vals)
            else:
                names.append(c.name)
        return names

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)