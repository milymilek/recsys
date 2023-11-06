import torch
from torch.nn.functional import pad

from features.features import SparseFeat, DenseFeat, VarlenFeat


# TODO: start+=feat['max_len'] powoduje blad, do sprawdzenia
class FeatureStore:
    def __init__(self, scheme_relations, scheme_items, scheme_users, emb_dims):
        self.tensor_id = 0

        self.edge_index = self._create_store(scheme_relations)
        self.user_feat = self._create_store(scheme_users)
        self.item_feat = self._create_store(scheme_items)
        self.emb_dims = self._create_emb_dims(emb_dims)
        self.dense_index = self.get_feature_index(DenseFeat)

    # TODO: implementacja metody do zwracania offsetu dla features + abstrakcje klas features
    # TODO: rozwiazanie indexu i pad indexu jako jednej zmiennej
    def _create_store(self, scheme):
        store = []
        pad_tensor_id = 0
        for feat in scheme.features:
            if isinstance(feat, VarlenFeat):
                offset = feat.max_len
                offset_pad = len(feat.vals)
            else:
                offset, offset_pad = 1, 1
            store.append(feat._replace(index=(self.tensor_id, self.tensor_id + offset),
                                       pad_index=(pad_tensor_id, pad_tensor_id + offset_pad)))
            self.tensor_id = self.tensor_id + offset
            pad_tensor_id += offset_pad
        return store

    def _create_emb_dims(self, emb_dims):
        dims = {}
        for f in self.features():
            if isinstance(f, VarlenFeat):
                dims[f.name] = emb_dims['varlen']
            elif isinstance(f, SparseFeat):
                dims[f.name] = emb_dims['sparse']
        return dims

    def features(self):
        return self.edge_index + self.user_feat + self.item_feat

    def get_input_dim(self):
        emb_len = sum(self.emb_dims.values())
        dense_len = len(self.get_feature_type(DenseFeat))
        return emb_len + dense_len

    def get_feature_type(self, feat_type):
        return [f for f in self.features() if isinstance(f, feat_type)]

    def get_emb_features(self):
        return [f for f in self.features() if isinstance(f, VarlenFeat) or isinstance(f, SparseFeat)]

    def get_feature_index(self, feat_type):
        feats = self.get_feature_type(feat_type)

        if not feats:
            return torch.tensor([], dtype=torch.long)
        return torch.hstack([torch.arange(f.pad_index[0], f.pad_index[1]) for f in feats])

    # TODO: zamiana warunku wyciągania odpowiedniego scheme (przepisanie store na inny? wzorzec)
    def attr2tensor(self, attr, scheme: str):
        if scheme == "edge_index":
            scheme = self.edge_index
        elif scheme == "item_feat":
            scheme = self.item_feat
        elif scheme == "user_feat":
            scheme = self.user_feat

        x_tensor = []
        for _, elem in attr.iterrows():
            elem_tensor = []
            for i, f in enumerate(scheme):
                elem_i = f.to_tensor(elem)
                elem_tensor.append(elem_i)
            elem_tensor = torch.cat(elem_tensor)
            x_tensor.append(elem_tensor)
        x_tensor = torch.stack(x_tensor).squeeze(2)
        return x_tensor