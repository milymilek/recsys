from collections import namedtuple


class SparseFeat(namedtuple('SparseFeat', ['name', 'map', 'num_emb', 'index'])):
    def __new__(cls, name, map, num_emb, index):
        return super(SparseFeat, cls).__new__(cls, name, map, num_emb, index)


class DenseFeat(namedtuple('DenseFeat', ['name', 'index'])):
    def __new__(cls, name, index):
        return super(DenseFeat, cls).__new__(cls, name, index)


class VarlenFeat(namedtuple('VarlenFeat', ['name', 'vals', 'num_emb', 'max_len', 'index'])):
    def __new__(cls, name, vals, num_emb, max_len, index):
        return super(VarlenFeat, cls).__new__(cls, name, vals, num_emb, max_len, index)
