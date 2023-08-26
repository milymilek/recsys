import numpy as np

def precision_k(reco_relevance, relevance, k=10):
    v = np.asarray(relevance.sum(axis=1).flatten(), dtype=int)[0].clip(1, k)
    bool_2d = np.vstack([np.concatenate((np.ones(i), np.zeros(k - i))) for i in v]).astype(bool)

    prec_k = (reco_relevance.getA().sum(axis=1, where=bool_2d) / v).mean()
    return prec_k


def recall_k(reco_relevance, relevance, k=10):
    sum_relevant = relevance.sum(axis=1)
    return (reco_relevance.sum(axis=1) / sum_relevant).mean()


def ndcg_k(reco_relevance, relevance, k=10):
    v = np.asarray(relevance.sum(axis=1).flatten(), dtype=int)[0].clip(1, k)
    ideal_relevance = np.vstack([np.concatenate((np.ones(i), np.zeros(k - i))) for i in v])

    discount = 1 / np.log2(np.arange(2, k + 2))
    idcg = (ideal_relevance * discount).sum(axis=1)
    dcg = (reco_relevance * discount).sum(axis=1)
    ndcg = (dcg / idcg).mean()

    return ndcg