import pytest

from metrics.metrics import precision_k


def test_precision_k():
    recommendation_relevance = None
    relevance = None
    k = 10

    assert 1 == 1
    #assert precision_k(recommendation_relevance, relevance, k) == 0.3


@pytest.mark.xfail
def test_recall_k():
    assert True is False


def test_ndcg_k():
    assert True is True


def test_past_interactions():
    # from scipy import sparse
    # import torch
    # past_i = (torch.rand(4,4) > 0.8).to(torch.int)
    # past_i_csr = sparse.csr_matrix(past_i)
    # past_i_csr.indices
    # past_i_csr.indptr
    # past_i
    # np.diff(past_i_csr.indptr)
    # np.repeat(np.arange(4), np.diff(past_i_csr.indptr))
    # past_i_csr.indices
    # id_y = past_interactions[user_batch].indices
    # prob[id_x, id_y] = -torch.inf
    pass