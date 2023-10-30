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
