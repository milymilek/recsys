import torch

from recsys.evaluation.evaluation import recommendation_relevance
from recsys.evaluation.utils import validate_metric_inputs


def recall_k(
    recommendations: torch.Tensor, ground_truth: torch.Tensor, k: int, users_idx: torch.Tensor, n_users: int, n_items: int
) -> torch.Tensor:
    recommendations_k = recommendations[:, :k]
    rel, rel_sum, rel_mask = recommendation_relevance(recommendations_k, ground_truth, users_idx, n_users, n_items)
    return torch.mean(torch.sum(rel[rel_mask], dim=1) / rel_sum[rel_mask])
