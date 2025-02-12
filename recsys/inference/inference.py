import numpy as np
from scipy.sparse import csr_matrix
import torch


def remove_past_interactions(prob: torch.Tensor, user_batch: torch.Tensor, past_interactions: csr_matrix) -> torch.Tensor:
    id_x = np.repeat(np.arange(user_batch.shape[0]), np.diff(past_interactions[user_batch].indptr)) # extract x-axis indices from csr
    id_y = past_interactions[user_batch].indices
    prob[id_x, id_y] = -torch.inf
    return prob


def recommend_k(user_emb, item_emb, past_interactions, k=10, user_batch_size=1000):
    item_emb = item_emb.T
    user_batches = torch.arange(user_emb.shape[0]).split(user_batch_size)
    recommended_batches = []

    for user_batch in user_batches:
        prob = (user_emb[user_batch] @ item_emb).sigmoid()
        prob = remove_past_interactions(prob, user_batch, past_interactions)
        recommended_batches.append(prob.topk(k, 1)[1])

    recommendations = torch.cat(recommended_batches, 0)
    return recommendations


def recommendation_relevance(recommendations, ground_truth):
    """
    Computes the relevance matrix of recommended items based on ground truth data.

    This function takes a matrix of recommended items and a ground truth sparse matrix, and calculates
    binary relevance of recommended items for each user. The relevance is determined by
    comparing the recommended items with the actual items in the ground truth.

    Args:
        recommendations (numpy.ndarray): A 2D matrix of shape (n_users, k) where k is the number of
            recommended items per user. Each row contains indices representing the recommended
            items for a user.
        ground_truth (scipy.sparse.csr_matrix): A sparse matrix of shape (n_users, n_items). The matrix
            contains binary values indicating whether an item is relevant (1) or not (0) for each user.

    Returns:
        numpy.matrix: A 2D matrix of shape (n_users, k) containing the relevance scores of the
        recommended items for each user.

    Raises:
        ValueError: If the dimensions of 'recommendations' and 'ground_truth' do not match or
            are incompatible for matrix operations.
    """
    n_users, n_items = ground_truth.shape
    k = recommendations.shape[1]

    if recommendations.shape[0] != n_users:
        raise ValueError("Number of users in 'recommendations' should match 'ground_truth'.")

    user_idx = np.repeat(np.arange(n_users), k)
    item_idx = recommendations.flatten()
    relevance = ground_truth[user_idx, item_idx].reshape(
        (n_users, k))  # `ground_truth` from indices corresponding to `recommendations`
    relevance_mask = np.asarray(
        (ground_truth.sum(axis=1) != 0)).ravel()  # Mask to filter out users with no interactions in `ground_truth`

    return relevance, relevance_mask


