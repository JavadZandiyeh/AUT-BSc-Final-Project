import torch
from utils import Metrics


def init_metrics(value=0):
    return {metric: value for metric in Metrics.list()}


def precision_at_k(y_pred, y, k):
    pass


def precision_at_k1(y_true, y_scores, k=10):
    """
    Calculate precision@k
    y_true: true labels (ground truth)
    y_scores: predicted scores for each class
    k: top k items to consider
    """
    _, top_indices = torch.topk(y_scores, k, dim=1)
    top_k_labels = torch.gather(y_true, 1, top_indices)

    num_correct = (top_k_labels.sum(dim=1) > 0).sum().item()
    precision = num_correct / (len(y_true) * k)
    return precision


def calculate_metrics(y_pred, y, loss, data, edge_mask_indices):
    results = init_metrics(0)

    # MSELoss
    results[Metrics.MSELOSS.value] = loss

    return results
