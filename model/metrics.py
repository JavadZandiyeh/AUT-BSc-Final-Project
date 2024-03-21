import torch
from utils import Metrics


def init_metrics(value=0):
    return {metric: value for metric in Metrics.list()}


def precision_at_k(user_rec_dict, k=10):
    p_at_k_dict = dict()

    for user, rec in user_rec_dict.items():
        k_adjusted = min(k, rec.size(1))
        _, topk_indices = torch.topk(rec[0], k_adjusted)
        p_at_k_dict[user] = rec[1, topk_indices].round().sum().float() / k_adjusted

    p_at_k_avg = sum(p_at_k_dict.values()) / len(p_at_k_dict)

    return p_at_k_dict, p_at_k_avg


def calculate_metrics(data, batch_mask, y_pred_batch, y_batch, loss):
    results = init_metrics(0)

    edge_index_batch = data.edge_index[:, batch_mask]

    users = data.node_mask_user.nonzero().squeeze().tolist()

    user_rec_dict = dict()  # recommendation dict for each user
    for user in users:
        edge_indices = edge_index_batch[0].eq(user).nonzero().squeeze()
        user_rec_dict[user] = torch.vstack((y_pred_batch[edge_indices], y_batch[edge_indices]))

    # MSELoss
    results[Metrics.MSELOSS.value] = loss.item()

    # P@K
    _, p_at_k = precision_at_k(user_rec_dict, 10)
    results[Metrics.P_AT_K.value] = p_at_k

    return results
