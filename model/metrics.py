import torch
from utils import Metrics


def init_metrics(value=0):
    return {metric: value for metric in Metrics.list()}


def p_at_k(edge_index, users, y_pred, y, k=10):
    p_at_k_avg = 0

    for user in users:
        user_edge_mask = edge_index[0].eq(user)

        k_adjusted = min(k, int(user_edge_mask.sum()))

        if k_adjusted > 0:
            y_pred_user, y_user = y_pred[user_edge_mask], y[user_edge_mask]

            _, topk_indices = torch.topk(y_pred_user, k_adjusted)

            p_at_k_avg += float(y_user[topk_indices].round().sum()) / k_adjusted

    p_at_k_avg /= len(users)

    return p_at_k_avg


def calculate_metrics(data, batch_mask, y_pred_batch, y_batch, loss):
    results = init_metrics(0)

    edge_index_batch = data.edge_index[:, batch_mask]

    users = data.node_mask_user.nonzero().squeeze().tolist()

    # MSELoss
    results[Metrics.MSELOSS.value] = loss.item()

    # P@K
    results[Metrics.P_AT_K.value] = p_at_k(edge_index_batch, users, y_pred_batch, y_batch)

    return results
