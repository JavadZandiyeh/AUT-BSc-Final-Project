import torch
from utils import Metrics
import utils


def init_metrics(value=0):
    return {metric: value for metric in Metrics.list()}


def precision_at_k(users, edge_index, y_pred, y_rel, k=10):
    p_at_k_avg = 0
    num_users = 0

    for user in users:
        edge_mask_user = edge_index[0].eq(user)
        y_pred_user, y_rel_user = y_pred[edge_mask_user], y_rel[edge_mask_user]
        k_adjusted = min(k, int(edge_mask_user.sum()))

        if k_adjusted > 0:
            _, topk_indices = torch.topk(y_pred_user, k_adjusted)
            p_at_k_avg += y_rel_user[topk_indices].sum() / k_adjusted
            num_users += 1

    p_at_k_avg /= max(1, num_users)

    return float(p_at_k_avg)


def recall_at_k(users, edge_index, y_pred, y_rel, k=10):
    r_at_k_avg = 0
    num_users = 0

    for user in users:
        edge_mask_user = edge_index[0].eq(user)
        y_pred_user, y_rel_user = y_pred[edge_mask_user], y_rel[edge_mask_user]
        tot_rel = y_rel_user.sum()
        k_adjusted = min(k, int(edge_mask_user.sum()))

        if tot_rel > 0 and k_adjusted > 0:
            _, topk_indices = torch.topk(y_pred_user, k_adjusted)
            r_at_k_avg += y_rel_user[topk_indices].sum() / tot_rel
            num_users += 1

    r_at_k_avg /= max(1, num_users)

    return float(r_at_k_avg)


def f_score(p_at_k, r_at_k, beta=1):
    return float(((1 + beta ** 2) * p_at_k * r_at_k) / ((beta ** 2 * p_at_k) + r_at_k))


def ndcg_at_k(users, edge_index, y_pred, y, k=10):
    ndcg_at_k_avg = 0
    num_users = 0

    for user in users:
        edge_mask_user = edge_index[0].eq(user)
        k_adjusted = min(k, int(edge_mask_user.sum()))

        if k_adjusted > 0:
            d = torch.arange(2, k_adjusted + 2).log2().to(utils.device)  # discount

            y_pred_user, y_user = y_pred[edge_mask_user], y[edge_mask_user]

            _, topk_y_pred_user_indices = torch.topk(y_pred_user, k_adjusted)
            topk_y_user, _ = torch.topk(y_user, k_adjusted)

            dcg_y_pred_user = (y_user[topk_y_pred_user_indices] / d).sum()  # discount cumulative gain (predicted)
            dcg_y_user = (topk_y_user / d).sum()  # discount cumulative gain (normal)

            # normalized discount cumulative gain
            if dcg_y_user > 0:
                ndcg_user = dcg_y_pred_user / dcg_y_user
                ndcg_at_k_avg += ndcg_user
                num_users += 1

    ndcg_at_k_avg /= max(1, num_users)

    return float(ndcg_at_k_avg)


def calculate_metrics(users, edge_index, y_pred, y, loss):
    results = init_metrics(0)

    y_rel = utils.classify(y, [0, 1])  # 0: non-relevance, 1: relevance

    # MSELoss
    results[Metrics.MSELOSS.value] = loss

    # P@K
    results[Metrics.P_AT_K.value] = precision_at_k(users, edge_index, y_pred, y_rel)

    # R@K
    results[Metrics.R_AT_K.value] = recall_at_k(users, edge_index, y_pred, y_rel)

    # FScore
    results[Metrics.FScore.value] = f_score(results[Metrics.P_AT_K.value], results[Metrics.R_AT_K.value])

    # NDCG@K
    results[Metrics.NDCG_AT_K.value] = ndcg_at_k(users, edge_index, y_pred, y)

    return results
