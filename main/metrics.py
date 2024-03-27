import torch
from utils import Metrics
import utils


def init_metrics(value=0):
    return {metric: value for metric in Metrics.list()}


def _map(users, edge_index, y_pred, y_rel, k=None):
    __map, num_users = 0, 0

    for user in users:
        edge_mask_user = edge_index[0].eq(user)

        num_neighbors = int(edge_mask_user.sum())
        k_adjusted = num_neighbors if (k is None) else min(k, num_neighbors)

        if k_adjusted > 0:
            y_pred_user, y_rel_user = y_pred[edge_mask_user], y_rel[edge_mask_user]

            topk_indices = torch.topk(y_pred_user, k_adjusted)[1]

            pred = y_rel_user[topk_indices]
            true_pred_indices = pred.nonzero().squeeze()
            true_pred_count = true_pred_indices.numel()
            pred[true_pred_indices] = torch.arange(1, true_pred_count + 1, dtype=pred.dtype).to(utils.device)
            pred /= torch.arange(1, k_adjusted + 1, dtype=pred.dtype).to(utils.device)

            __map += pred.sum() / max(1, true_pred_count)
            num_users += 1

    __map /= max(1, num_users)

    return float(__map)


def r_at_k(users, edge_index, y_pred, y_rel, k=None):
    _r_at_k, num_users = 0, 0

    for user in users:
        edge_mask_user = edge_index[0].eq(user)

        y_pred_user, y_rel_user = y_pred[edge_mask_user], y_rel[edge_mask_user]
        tot_rel = y_rel_user.sum()

        num_neighbors = int(edge_mask_user.sum())
        k_adjusted = num_neighbors if (k is None) else min(k, num_neighbors)

        if tot_rel > 0 and k_adjusted > 0:
            topk_indices = torch.topk(y_pred_user, k_adjusted)[1]
            _r_at_k += y_rel_user[topk_indices].sum() / tot_rel
            num_users += 1

    _r_at_k /= max(1, num_users)

    return float(_r_at_k)


def f_score(precision, recall, beta=1):
    return float(((1 + beta ** 2) * precision * recall) / (((beta ** 2) * precision) + recall))


def ndcg(users, edge_index, y_pred, y, k=None):
    _ndcg, num_users = 0, 0

    for user in users:
        edge_mask_user = edge_index[0].eq(user)

        num_neighbors = int(edge_mask_user.sum())
        k_adjusted = num_neighbors if (k is None) else min(k, num_neighbors)

        if k_adjusted > 0:
            d = torch.arange(2, k_adjusted + 2).log2().to(utils.device)  # discount

            y_pred_user, y_user = y_pred[edge_mask_user], y[edge_mask_user]

            topk_indices = torch.topk(y_pred_user, k_adjusted)[1]
            topk_y_user = torch.topk(y_user, k_adjusted)[0]

            dcg_y_pred_user = ((2 ** y_user[topk_indices] - 1) / d).sum()  # discount cumulative gain (predicted)
            dcg_y_user = ((2 ** topk_y_user - 1) / d).sum()  # discount cumulative gain (normal)

            # normalized discount cumulative gain
            if dcg_y_user > 0:
                ndcg_user = dcg_y_pred_user / dcg_y_user
                _ndcg += ndcg_user
                num_users += 1

    _ndcg /= max(1, num_users)

    return float(_ndcg)


def mrr(users, edge_index, y_pred, y_rel, k=None):
    _mrr, num_users = 0, 0

    for user in users:
        edge_mask_user = edge_index[0].eq(user)

        num_neighbors = int(edge_mask_user.sum())
        k_adjusted = num_neighbors if (k is None) else min(k, num_neighbors)

        if k_adjusted > 0:
            y_pred_user, y_rel_user = y_pred[edge_mask_user], y_rel[edge_mask_user]

            topk_indices = torch.topk(y_pred_user, k_adjusted)[1]

            top1_value, top1_index = torch.topk(y_rel_user[topk_indices], 1)

            if top1_value > 0:
                _mrr += (1 / (top1_index + 1))

            num_users += 1

    _mrr /= max(1, num_users)

    return float(_mrr)


def calculate_metrics(users, edge_index, y_pred, y, loss):
    results = init_metrics(0)

    y_rel = utils.classify(y, [0, 1])  # 0: non-relevance, 1: relevance

    results[Metrics.MSELOSS.value] = loss

    results[Metrics.MAP.value] = _map(users, edge_index, y_pred, y_rel)

    # results[Metrics.R_AT_K.value] = r_at_k(users, edge_index, y_pred, y_rel, 10)

    # results[Metrics.FScore.value] = f_score(results[Metrics.MAP.value], results[Metrics.R_AT_K.value])

    results[Metrics.NDCG.value] = ndcg(users, edge_index, y_pred, y)

    results[Metrics.MRR.value] = mrr(users, edge_index, y_pred, y_rel)

    return results
