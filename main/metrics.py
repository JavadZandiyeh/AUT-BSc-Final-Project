import torch
from utils import Metrics
import utils
import random


class MetricsCalculation:
    def __init__(self, data, h, edge_mask_eval):
        self.data = data
        self.h = h
        self.edge_mask_eval = edge_mask_eval

        self.users = set(data.node_mask_user.nonzero().squeeze().tolist())
        self.items = set(data.node_mask_item.nonzero().squeeze().tolist())

        self.y_pred = utils.edge_prediction(h, data.edge_index) * data.edge_mask_uiu

    def item_recom_sampling(self, user, weight=4):
        """ sampling items for the user to evaluate our recommender system """
        edge_mask_user = self.data.edge_index[0].eq(user)  # edge in one direction
        edge_mask_user_train = edge_mask_user * self.data.edge_mask_train
        edge_mask_user_eval = edge_mask_user * self.edge_mask_eval

        # ground truth
        y_pred_user_eval, y_user_eval = self.y_pred[edge_mask_user_eval], self.data.y[edge_mask_user_eval]

        # sample
        edge_index_user_train = self.data.edge_index[:, edge_mask_user_train]
        edge_index_user_eval = self.data.edge_index[:, edge_mask_user_eval]

        train_items = utils.get_neighbors(user, edge_index_user_train)
        eval_items = utils.get_neighbors(user, edge_index_user_eval)
        other_items = self.items - train_items - eval_items

        num_samples = min(round(weight * len(eval_items)), len(other_items))
        sampled_items = random.sample(other_items, num_samples)

        edge_index_user_sample = torch.tensor(
            [[user] * num_samples, sampled_items],
            dtype=edge_index_user_train.dtype
        ).to(utils.device)

        y_pred_user_sample = utils.edge_prediction(self.h, edge_index_user_sample)
        y_user_sample = torch.zeros_like(y_pred_user_sample)

        # concat samples and ground truth
        y_pred_user_sample = torch.hstack((y_pred_user_sample, y_pred_user_eval))  # predicted
        y_user_sample = torch.hstack((y_user_sample, y_user_eval))  # actual
        edge_index_user_sample = torch.hstack((edge_index_user_sample, edge_index_user_eval))  # edge_index

        return edge_index_user_sample, y_pred_user_sample, y_user_sample

    def get_results(self):
        results = utils.init_metrics(0)

        results[Metrics.MAP.value] = self.get_map()
        results[Metrics.MRR.value] = self.get_mrr()
        results[Metrics.NDCG.value] = self.get_ndcg()
        results[Metrics.R_AT_K.value] = self.get_r_at_k()
        results[Metrics.FScore.value] = self.get_f_score(results[Metrics.MAP.value], results[Metrics.R_AT_K.value])

        return results

    def get_map(self, weight=4):  # Mean Average Precision
        _map, num_users = 0, 0

        for user in self.users:
            _, y_pred_user_sample, y_user_sample = self.item_recom_sampling(user, weight)

            if (num_samples := y_user_sample.numel()) == 0:
                continue

            k = round(num_samples / (weight + 1))
            topk_indices = torch.topk(y_pred_user_sample, k)[1]
            pred = y_user_sample[topk_indices]
            true_pred_indices = pred.nonzero().squeeze()
            true_pred_count = true_pred_indices.numel()
            pred[true_pred_indices] = torch.arange(1, true_pred_count + 1, dtype=pred.dtype).to(utils.device)
            pred /= torch.arange(1, k + 1, dtype=pred.dtype).to(utils.device)
            _map_user = pred.sum() / max(1, true_pred_count)

            _map += _map_user
            num_users += 1

        _map /= num_users

        return float(_map)

    def get_mrr(self, weight=4):
        mrr, num_users = 0, 0

        for user in self.users:
            _, y_pred_user_sample, y_user_sample = self.item_recom_sampling(user, weight)

            if (num_samples := y_user_sample.numel()) == 0:
                continue

            k = round(num_samples / (weight + 1))
            topk_indices = torch.topk(y_pred_user_sample, k)[1]
            pred = y_user_sample[topk_indices]
            top1_value, top1_index = torch.topk(pred, 1)
            mrr_user = (1 / (top1_index + 1)) if top1_value > 0 else 0

            mrr += mrr_user
            num_users += 1

        mrr /= num_users

        return float(mrr)

    def get_ndcg(self, weight=4):
        ndcg, num_users = 0, 0

        for user in self.users:
            _, y_pred_user_sample, y_user_sample = self.item_recom_sampling(user, weight)

            if (num_samples := y_user_sample.numel()) == 0:
                continue

            k = round(num_samples / (weight + 1))
            topk_indices = torch.topk(y_pred_user_sample, k)[1]
            pred = y_user_sample[topk_indices]
            topk_y_user = torch.topk(y_user_sample, k)[0]

            d = torch.arange(2, k + 2).log2().to(utils.device)
            dcg_y_pred_user = ((2 ** pred - 1) / d).sum()
            dcg_y_user = ((2 ** topk_y_user - 1) / d).sum()

            if dcg_y_user == 0:
                continue

            ndcg_user = dcg_y_pred_user / dcg_y_user

            ndcg += ndcg_user
            num_users += 1

        ndcg /= num_users

        return float(ndcg)

    def get_r_at_k(self, weight=4):
        r_at_k, num_users = 0, 0

        for user in self.users:
            _, y_pred_user_sample, y_user_sample = self.item_recom_sampling(user, weight)

            if (num_samples := y_user_sample.numel()) == 0:
                continue

            k = round(num_samples / (weight + 1))
            topk_indices = torch.topk(y_pred_user_sample, k)[1]
            pred = y_user_sample[topk_indices]
            r_at_k_user = pred.count_nonzero() / k

            r_at_k += r_at_k_user
            num_users += 1

        r_at_k /= num_users

        return float(r_at_k)

    def get_f_score(self, precision, recall, beta=1):
        return float(((1 + beta ** 2) * precision * recall) / (((beta ** 2) * precision) + recall))
