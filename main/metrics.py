import math
import torch
from utils import EngineSteps, Metrics
import utils
import random
import torchmetrics


class MetricsCalculation:
    def __init__(self, data, h, eval_type: EngineSteps, sampling_weight=4, k=10):
        self.h = h
        self.k = k
        self.data = data
        self.sampling_weight = sampling_weight

        self.spearman = torchmetrics.SpearmanCorrCoef()

        self.edge_mask_eval = data.edge_mask_val if eval_type == EngineSteps.VAL else data.edge_mask_test

        self.y_pred = utils.edge_prediction(h, data.edge_index) * data.edge_mask_uiu

        self.users = data.node_mask_user.nonzero().squeeze().tolist()
        self.items = data.node_mask_item.nonzero().squeeze().tolist()

        self.items_pop = self.items_pop()
        self.items_pop_dict = dict(zip(self.items, self.items_pop))

        self.items_rank_avg_eval = self.items_rank_avg_eval()

    def items_recom(self, user):
        # mask
        edge_mask_user = self.data.edge_index[0].eq(user)  # edge in one direction
        edge_mask_user_train = edge_mask_user * self.data.edge_mask_train
        edge_mask_user_eval = edge_mask_user * self.edge_mask_eval

        # edge_index
        edge_index_user_train = self.data.edge_index[:, edge_mask_user_train]
        edge_index_user_eval = self.data.edge_index[:, edge_mask_user_eval]

        # step1: ground truth items
        y_pred_user_eval = self.y_pred[edge_mask_user_eval]
        y_user_eval = self.data.y[edge_mask_user_eval]

        # step2: non-neighbors items
        train_items = utils.get_neighbors(user, edge_index_user_train)
        eval_items = utils.get_neighbors(user, edge_index_user_eval)
        other_items = set(self.items) - train_items - eval_items

        edge_index_user_others = torch.tensor(
            [[user] * len(other_items), list(other_items)],
            dtype=self.data.edge_index.dtype
        ).to(utils.device)

        y_pred_user_others = utils.edge_prediction(self.h, edge_index_user_others)
        y_user_others = torch.zeros_like(y_pred_user_others)

        # step3: concat non-neighboring items and ground truth items
        y_pred_user = torch.hstack((y_pred_user_others, y_pred_user_eval))  # predicted
        y_user = torch.hstack((y_user_others, y_user_eval))  # actual
        edge_index_user = torch.hstack((edge_index_user_others, edge_index_user_eval))

        return edge_index_user, y_pred_user, y_user

    def items_recom_with_sampling(self, user):
        """ sampling items for the user to evaluate our recommender system """
        edge_mask_user = self.data.edge_index[0].eq(user)  # edge in one direction
        edge_mask_user_train = edge_mask_user * self.data.edge_mask_train
        edge_mask_user_eval = edge_mask_user * self.edge_mask_eval

        # step1: ground truth
        y_pred_user_eval, y_user_eval = self.y_pred[edge_mask_user_eval], self.data.y[edge_mask_user_eval]

        # step2: sample
        edge_index_user_train = self.data.edge_index[:, edge_mask_user_train]
        edge_index_user_eval = self.data.edge_index[:, edge_mask_user_eval]

        train_items = utils.get_neighbors(user, edge_index_user_train)
        eval_items = utils.get_neighbors(user, edge_index_user_eval)
        other_items = set(self.items) - train_items - eval_items

        num_samples = min(round(self.sampling_weight * len(eval_items)), len(other_items))
        sampled_items = random.sample(other_items, num_samples)

        edge_index_user_sample = torch.tensor(
            [[user] * num_samples, sampled_items],
            dtype=self.data.edge_index.dtype
        ).to(utils.device)

        y_pred_user_sample = utils.edge_prediction(self.h, edge_index_user_sample)
        y_user_sample = torch.zeros_like(y_pred_user_sample)

        # step3: concat samples and ground truth
        y_pred_user_sample = torch.hstack((y_pred_user_sample, y_pred_user_eval))  # predicted
        y_user_sample = torch.hstack((y_user_sample, y_user_eval))  # actual
        edge_index_user_sample = torch.hstack((edge_index_user_sample, edge_index_user_eval))

        return edge_index_user_sample, y_pred_user_sample, y_user_sample

    def items_pop(self):  # items popularity in the train data (over all users)
        edge_index_train = self.data.edge_index[:, self.data.edge_mask_train]
        items_pop = utils.get_degrees(self.items, edge_index_train)

        return items_pop

    def items_rank_avg_eval(self):  # average of eval-items ranks (over matched users)
        num_items = len(self.items)

        items_rank_sum = [0 for _ in range(num_items)]  # summation of ranks for each item over user profiles
        items_num_matched_users = [0 for _ in range(num_items)]  # num occurrence of item in users profile

        for user in self.users:
            edge_mask_user_eval = self.data.edge_index[0].eq(user) * self.edge_mask_eval

            if edge_mask_user_eval.sum() == 0:
                continue  # there is no edge in the evaluation dataset of user

            edge_index_user_eval = self.data.edge_index[:, edge_mask_user_eval]

            items_user = edge_index_user_eval[1].tolist()

            # find rank of items in the user profile
            y_pred_user = self.y_pred[edge_mask_user_eval]
            indices = y_pred_user.argsort(descending=True)
            items_user_rank = torch.zeros_like(y_pred_user, dtype=torch.int64)
            items_user_rank[indices] = torch.arange(1, len(items_user) + 1, dtype=torch.int64).to(utils.device)

            for item, rank in zip(items_user, items_user_rank):
                index = self.items.index(item)
                items_rank_sum[index] += rank
                items_num_matched_users[index] += 1

        items_rank_avg = [(items_rank_sum[i] / items_num_matched_users[i])
                          if (items_num_matched_users[i] != 0) else math.inf
                          for i in range(num_items)]

        return items_rank_avg

    def get_results(self):
        results = utils.init_metrics(0)

        results[Metrics.MAP.value] = self.get_map()
        results[Metrics.MRR.value] = self.get_mrr()
        results[Metrics.NDCG.value] = self.get_ndcg()
        results[Metrics.RECALL.value] = self.get_recall()
        results[Metrics.FScore.value] = self.get_f_score(results[Metrics.MAP.value], results[Metrics.RECALL.value])
        results[Metrics.PRU.value] = self.get_pru()
        results[Metrics.PRI.value] = self.get_pri()

        return results

    def get_map(self):  # Mean Average Precision
        _map, num_users = 0, 0

        for user in self.users:
            if (self.data.edge_index[0].eq(user) * self.edge_mask_eval).sum() == 0:
                continue  # there is no edge in the evaluation dataset of user
            num_users += 1

            _, y_pred_user, y_user = self.items_recom(user)

            topk_indices = torch.topk(y_pred_user, self.k)[1]
            pred = y_user[topk_indices]
            true_pred_indices = pred.nonzero().squeeze()
            true_pred_count = true_pred_indices.numel()
            pred[true_pred_indices] = torch.arange(1, true_pred_count + 1, dtype=pred.dtype).to(utils.device)
            pred /= torch.arange(1, self.k + 1, dtype=pred.dtype).to(utils.device)
            _map_user = pred.sum() / max(1, true_pred_count)
            _map += _map_user

        _map /= num_users

        return round(float(_map), 4)

    def get_mrr(self):
        mrr, num_users = 0, 0

        for user in self.users:
            if (self.data.edge_index[0].eq(user) * self.edge_mask_eval).sum() == 0:
                continue  # there is no edge in the evaluation dataset of user
            num_users += 1

            _, y_pred_user, y_user = self.items_recom(user)

            topk_indices = torch.topk(y_pred_user, self.k)[1]
            pred = y_user[topk_indices]
            top1_value, top1_index = torch.topk(pred, 1)

            if top1_value != 0:
                mrr_user = 1 / (top1_index + 1)
                mrr += mrr_user

        mrr /= num_users

        return round(float(mrr), 4)

    def get_ndcg(self):
        ndcg, num_users = 0, 0

        for user in self.users:
            if (self.data.edge_index[0].eq(user) * self.edge_mask_eval).sum() == 0:
                continue  # there is no edge in the evaluation dataset of user
            num_users += 1

            _, y_pred_user, y_user = self.items_recom(user)

            topk_indices = torch.topk(y_pred_user, self.k)[1]
            pred = y_user[topk_indices]
            topk_y_user = torch.topk(y_user, self.k)[0]

            d = torch.arange(2, self.k + 2).log2().to(utils.device)
            dcg_y_pred_user = ((2 ** pred - 1) / d).sum()
            dcg_y_user = ((2 ** topk_y_user - 1) / d).sum()

            if dcg_y_user != 0:
                ndcg_user = dcg_y_pred_user / dcg_y_user
                ndcg += ndcg_user

        ndcg /= num_users

        return round(float(ndcg), 4)

    def get_recall(self):
        recall, num_users = 0, 0

        for user in self.users:
            num_eval_edges = (self.data.edge_index[0].eq(user) * self.edge_mask_eval).sum()

            if num_eval_edges == 0:
                continue  # there is no edge in the evaluation dataset of user
            num_users += 1

            _, y_pred_user, y_user = self.items_recom(user)

            topk_indices = torch.topk(y_pred_user, self.k)[1]
            pred = y_user[topk_indices]
            recall_user = pred.count_nonzero() / min(self.k, num_eval_edges)

            recall += recall_user

        recall /= num_users

        return round(float(recall), 4)

    @staticmethod
    def get_f_score(precision, recall, beta=1):
        if precision > 0 or recall > 0:
            f_score = ((1 + beta ** 2) * precision * recall) / (((beta ** 2) * precision) + recall)
        else:
            f_score = 0

        return round(float(f_score), 4)

    def get_pru(self):
        pru, num_users = 0, 0

        for user in self.users:
            edge_mask_user_eval = self.data.edge_index[0].eq(user) * self.edge_mask_eval

            if edge_mask_user_eval.sum() == 0:
                continue
            num_users += 1

            edge_index_user_eval = self.data.edge_index[:, edge_mask_user_eval]

            items_user = edge_index_user_eval[1].tolist()

            # find rank of items in the user profile
            y_pred_user = self.y_pred[edge_mask_user_eval]
            indices = y_pred_user.argsort(descending=True)
            items_user_rank = torch.zeros_like(y_pred_user, dtype=torch.float32)
            items_user_rank[indices] = torch.arange(1, len(items_user) + 1, dtype=torch.float32).to(utils.device)

            items_pop_user = [self.items_pop_dict[item] for item in items_user]
            items_pop_user = torch.tensor(items_pop_user, dtype=torch.float32).to(utils.device)

            pru += self.spearman(items_pop_user, items_user_rank)

        pru /= - num_users

        return round(float(pru), 4)

    def get_pri(self):
        items_pop = torch.tensor(self.items_pop, dtype=torch.float32).to(utils.device)
        items_rank_avg_eval = torch.tensor(self.items_rank_avg_eval, dtype=torch.float32).to(utils.device)

        # items that have been occurred at least once in the user profiles of the eval dataset
        non_inf_mask = (items_rank_avg_eval != torch.inf)

        pri = - self.spearman(items_pop[non_inf_mask], items_rank_avg_eval[non_inf_mask])

        return round(float(pri), 4)
