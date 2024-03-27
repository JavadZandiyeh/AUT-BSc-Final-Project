import os
import pprint
import random

import numpy as np
import torch
import torch_geometric as pyg
import networkx as nx
import enum
from torch.utils.tensorboard.writer import SummaryWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ExtendedEnum(enum.Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class DistType(ExtendedEnum):
    NORMAL = 'normal'
    UNIFORM = 'uniform'
    XAVIER_NORMAL = 'xavier_normal'
    XAVIER_UNIFORM = 'xavier_uniform'
    KAIMING_NORMAL = 'kaiming_normal'
    KAIMING_UNIFORM = 'kaiming_uniform'


class EngineSteps(ExtendedEnum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class Metrics(ExtendedEnum):
    MSELOSS = 'MSELoss'  # Mean Squared Error Loss
    MAP = 'MAP'  # Mean Average Precision
    NDCG = 'NDCG'  # Normalized Discount Cumulative Gain
    MRR = 'MRR'  # Mean Reciprocal Rank
    # PRU = 'PRU'  # Preference Ranking Utility
    # PRI = 'PRI'  # Preference Ranking Index
    # R_AT_K = 'R_AT_K'  # Recall at K
    # FScore = 'FScore'  # F-beta Score


def get_edge_att(x, edge_index, edge_attr):
    x1, x2 = x[edge_index[0], :], x[edge_index[1], :]

    cosine_sim = torch.nn.functional.cosine_similarity(x1, x2, dim=1)
    edge_att = cosine_sim * edge_attr

    return edge_att


def print_graph_diameter(x, edge_index, approximate=False):
    data = pyg.data.Data(x=x, edge_index=edge_index)

    graph = pyg.utils.to_networkx(data, to_undirected=True)

    connected_components = list(nx.connected_components(graph))

    for i, component in enumerate(connected_components, 1):
        subgraph = graph.subgraph(component)
        diameter = nx.approximation.diameter(subgraph) if approximate else nx.diameter(subgraph)
        print(f"component: {i}, length {len(component)}, diameter: {diameter}")


def edge_prediction(x, edge_index):
    x_j = torch.index_select(x, 0, edge_index[0])
    x_i = torch.index_select(x, 0, edge_index[1])

    return torch.nn.functional.cosine_similarity(x_j, x_i, dim=1)


def get_tensor_distribution(shape, _type: DistType = None):
    dist = torch.empty(shape).to(device)

    if _type == DistType.XAVIER_NORMAL:
        torch.nn.init.xavier_normal_(dist)
    elif _type == DistType.XAVIER_UNIFORM:
        torch.nn.init.xavier_uniform_(dist)
    elif _type == DistType.NORMAL:
        torch.nn.init.normal_(dist)
    elif _type == DistType.UNIFORM:
        torch.nn.init.uniform_(dist)
    elif _type == DistType.KAIMING_NORMAL:
        torch.nn.init.kaiming_normal_(dist)
    elif _type == DistType.KAIMING_UNIFORM:
        torch.nn.init.kaiming_uniform_(dist)

    return dist


def get_nx_graph(edge_index):
    edges = edge_index.T.tolist()

    graph = nx.Graph()
    graph.add_edges_from(edges)

    return graph


def is_bipartite(edge_index):
    graph = get_nx_graph(edge_index)

    if nx.is_bipartite(graph):
        s1, s2 = nx.bipartite.sets(graph)
        return True, s1, s2

    return False, set(), set()


def get_adj_dict(edge_index):
    graph = get_nx_graph(edge_index)

    return nx.to_dict_of_lists(graph)


def lexsort_tensor(tup):
    # convert all tuple values to numpy type
    tup = tuple(map(lambda x: x.to('cpu').numpy(), list(tup)))
    indices = np.lexsort(tup)

    # convert back to torch type and the device
    indices = torch.from_numpy(indices).to(device)

    return indices


def neg_edge_sampling(edge_index, num_neg_samples=None, rate=1.0):  # for undirected graphs only
    bipartite, s1, s2 = is_bipartite(edge_index)  # s1 is the set of users, and s2 is the set of items

    if bipartite:  # degree-based negative sampling for the bipartite graph
        rate = max(0.0, rate)
        adj_dict = get_adj_dict(edge_index)
        neg_edge_index = list()

        for node in s1:
            neighbors = set(adj_dict.get(node))
            non_neighbors = s2 - neighbors

            num_neg_samples_node = round(len(neighbors) * rate) if (num_neg_samples is None) else num_neg_samples
            num_neg_samples_node = min(num_neg_samples_node,  len(non_neighbors))

            neg_neighbors = random.sample(non_neighbors, num_neg_samples_node)

            for neg_neighbor in neg_neighbors:
                neg_edge_index.append([node, neg_neighbor])
                neg_edge_index.append([neg_neighbor, node])

        neg_edge_index = torch.tensor(neg_edge_index).T.to(device)
    else:
        edge_index_src = edge_index[:, edge_index[0, :] < edge_index[1, :]]
        num_nodes = len(torch.unique(edge_index_src))

        if num_neg_samples is None:
            num_neg_samples = 1

        neg_edge_index1 = pyg.utils.negative_sampling(edge_index_src, num_nodes, num_neg_samples)
        neg_edge_index2 = torch.vstack((neg_edge_index1[1], neg_edge_index1[0]))
        neg_edge_index = torch.hstack((neg_edge_index1, neg_edge_index2))

    neg_edge_index = neg_edge_index[:, lexsort_tensor((neg_edge_index[1], neg_edge_index[0]))]

    return neg_edge_index.int()


def pos_edge_sampling(edge_index, num_pos_samples=1, replacement=False):  # for undirected graphs only
    edge_index_cloned = edge_index.clone()

    # step 1: switch to cpu for working with numpy arrays
    edge_index_cloned = edge_index_cloned.to('cpu').numpy()
    indices = np.arange(edge_index_cloned.shape[1]).reshape(1, -1)
    edge_index_cloned = np.vstack((edge_index_cloned, indices))  # add indices of each column

    # step 2: edge_index src-to-trg and trg-to-src
    src = edge_index_cloned[:, edge_index_cloned[0, :] < edge_index_cloned[1, :]]
    src = src[:, np.lexsort((src[1], src[0]))]

    trg = edge_index_cloned[:, edge_index_cloned[0, :] > edge_index_cloned[1, :]]
    trg = trg[:, np.lexsort((trg[0], trg[1]))]

    edge_index_cloned = np.vstack((src, trg))

    # step 3: sampling
    indices = np.arange(edge_index_cloned.shape[1]).reshape(1, -1).squeeze()
    indices = np.sort(np.random.choice(indices, size=num_pos_samples, replace=replacement))

    edge_index_sampled = edge_index_cloned[:, indices]
    edge_index_sampled = np.hstack((edge_index_sampled[:3, :], edge_index_sampled[3:, :]))
    edge_index_sampled = edge_index_sampled[:, np.lexsort((edge_index_sampled[1], edge_index_sampled[0]))]

    # step 4: convert to tensor and change the device
    edge_index_sampled = torch.from_numpy(edge_index_sampled).to(device)

    return edge_index_sampled[:2, :].int(), edge_index_sampled[2].int()  # edge_index, indices


def train_val_test_division(edge_index):  # for undirected graphs only
    num_edges = round(edge_index.size(1) / 2)

    num_train = round(num_edges * 0.8)
    num_val = round((num_edges - num_train) * 0.5)
    num_test = num_edges - num_train - num_val

    mask = torch.zeros_like(edge_index[0])

    _, val_indices = pos_edge_sampling(edge_index, num_val)
    mask[val_indices] = 1

    train_test_indices = torch.nonzero(mask.eq(0)).T.squeeze()
    _, test_indices = pos_edge_sampling(edge_index[:, train_test_indices], num_test)
    test_indices = train_test_indices[test_indices]
    mask[test_indices] = 2

    train_mask = torch.zeros_like(edge_index[0])
    train_mask[torch.nonzero(mask.eq(0)).T.squeeze()] = 1

    val_mask = torch.zeros_like(edge_index[0])
    val_mask[torch.nonzero(mask.eq(1)).T.squeeze()] = 1

    test_mask = torch.zeros_like(edge_index[0])
    test_mask[torch.nonzero(mask.eq(2)).T.squeeze()] = 1

    return train_mask.bool(), val_mask.bool(), test_mask.bool()


def edge_sampling(data, pos_rate=0.7, neg_rate=1.0, pos=True, neg=True, pos_replacement=False):  # for undirected graphs only
    cdata = data.clone()

    """ step 1: positive sampling mask """
    pos_rate = max(0.0, pos_rate)

    if pos and (pos_rate != 1 or pos_replacement):
        num_edges_uiu = int(torch.count_nonzero(data.edge_mask_uiu) / 2)
        num_samples_uiu = round(num_edges_uiu * pos_rate)

        # step 1: pos_edge_mask_uiu
        edge_index_uiu = data.edge_index[:, data.edge_mask_uiu]
        _, pos_indices_uiu = pos_edge_sampling(edge_index_uiu, num_samples_uiu, pos_replacement)

        all_indices_uiu = torch.nonzero(data.edge_mask_uiu).T.squeeze()
        pos_indices_uiu = all_indices_uiu[pos_indices_uiu]

        pos_edge_mask_uiu = torch.zeros_like(data.edge_mask_uiu)
        pos_edge_mask_uiu[pos_indices_uiu] = True

        # step 2: update the data
        sampling_mask = torch.bitwise_or(data.edge_mask_ii, pos_edge_mask_uiu)

        cdata.edge_index = data.edge_index[:, sampling_mask]
        cdata.edge_mask_uiu = data.edge_mask_uiu[sampling_mask]
        cdata.edge_mask_ii = data.edge_mask_ii[sampling_mask]
        cdata.edge_attr = data.edge_attr[sampling_mask]
        cdata.y = data.y[sampling_mask]
        cdata.edge_mask_train = data.edge_mask_train[sampling_mask]
        cdata.edge_mask_val = data.edge_mask_val[sampling_mask]
        cdata.edge_mask_test = data.edge_mask_test[sampling_mask]

    """ step 2: negative sampling """
    neg_rate = max(0.0, neg_rate)

    if neg and (neg_rate > 0.0):
        edge_index_train = cdata.edge_index[:, cdata.edge_mask_train]

        neg_edge_index_uiu = neg_edge_sampling(edge_index_train, rate=neg_rate)
        neg_edge_mask_uiu = torch.ones_like(neg_edge_index_uiu[0])
        neg_edge_mask_ii = torch.zeros_like(neg_edge_index_uiu[0])
        neg_edge_attr = torch.zeros_like(neg_edge_index_uiu[0])
        neg_y = torch.zeros_like(neg_edge_index_uiu[0])
        neg_edge_mask_train = torch.ones_like(neg_edge_index_uiu[0])
        neg_edge_mask_val = torch.zeros_like(neg_edge_index_uiu[0])
        neg_edge_mask_test = torch.zeros_like(neg_edge_index_uiu[0])

        cdata_stacked = torch.vstack((
            cdata.edge_index,
            cdata.edge_mask_uiu,
            cdata.edge_mask_ii,
            cdata.edge_attr,
            cdata.y,
            cdata.edge_mask_train,
            cdata.edge_mask_val,
            cdata.edge_mask_test
        ))

        neg_cdata_stacked = torch.vstack((
            neg_edge_index_uiu,
            neg_edge_mask_uiu,
            neg_edge_mask_ii,
            neg_edge_attr,
            neg_y,
            neg_edge_mask_train,
            neg_edge_mask_val,
            neg_edge_mask_test
        ))

        stack = torch.hstack((cdata_stacked, neg_cdata_stacked))
        stack_sorted_indices = lexsort_tensor((stack[1], stack[0]))
        stack = stack[:, stack_sorted_indices]

        cdata.edge_index = stack[:2, :]
        cdata.edge_mask_uiu = stack[2, :]
        cdata.edge_mask_ii = stack[3, :]
        cdata.edge_attr = stack[4, :]
        cdata.y = stack[5, :]
        cdata.edge_mask_train = stack[6, :]
        cdata.edge_mask_val = stack[7, :]
        cdata.edge_mask_test = stack[8, :]

    # cast to standard types
    cdata.edge_index = cdata.edge_index.long()
    cdata.edge_mask_uiu = cdata.edge_mask_uiu.bool()
    cdata.edge_mask_ii = cdata.edge_mask_ii.bool()
    cdata.edge_attr = cdata.edge_attr.float()
    cdata.y = cdata.y.float()
    cdata.edge_mask_train = cdata.edge_mask_train.bool()
    cdata.edge_mask_val = cdata.edge_mask_val.bool()
    cdata.edge_mask_test = cdata.edge_mask_test.bool()

    return cdata


def get_edge_match_indices(edge_index):  # for undirected graphs only
    edge_index_cloned = edge_index.clone()

    indices = torch.arange(0, edge_index_cloned.size(1)).reshape(1, -1).squeeze().to(device)
    edge_index_cloned = torch.vstack((edge_index_cloned, indices))  # add indices of each column

    src = edge_index_cloned[:, edge_index_cloned[0, :] < edge_index_cloned[1, :]]
    src = src[:, lexsort_tensor((src[1], src[0]))]

    trg = edge_index_cloned[:, edge_index_cloned[0, :] > edge_index_cloned[1, :]]
    trg = trg[:, lexsort_tensor((trg[0], trg[1]))]

    edge_match_indices = torch.vstack((src[2], trg[2]))

    return edge_match_indices


def mini_batching(edge_index, num_batches):  # for undirected graphs only
    # edge_match_indices and shuffling
    edge_match_indices = get_edge_match_indices(edge_index)
    num_edges = edge_match_indices.size(1)
    edge_match_indices = edge_match_indices[:, torch.randperm(num_edges)]  # shuffling

    # batch sizes
    quotient, remainder = num_edges // num_batches, num_edges % num_batches
    batch_sizes = torch.zeros((num_batches,), dtype=torch.int64).to(device)
    batch_sizes += quotient
    batch_sizes[:remainder] += 1

    # batches
    batches, pos1 = list(), 0

    for i in range(num_batches):
        pos2 = pos1 + batch_sizes[i]

        batch, _ = edge_match_indices[:, pos1:pos2].reshape(1, -1).squeeze().sort()

        batches.append(batch)

        pos1 = pos2

    return batches  # indices of edge_index


def classify(y: torch.Tensor, classes: list | torch.Tensor = [0, 1]):
    def get_class(value):
        return min(classes, key=lambda x: abs(x - value))

    y_clone = y.clone().to('cpu')

    return y_clone.apply_(get_class).to(device)


""" save results """


def create_summary_writer(settings) -> SummaryWriter:
    base_path = '../runs'

    run_details = f'e{settings["epochs"]}'\
        + f'-b{settings["num_batches"]}'\
        + f'-lr{settings["learning_rate"]}'\
        + f'-pos{settings["pos_sampling_rate"]}'\
        + f'-neg{settings["neg_sampling_rate"]}'

    path = os.path.join(base_path, settings['run_name'], run_details, settings['model_name'])

    return SummaryWriter(log_dir=path)


def epoch_summary_write(writer: SummaryWriter, epoch, train_results, val_results):
    # results
    results = {metric: {'train': 0, 'val': 0} for metric in Metrics.list()}

    for step, step_results in [('train', train_results), ('val', val_results)]:
        for metric, value in step_results.items():
            results[metric][step] = value

    # terminal
    pprint.pprint(results)

    # writer
    for metric, metric_results in results.items():
        writer.add_scalars(main_tag=metric, tag_scalar_dict=metric_results, global_step=epoch)


""" save and load the model """


def get_model_path(settings):
    run_details = f'e{settings["epochs"]}' \
                  + f'-b{settings["num_batches"]}' \
                  + f'-lr{settings["learning_rate"]}' \
                  + f'-pos{settings["pos_sampling_rate"]}' \
                  + f'-neg{settings["neg_sampling_rate"]}'

    path = os.path.join('../models', settings["run_name"], run_details)

    return path


def save_model(model, settings):
    model_name = f'{settings["model_name"]}.pth'

    path = get_model_path(settings)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, model_name)

    torch.save(obj=model.state_dict(), f=path)


def load_model(model, settings):
    model_name = f'{settings["model_name"]}.pth'

    path = get_model_path(settings)
    path = os.path.join(path, model_name)

    model.load_state_dict(torch.load(path, map_location=device))

    return model
