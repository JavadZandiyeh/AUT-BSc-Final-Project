import random

import numpy as np
import torch
import torch_geometric as pyg
import networkx as nx
from enum import Enum


class DistType(Enum):
    NORMAL = 'normal'
    UNIFORM = 'uniform'
    XAVIER_NORMAL = 'xavier_normal'
    XAVIER_UNIFORM = 'xavier_uniform'
    KAIMING_NORMAL = 'kaiming_normal'
    KAIMING_UNIFORM = 'kaiming_uniform'


class DeviceType(Enum):
    CPU = 'cpu'
    CUDA = 'cuda'


def get_edge_att(x, edge_index, edge_attr):
    """ This function takes the edge attributes as input and returns the corresponding edge attentions. """
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


def get_device():
    device = DeviceType.CUDA.value if torch.cuda.is_available() else DeviceType.CPU.value
    device = torch.device(device)

    return device


def get_edge_y_pred(h, edge_index, edge_attr, edge_mask):
    h_j = torch.index_select(h, 0, edge_index[0])
    h_i = torch.index_select(h, 0, edge_index[1])

    edge_y_pred = torch.zeros_like(edge_attr)
    edge_y_pred[edge_mask] = torch.nn.functional.cosine_similarity(h_j, h_i, dim=1)

    return edge_y_pred


def get_tensor_distribution(shape, _type: DistType = None):
    dist = torch.empty(shape).to(get_device())

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
    tup = tuple(map(lambda x: x.to(DeviceType.CPU.value).numpy(), list(tup)))
    indices = np.lexsort(tup)

    # convert back to torch type and the device
    indices = torch.from_numpy(indices).to(get_device())

    return indices


def neg_edge_sampling(edge_index, num_neg_samples=1):  # for undirected graphs only
    bipartite, s1, s2 = is_bipartite(edge_index)  # s1 is the set of users, and s2 is the set of items

    if not bipartite:
        edge_index_src = edge_index[:, edge_index[0, :] < edge_index[1, :]]
        num_nodes = len(torch.unique(edge_index_src))

        neg_edge_index1 = pyg.utils.negative_sampling(edge_index_src, num_nodes, num_neg_samples)
        neg_edge_index2 = torch.vstack((neg_edge_index1[1], neg_edge_index1[0]))
        neg_edge_index = torch.hstack((neg_edge_index1, neg_edge_index2))
        neg_edge_index = neg_edge_index[:, lexsort_tensor((neg_edge_index[1], neg_edge_index[0]))]

        return neg_edge_index

    """ degree-based negative sampling for the bipartite graph """
    adj_dict = get_adj_dict(edge_index)
    neg_edge_index = list()

    for node in s1:
        neighbors = set(adj_dict.get(node))
        non_neighbors = s2 - neighbors

        num_neg_samples = min(len(neighbors), len(non_neighbors))

        neg_neighbors = random.sample(non_neighbors, num_neg_samples)

        for neg_neighbor in neg_neighbors:
            neg_edge_index.append([node, neg_neighbor])
            neg_edge_index.append([neg_neighbor, node])

    neg_edge_index = torch.tensor(neg_edge_index).T.to(get_device())
    neg_edge_index = neg_edge_index[:, lexsort_tensor((neg_edge_index[1], neg_edge_index[0]))]

    return neg_edge_index


def pos_edge_sampling(edge_index, num_pos_samples=1, replacement=False):  # for undirected graphs only
    edge_index_cloned = edge_index.clone()

    # step 1: switch to cpu for working with numpy arrays
    edge_index_cloned = edge_index_cloned.to(DeviceType.CPU.value).numpy()
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
    edge_index_sampled = torch.from_numpy(edge_index_sampled).to(get_device())

    return edge_index_sampled[:2, :], edge_index_sampled[2]  # edge_index, indices


def edge_sampling(data, rate=0.7, neg=True, pos_replacement=False):
    data_cloned = data.clone()

    """ step 1: positive sampling mask """
    num_edges_uiu = int(torch.count_nonzero(data.edge_mask_uiu) / 2)
    num_samples_uiu = round(num_edges_uiu * rate)

    # step 1: pos_edge_mask_uiu
    edge_index_uiu = data.edge_index[:, data.edge_mask_uiu]
    _, pos_indices_uiu = pos_edge_sampling(edge_index_uiu, num_samples_uiu, pos_replacement)

    all_indices_uiu = torch.nonzero(data.edge_mask_uiu).T.squeeze()
    pos_indices_uiu = all_indices_uiu[pos_indices_uiu]

    pos_edge_mask_uiu = torch.zeros_like(data.edge_mask_uiu)
    pos_edge_mask_uiu[pos_indices_uiu] = True

    # step 2: update the data
    sampling_mask = torch.bitwise_or(data.edge_mask_ii, pos_edge_mask_uiu)

    data_cloned.edge_index = data.edge_index[:, sampling_mask]
    data_cloned.edge_mask_uiu = data.edge_mask_uiu[sampling_mask]
    data_cloned.edge_mask_ii = data.edge_mask_ii[sampling_mask]
    data_cloned.edge_attr = data.edge_attr[sampling_mask]
    data_cloned.y = data.y[sampling_mask]
    data_cloned.edge_mask_train = data.edge_mask_train[sampling_mask]
    data_cloned.edge_mask_test = data.edge_mask_test[sampling_mask]
    data_cloned.edge_mask_val = data.edge_mask_val[sampling_mask]

    """ step 2: negative sampling """
    if not neg:
        return data_cloned

