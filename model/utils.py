import random

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    edges = edge_index.numpy().T.tolist()

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


def negative_sampling(edge_index, num_neg_samples=None):
    bipartite, s1, s2 = is_bipartite(edge_index)  # s1 is the set of users, and s2 is the set of items

    if bipartite is False:
        num_nodes = len(torch.unique(edge_index))
        neg_edge_index = pyg.utils.negative_sampling(edge_index, num_nodes, num_neg_samples)

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

    neg_edge_index = torch.tensor(neg_edge_index).T

    return neg_edge_index


def positive_sampling(edge_index, num_pos_samples):
    pass

