import torch
import torch_geometric as pyg
import networkx as nx


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
