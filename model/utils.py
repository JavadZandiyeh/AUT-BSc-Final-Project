import torch
import torch_geometric as pyg
import networkx as nx


def get_edge_att_optimized(x, edge_index, edge_attr):
    """ A memory-optimized edge attention calculator has been implemented with chunk processing
    to enhance efficiency in handling large datasets. This function takes the edge attributes
    as input and returns the corresponding edge attentions.
    """

    # Initialize values
    num_nodes, num_features = x.size()
    division = [i for i in range(1, num_nodes + 1) if num_nodes % i == 0]

    # Update edge attentions
    edge_att = pyg.utils.to_dense_adj(edge_index, edge_attr=edge_attr).squeeze(0)
    norm_x = torch.norm(x, dim=1).view(num_nodes, 1)

    chunk = int(num_nodes / division[3])  # Notice: It can cause an error for an out-of-index situation
    start_point, end_point = 0, chunk

    while start_point < num_nodes:
        # Step1: Calculate cosine similarity for the chunk
        x_chuck = x[start_point: (start_point + chunk), :].view(chunk, num_features)
        norm_x_chuck = torch.norm(x_chuck, dim=1).view(chunk, 1)
        cosine_similarity_chuck = (x_chuck @ x.T) / (norm_x_chuck @ norm_x.T)

        # Step 2: Update edge attentions on the chuck
        edge_att_chuck = edge_att[start_point: (start_point + chunk), :].view(chunk, num_nodes)
        edge_att[start_point: (start_point + chunk), :] = torch.mul(cosine_similarity_chuck, edge_att_chuck)

        start_point, end_point = start_point + chunk, end_point + chunk

    # Clear memory
    del norm_x, x_chuck, norm_x_chuck, cosine_similarity_chuck, edge_att_chuck

    edge_att = torch.clamp(edge_att, min=0)  # Set negative attentions to zero

    return pyg.utils.dense_to_sparse(edge_att)  # returns edge_index, edge_attr


def get_edge_att(x, edge_index, edge_attr):
    """ This function takes the edge attributes as input and returns the corresponding edge attentions. """

    # Initialize values
    num_nodes, num_features = x.size()

    # Update edge attentions
    edge_att = pyg.utils.to_dense_adj(edge_index, edge_attr=edge_attr).squeeze(0)

    # Compute similarity
    norm_x = torch.norm(x, dim=1).view(num_nodes, 1)
    cosine_similarity = (x @ x.T) / (norm_x @ norm_x.T)

    # Compute new edge attentions
    edge_att = torch.mul(cosine_similarity, edge_att)
    edge_att = torch.clamp(edge_att, min=0)  # Set negative attentions to zero

    return pyg.utils.dense_to_sparse(edge_att)[1]  # returns edge_att


def print_graph_diameter(x, edge_index, approximate=False):
    data = pyg.data.Data(x=x, edge_index=edge_index)

    graph = pyg.utils.to_networkx(data, to_undirected=True)

    connected_components = list(nx.connected_components(graph))

    for i, component in enumerate(connected_components, 1):
        subgraph = graph.subgraph(component)
        diameter = nx.approximation.diameter(subgraph) if approximate else nx.diameter(subgraph)
        print(f"component: {i}, length {len(component)}, diameter: {diameter}")


def get_cosine_similarity(matrix):
    matrix_norm = torch.nn.functional.normalize(matrix, p=2, dim=1)

    cosine_similarities = torch.matmul(matrix_norm, matrix_norm.t())

    return cosine_similarities
