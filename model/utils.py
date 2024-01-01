import torch
import torch_geometric as pyg


def get_edge_att_optimized(x, edge_index, edge_attr):
    """ A memory-optimized edge attention calculator implemented
    with chunk processing to enhance efficiency in handling large datasets
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

    return pyg.utils.dense_to_sparse(edge_att)  # returns edge_index, edge_attr


def get_edge_att(x, edge_index, edge_attr):
    """ An edge attention calculator """

    # Initialize values
    num_nodes, num_features = x.size()

    # Update edge attentions
    edge_att = pyg.utils.to_dense_adj(edge_index, edge_attr=edge_attr).squeeze(0)
    norm_x = torch.norm(x, dim=1).view(num_nodes, 1)
    cosine_similarity_chuck = (x @ x.T) / (norm_x @ norm_x.T)
    edge_att = torch.mul(cosine_similarity_chuck, edge_att)

    return pyg.utils.dense_to_sparse(edge_att)  # returns edge_index, edge_attr
