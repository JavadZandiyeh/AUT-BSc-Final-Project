from typing import Optional

import torch
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.typing import torch_scatter
import utils


class CustomizedGAT(pyg.nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation
        self.linear = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_attr):
        """
        :param x: (num_nodes * in_channels)
        :param edge_index: (2 * num_edges)
        :param edge_attr: (num_edges)
        :return: (num_nodes * out_channels)
        """
        # Compute edge attentions
        _, edge_att = utils.get_edge_att(x, edge_index, edge_attr)

        # Perform softmax on the edge attentions
        # edge_att = pyg.utils.softmax(src=edge_att, index=edge_index[1])

        # Linearly transform node feature matrix
        x = self.linear(x)

        # Propagate messages
        out = self.propagate(edge_index=edge_index, x=x, edge_att=edge_att)

        out += self.bias
        out = torch.sigmoid(out)

        return out

    def message(self, x_j, edge_att):
        """
        :param x_j: (num_edges * in_channels)
        :param edge_att: (num_edges)
        :return: (num_edges * in_channels)
        """
        return edge_att.view(-1, 1) * x_j

    # def aggregate(self, msg_out, edge_index):
    #     """
    #     :param msg_out: (num_edges * in_channels)
    #     :param edge_index: (2 * num_edges)
    #     :return: (num_nodes * in_channels)
    #     """
    #     return None

    def update(self, aggr_out, x):
        """
        :param aggr_out: (num_nodes * in_channels)
        :return: (num_nodes * out_channels)
        """
        return aggr_out + x


class ItemItemModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.cgat1 = CustomizedGAT(in_channels, hidden_channels)
        self.cgat2 = CustomizedGAT(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        h1 = self.cgat1(x, edge_index, edge_attr)
        h2 = self.cgat2(h1, edge_index, edge_attr)

        return h2
