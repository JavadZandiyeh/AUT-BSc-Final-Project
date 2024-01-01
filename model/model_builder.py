from typing import Optional

import torch
import torch_geometric as pyg
from torch import Tensor

import utils


class CustomizedGAT(pyg.nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.linear = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_attr):
        num_nodes, num_features = x.size()

        # Compute edge attentions
        _, edge_att = utils.get_edge_att(x, edge_index, edge_attr)

        # Add self-loops and set edge_att for self-loops to 1.0
        edge_index, edge_att = pyg.utils.add_self_loops(edge_index, edge_attr=edge_att, fill_value=1.0)

        # Perform softmax on the edge attentions
        edge_att = pyg.utils.softmax(src=edge_att, index=edge_index[1])

        # Linearly transform node feature matrix
        x = self.linear(x)

        # Propagate messages
        out = self.propagate(edge_index=edge_index, x=x, edge_att=edge_att)

        # Add bias
        out += self.bias

        return out

    def message(self, x_j, x, edge_att):
        return edge_att.view(-1, 1) * x_j
