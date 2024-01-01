import torch
import torch_geometric as pyg

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

        # Step 1: Compute edge attentions
        edge_index, edge_att = utils.get_edge_att(x, edge_index, edge_attr)

        # Step 2: Add self-loops and set edge_attr for self-loops to 1.0
        edge_index, edge_att = pyg.utils.add_self_loops(edge_index, edge_attr=edge_att, fill_value=1.0)

        # Step 3: Linearly transform node feature matrix
        x = self.linear(x)

        # Step 4: Propagate messages
        out = self.propagate(edge_index, x=x, edge_att=edge_att)

        # Step 5: Add bias
        out += self.bias

        return out

    def message(self, x_j, edge_att):
        return edge_att.view(-1, 1) * x_j
