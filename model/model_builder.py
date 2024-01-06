import torch
import torch_geometric as pyg

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
        # Compute edge attentions
        _, edge_att = utils.get_edge_att(x, edge_index, edge_attr)

        # Add self-loops and set edge_att for self-loops to 1.0
        edge_index, edge_att = pyg.utils.add_self_loops(edge_index, edge_attr=edge_att, fill_value=1.0)

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
        return edge_att.view(-1, 1) * x_j


class ItemItemModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.cgat1 = CustomizedGAT(in_channels, hidden_channels)
        self.cgat2 = CustomizedGAT(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        h1 = self.cgat1(x, edge_index, edge_attr)
        h2 = self.cgat2(h1, edge_index, edge_attr)

        return h2
