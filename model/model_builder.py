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


class BigraphModel(torch.nn.Module):
    def __init__(self, channels_ii: list, channels_ui: list):
        super().__init__()
        self.cgat1_ii = CustomizedGAT(channels_ii[0], channels_ii[1])
        self.cgat2_ii = CustomizedGAT(channels_ii[1], channels_ii[2])

        self.cgat1_ui = CustomizedGAT(channels_ui[0], channels_ui[1])
        self.cgat2_ui = CustomizedGAT(channels_ui[1], channels_ui[2])

    def forward(self, x_ii, edge_index_ii, edge_attr_ii, x_ui, edge_index_ui, edge_attr_ui):
        h1_ii = self.cgat1_ii(x_ii, edge_index_ii, edge_attr_ii)
        h2_ii = self.cgat2_ii(h1_ii, edge_index_ii, edge_attr_ii)

        x_ui_cloned = x_ui.clone()
        x_ui_cloned[:h2_ii.size(0)] = h2_ii

        h1_ui = self.cgat1_ui(x_ui_cloned, edge_index_ui, edge_attr_ui)
        h2_ui = self.cgat1_ui(h1_ui, edge_index_ui, edge_attr_ui)

        h2_ui_j = torch.index_select(h2_ui, 0, edge_index_ui[0])
        h2_ui_i = torch.index_select(h2_ui, 0, edge_index_ui[1])
        edge_att_ui = torch.nn.functional.cosine_similarity(h2_ui_i, h2_ui_j, dim=1)

        return edge_att_ui
