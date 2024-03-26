import torch
import torch_geometric as pyg
import utils


class CustomizedGAT(pyg.nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_attr, node_mask=None):
        """
        :param x: (num_nodes * in_channels)
        :param edge_index: (2 * num_edges)
        :param edge_attr: (num_edges)
        :param node_mask: (num_nodes)
        :return: (num_nodes * out_channels)
        """
        forward_out = x.clone()

        if node_mask is None:
            node_mask = torch.ones(forward_out.shape[0], dtype=torch.bool)

        forward_out[node_mask] = self.lin(forward_out[node_mask])

        # edge_attr = pyg.utils.softmax(src=edge_attr, index=edge_index[1])

        forward_out = self.propagate(x=forward_out, edge_index=edge_index, edge_attr=edge_attr, node_mask=node_mask)

        forward_out[node_mask] += self.bias

        forward_out[node_mask] = torch.sigmoid(forward_out[node_mask])

        return forward_out

    def message(self, x_j, edge_attr):
        """
        :param x_j: (num_edges * in_channels)
        :param edge_attr: (num_edges)
        :return: (num_edges * in_channels)
        """
        msg_out = edge_attr.view(-1, 1) * x_j

        return msg_out

    def aggregate(self, msg_out, x, edge_index, node_mask):
        """
        :param msg_out: (num_edges * in_channels)
        :param x: (num_nodes * in_channels)
        :param edge_index: (2 * num_edges)
        :param node_mask: (num_nodes)
        :return: (num_nodes * in_channels)
        """
        aggr_out = torch.zeros_like(x)

        for i in torch.nonzero(node_mask).squeeze():
            indices = torch.where(edge_index[1] == i)[0]
            msgs = msg_out[indices]

            aggr_out[i] = torch.mean(msgs, dim=0)

        return aggr_out

    def update(self, aggr_out, x):
        """
        :param aggr_out: (num_nodes * in_channels)
        :param x: (num_nodes * in_channels)
        :return: (num_nodes * out_channels)
        """
        update_out = aggr_out + x

        return update_out


class BigraphModel(torch.nn.Module):
    def __init__(self, channels_ii: list, channels_uiu: list):
        super().__init__()
        self.cgat1_ii = CustomizedGAT(channels_ii[0], channels_ii[1])
        self.cgat2_ii = CustomizedGAT(channels_ii[1], channels_ii[2])

        self.cgat1_uiu = CustomizedGAT(channels_uiu[0], channels_uiu[1])
        self.cgat2_uiu = CustomizedGAT(channels_uiu[1], channels_uiu[2])

    def forward(self, data):
        """ item-item graph """
        edge_index_ii, edge_attr_ii = data.edge_index[:, data.edge_mask_ii], data.edge_attr[data.edge_mask_ii]

        edge_attr1_ii = utils.get_edge_att(data.x, edge_index_ii, edge_attr_ii)
        x1_ii = self.cgat1_ii(data.x, edge_index_ii, edge_attr1_ii, data.node_mask_item)

        edge_attr2_ii = utils.get_edge_att(x1_ii, edge_index_ii, edge_attr_ii)
        x2_ii = self.cgat2_ii(x1_ii, edge_index_ii, edge_attr2_ii, data.node_mask_item)

        """ user-item graph """
        edge_index_uiu, edge_attr_uiu = data.edge_index[:, data.edge_mask_uiu], data.edge_attr[data.edge_mask_uiu]

        x1_uiu = self.cgat1_uiu(x2_ii, edge_index_uiu, edge_attr_uiu)
        x2_uiu = self.cgat1_uiu(x1_uiu, edge_index_uiu, edge_attr_uiu)

        x2_uiu_j = torch.index_select(x2_uiu, 0, edge_index_uiu[0])
        x2_uiu_i = torch.index_select(x2_uiu, 0, edge_index_uiu[1])

        edge_y_pred = torch.zeros_like(data.edge_attr)
        edge_y_pred[data.edge_mask_uiu] = torch.nn.functional.cosine_similarity(x2_uiu_j, x2_uiu_i, dim=1)

        return edge_y_pred


class GATv2ConvModel(torch.nn.Module):
    def __init__(self, channels: list):
        super().__init__()

        self.layers = torch.nn.ModuleList()

        for i in range(len(channels) - 1):
            self.layers.add_module(
                f'gat{i + 1}',
                pyg.nn.GATv2Conv(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    edge_dim=1,
                    concat=False,
                    fill_value=1
                )
            )

    def forward(self, data):
        # prepare required variables
        edge_index = data.edge_index[:, data.edge_mask_train]
        edge_attr = data.edge_attr[data.edge_mask_train]
        h = data.x.clone()

        # forward over layers
        for layer in self.layers:
            h = layer(x=h, edge_index=edge_index, edge_attr=edge_attr)

        return h


class LightGCNModel(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, num_layers, alpha):
        super().__init__()

        self.light_gcn = pyg.nn.LightGCN(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            alpha=alpha
        )

    def forward(self, data):
        # prepare required variables
        edge_index = data.edge_index[:, data.edge_mask_train]
        edge_attr = data.edge_attr[data.edge_mask_train]
        h = data.x.clone()

        self.light_gcn.embedding = torch.nn.Embedding(
            num_embeddings=self.num_nodes,
            embedding_dim=self.embedding_dim,
            _weight=h
        )

        h = self.light_gcn.get_embedding(edge_index=edge_index, edge_weight=edge_attr)

        return h
