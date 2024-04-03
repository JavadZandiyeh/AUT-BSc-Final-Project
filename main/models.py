import torch
import torch_geometric as pyg


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
            node_mask = torch.ones(forward_out.size(0), dtype=torch.bool)

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


class BigraphGATv2Model(torch.nn.Module):
    def __init__(self, channels_ii: list, channels_uiu: list):
        super().__init__()

        self.num_layers_ii, self.num_layers_uiu = len(channels_ii) - 1, len(channels_uiu) - 1

        layers_ii = [pyg.nn.GATv2Conv(channels_ii[i], channels_ii[i + 1], edge_dim=1, concat=False, fill_value=1)
                     for i in range(self.num_layers_ii)]

        layers_uiu = [pyg.nn.GATv2Conv(channels_uiu[i], channels_uiu[i + 1], edge_dim=1, concat=False, fill_value=1)
                      for i in range(self.num_layers_uiu)]

        self.layers = torch.nn.ModuleList(layers_ii + layers_uiu)

    def forward(self, data):
        """ item-item graph """
        edge_index_ii = data.edge_index[:, data.edge_mask_ii] - data.node_mask_item.nonzero()[0]
        edge_attr_ii = data.edge_attr[data.edge_mask_ii]

        h_ii = data.x[data.node_mask_item, :].clone()

        for i in range(self.num_layers_ii):
            h_ii = self.layers[i](h_ii, edge_index_ii, edge_attr_ii)

        """ user-item graph """
        edge_index_uiu = data.edge_index[:, data.edge_mask_uiu * data.edge_mask_train]
        edge_attr_uiu = data.edge_attr[data.edge_mask_uiu * data.edge_mask_train]

        h_uiu = data.x.clone()
        h_uiu[data.node_mask_item, :] = h_ii

        for i in range(self.num_layers_uiu):
            h_uiu = self.layers[self.num_layers_ii + i](h_uiu, edge_index_uiu, edge_attr_uiu)

        return h_uiu


class BigraphLightModel(torch.nn.Module):
    def __init__(self, num_nodes_ii, num_nodes_uiu, embedding_dim, num_layers_ii, num_layers_uiu, init_x_items=None):
        super().__init__()

        self.layers_ii = pyg.nn.LightGCN(
            num_nodes=num_nodes_ii,
            embedding_dim=embedding_dim,
            num_layers=num_layers_ii,
        )

        self.layers_uiu = pyg.nn.LightGCN(
            num_nodes=num_nodes_uiu,
            embedding_dim=embedding_dim,
            num_layers=num_layers_uiu,
        )

        if init_x_items is not None:
            self.layers_ii.embedding.weight.data = init_x_items.clone()

    def forward(self, data):
        """ item-item graph """
        edge_index_ii = data.edge_index[:, data.edge_mask_ii] - data.node_mask_item.nonzero()[0]  # start from zero
        edge_attr_ii = data.edge_attr[data.edge_mask_ii]

        h_ii = self.layers_ii.get_embedding(edge_index_ii, edge_attr_ii)

        self.layers_uiu.embedding.weight.data[data.node_mask_item, :] = h_ii.clone()

        """ user-item graph """
        edge_index_uiu = data.edge_index[:, data.edge_mask_uiu * data.edge_mask_train]
        edge_attr_uiu = data.edge_attr[data.edge_mask_uiu * data.edge_mask_train]

        h_uiu = self.layers_uiu.get_embedding(edge_index_uiu, edge_attr_uiu)

        self.layers_ii.embedding.weight.data = h_uiu[data.node_mask_item, :].clone()

        return h_uiu


class GATv2ConvModel(torch.nn.Module):
    def __init__(self, channels: list):
        super().__init__()
        self.num_layers = len(channels) - 1

        layers = [pyg.nn.GATv2Conv(channels[i], channels[i + 1], edge_dim=1, concat=False, fill_value=1)
                  for i in range(self.num_layers)]

        self.layers = torch.nn.ModuleList(layers)

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
    def __init__(self, num_nodes, embedding_dim, num_layers, alpha=None, init_x=None):
        super().__init__()

        self.light_gcn = pyg.nn.LightGCN(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            alpha=alpha
        )

        if init_x is not None:
            self.light_gcn.embedding.weight.data = init_x.clone()

    def forward(self, data):
        edge_index = data.edge_index[:, data.edge_mask_train]
        edge_attr = data.edge_attr[data.edge_mask_train]

        h = self.light_gcn.get_embedding(edge_index=edge_index, edge_weight=edge_attr)

        return h
