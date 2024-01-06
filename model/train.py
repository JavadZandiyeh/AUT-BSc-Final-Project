import configparser
import os

import torch

import model_builder
import networkx as nx
import torch_geometric as pyg

""" Basic setups """
# Read configs
config = configparser.ConfigParser()
config.read('config.ini')

# Setup hyperparameters
num_epochs = config.getint('model', 'num_epochs')
learning_rate = config.getfloat('model', 'learning_rate')

# Setup dataset
data_base_path = config.get('dataset', 'imdb_1m_path')
data_ii = torch.load(f'{data_base_path}/data_ii.pt')
data_ui_train = torch.load(f'{data_base_path}/data_ui_train.pt')
data_ui_test = torch.load(f'{data_base_path}/data_ui_test.pt')
data_ui_validation = torch.load(f'{data_base_path}/data_ui_validation.pt')

# Setup device and environment
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['TORCH'] = torch.__version__

if __name__ == '__main__':
    num_nodes, num_features = data_ii.x.size()

    # G = pyg.utils.to_networkx(data_ii, to_undirected=True)
    # connected_components = list(nx.connected_components(G))
    # for i, component in enumerate(connected_components, 1):
    #     print(len(component))
    #     subgraph = G.subgraph(component)
    #     diameter = nx.diameter(subgraph)
    #     print(f"Diameter of component {i}: {diameter}")

    model = model_builder.ItemItemModel(
        in_channels=num_features,
        hidden_channels=num_features,
        out_channels=num_features
    )

    out = model(data_ii.x, data_ii.edge_index, data_ii.edge_attr)

    # sample_tensor_normalized = torch.nn.functional.normalize(out, p=2, dim=1)
    # cosine_similarities = torch.matmul(sample_tensor_normalized, sample_tensor_normalized.t())
    # print(torch.min(cosine_similarities), torch.mean(cosine_similarities), torch.max(cosine_similarities))

    # flat_data = cosine_similarities.detach().numpy().flatten()
    # import matplotlib.pyplot as plt
    # plt.hist(flat_data, bins=30, color='blue', alpha=0.7)
    # plt.show()
