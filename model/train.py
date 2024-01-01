import configparser
import os

import torch
import utils
import model_builder
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

    model = model_builder.CustomizedGAT(in_channels=num_features, out_channels=2 * num_features)
    model(data_ii.x, data_ii.edge_index, data_ii.edge_attr)
