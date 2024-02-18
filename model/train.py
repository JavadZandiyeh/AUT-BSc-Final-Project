import configparser
import os

import torch
import engine
import model_builder
import utils

""" Basic setups """
# Read configs
config = configparser.ConfigParser()
config.read('config.ini')

# Setup hyperparameters
epochs = config.getint('model', 'epochs')
learning_rate = config.getfloat('model', 'learning_rate')

# Setup device and environment
device = utils.get_device()
os.environ['TORCH'] = torch.__version__

# Setup dataset
data_base_path = config.get('dataset', 'imdb_1m_path')
data = torch.load(f'{data_base_path}/data.pt')
data = data.to(device)

if __name__ == '__main__':
    channel = data.num_node_features

    model = model_builder.BigraphModel(
        channels_ii=[channel, channel, channel],
        channels_uiu=[channel, channel, channel]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = torch.nn.MSELoss().to(device)

    engine.train(
        model=model,
        data=data,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs
    )
