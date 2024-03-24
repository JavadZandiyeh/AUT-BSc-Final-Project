import configparser
import os

import torch

import engine
import models
import utils

config = configparser.ConfigParser()
config.read('config.ini')

device = utils.get_device()
os.environ['TORCH'] = torch.__version__

# Setup dataset
data_base_path = config.get('dataset', 'imdb_1m_path')
data = torch.load(f'{data_base_path}/data.pt')
data = data.to(device)

settings = {
    'epochs': config.getint('model', 'epochs'),
    'learning_rate': config.getfloat('model', 'learning_rate'),
    'num_batches': config.getint('model', 'num_batches'),
    'pos_sampling_rate': config.getfloat('model', 'pos_sampling_rate'),
    'neg_sampling_rate': config.getfloat('model', 'neg_sampling_rate'),
    'run_name': config.get('writer', 'run_name')
}

if __name__ == '__main__':
    channel = data.num_node_features

    # case 0
    # model = model_builder.BigraphModel(
    #     channels_ii=[channel, channel, channel],
    #     channels_uiu=[channel, channel, channel]
    # )

    # case 1
    model = models.GATv2ConvModel([channel, channel, channel])

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])

    loss_fn = torch.nn.MSELoss().to(device)

    writer = utils.create_summary_writer(model._get_name(), settings)

    engine.start(model, data, optimizer, loss_fn, writer, settings)

    writer.close()
