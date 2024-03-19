import configparser
import os

import torch
import engine
import models
import utils

""" Basic setups """
# Read configs
config = configparser.ConfigParser()
config.read('config.ini')

# Setup hyperparameters
epochs = config.getint('model', 'epochs')
learning_rate = config.getfloat('model', 'learning_rate')
batch_size = config.getint('model', 'batch_size')

# Setup device and environment
device = utils.get_device()
os.environ['TORCH'] = torch.__version__

# Setup dataset
data_base_path = config.get('dataset', 'imdb_1m_path')
data = torch.load(f'{data_base_path}/data.pt')
data = data.to(device)

# Setup writer
writer_base_path = config.get('writer', 'base_path')
writer_experiment_name = config.get('writer', 'experiment')
writer_model_name = config.get('writer', 'model')
writer_extra = config.get('writer', 'extra')

if __name__ == '__main__':
    channel = data.num_node_features

    # case 0
    # model = model_builder.BigraphModel(
    #     channels_ii=[channel, channel, channel],
    #     channels_uiu=[channel, channel, channel]
    # )

    # case 1
    model = models.GATv2ConvModel(
        channels=[channel, channel, channel]
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = torch.nn.MSELoss().to(device)

    writer = utils.create_summary_writer(writer_base_path, writer_experiment_name, writer_model_name, writer_extra)

    engine.start(
        model=model,
        data=data,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        batch_size=batch_size,
        writer=writer
    )

    writer.close()
