import configparser
import os
import pprint

import torch

import engine
import models
import utils

os.environ['TORCH'] = torch.__version__

config = configparser.ConfigParser()
config.read('config.ini')

phase = config.get('run', 'phase')

data_path = config.get('run', 'data_path')
data = torch.load(data_path).to(utils.device)


def get_model(model_name):
    channel = data.num_node_features

    if model_name == 'GATv2ConvModel':
        model = models.GATv2ConvModel(
            channels=[channel, channel, channel, channel]
        )
    elif model_name == 'LightGCNModel':
        model = models.LightGCNModel(
            num_nodes=data.num_nodes,
            embedding_dim=channel,
            num_layers=3,
            embedding=data.x.clone()
        )
    elif model_name == 'BigraphGATv2Model':
        model = models.BigraphGATv2Model(
            channels_ii=[channel, channel, channel, channel],
            channels_uiu=[channel, channel, channel, channel]
        )
    else:
        model = models.BigraphLightModel(
            num_nodes_ii=data.num_items,
            num_nodes_uiu=data.num_nodes,
            embedding_dim=channel,
            num_layers_ii=3,
            num_layers_uiu=3
        )

    return model


def get_settings(section):
    settings = {
        'run_name': config.get(section, 'run_name'),
        'model_name': config.get(section, 'model_name'),
        'epochs': config.getint(section, 'epochs'),
        'learning_rate': config.getfloat(section, 'learning_rate'),
        'num_batches': config.getint(section, 'num_batches'),
        'pos_sampling_rate': config.getfloat(section, 'pos_sampling_rate'),
        'neg_sampling_rate': config.getfloat(section, 'neg_sampling_rate')
    }

    return settings


def start_train(model, loss_fn, settings):
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])

    writer = utils.create_summary_writer(settings)

    engine.start(model, data, optimizer, loss_fn, writer, settings)

    writer.close()

    utils.save_model(model, settings)


def start_test(model, loss_fn, settings):
    model = utils.load_model(model, settings)

    test_loss, test_results = engine.eval_step(model, data, loss_fn, utils.EngineSteps.TEST)

    pprint.pprint({'loss': test_loss} | test_results)


if __name__ == '__main__':
    _settings = get_settings(phase)
    _model = get_model(_settings['model_name']).to(utils.device)
    _loss_fn = torch.nn.MSELoss().to(utils.device)

    if phase == 'train':
        start_train(_model, _loss_fn, _settings)
    else:
        start_test(_model, _loss_fn, _settings)
