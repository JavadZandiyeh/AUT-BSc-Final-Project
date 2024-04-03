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

settings = {
    'run_name': config.get('settings', 'run_name'),
    'model_name': config.get('settings', 'model_name'),
    'epochs': config.getint('settings', 'epochs'),
    'learning_rate': config.getfloat('settings', 'learning_rate'),
    'num_batches': config.getint('settings', 'num_batches'),
    'pos_sampling_rate': config.getfloat('settings', 'pos_sampling_rate'),
    'neg_sampling_rate': config.getfloat('settings', 'neg_sampling_rate'),
    'topk': config.getint('settings', 'topk')
}


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
            init_x=data.x
        )
    elif model_name == 'BigraphGATv2Model':
        model = models.BigraphGATv2Model(
            channels_ii=[channel, channel, channel, channel],
            channels_uiu=[channel, channel, channel, channel]
        )
    else:  # BigraphLightModel
        model = models.BigraphLightModel(
            num_nodes_ii=data.num_items,
            num_nodes_uiu=data.num_nodes,
            embedding_dim=channel,
            num_layers_ii=3,
            num_layers_uiu=3,
            init_x_items=data.x[data.node_mask_item, :]
        )

    return model


def start_train(model, loss_fn):
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])

    writer = utils.create_summary_writer(settings)

    engine.start(
        model=model,
        data=data,
        optimizer=optimizer,
        loss_fn=loss_fn,
        writer=writer,
        settings=settings
    )

    writer.close()

    utils.save_model(model, settings)


def start_test(model, loss_fn):
    model = utils.load_model(model, settings)

    test_loss, test_results = engine.eval_step(
        model=model,
        data=data,
        loss_fn=loss_fn,
        eval_type=utils.EngineSteps.TEST,
        topk=settings['topk']
    )

    pprint.pprint({'loss': test_loss} | test_results)


if __name__ == '__main__':
    _model = get_model(settings['model_name']).to(utils.device)
    _loss_fn = torch.nn.MSELoss().to(utils.device)

    data.edge_attr[data.edge_mask_ii] = utils.get_edge_att(data.x, data.edge_index, data.edge_attr)[data.edge_mask_ii]

    start_train(_model, _loss_fn) if (phase == 'train') else start_test(_model, _loss_fn)
