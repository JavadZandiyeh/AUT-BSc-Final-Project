import tqdm
import torch
import datetime
import utils
import metrics
from utils import EngineSteps
import pprint


def train_step(model, data, optimizer, loss_fn, num_batches):
    results = metrics.init_metrics(0)

    model.train()

    # helpful vars
    users = data.node_mask_user.nonzero().squeeze().tolist()
    edge_index_train = data.edge_index[:, data.edge_mask_train]

    indices_batches = utils.mini_batching(edge_index_train, num_batches)

    for num_batch, indices_batch in enumerate(indices_batches):
        # create an edge mask for this batch
        indices_batch = (data.edge_mask_train.nonzero().T.squeeze())[indices_batch]
        edge_mask_batch = torch.zeros_like(data.edge_index[0]).bool()
        edge_mask_batch[indices_batch] = True

        # helpful vars
        edge_index_batch = data.edge_index[:, edge_mask_batch]
        y_batch = data.y[edge_mask_batch]

        h_batch = model(data)  # final embedding

        y_pred_batch = utils.edge_prediction(h_batch, edge_index_batch)

        loss = loss_fn(y_pred_batch, y_batch)

        results_batch = metrics.calculate_metrics(users, edge_index_batch, y_pred_batch, y_batch, loss.item())
        results = {metric: results[metric] + result_batch for metric, result_batch in results_batch.items()}

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    results = {metric: result / num_batches for metric, result in results.items()}

    return results


def eval_step(model, data, loss_fn, eval_type):
    model.eval()

    with torch.inference_mode():
        edge_mask_eval = data.edge_mask_val if eval_type == EngineSteps.VAL else data.edge_mask_test

        # helpful vars
        users = data.node_mask_user.nonzero().squeeze().tolist()
        edge_index_eval = data.edge_index[:, edge_mask_eval]
        y_eval = data.y[edge_mask_eval]

        h_eval = model(data)  # final embedding

        y_pred_eval = utils.edge_prediction(h_eval, edge_index_eval)

        loss = loss_fn(y_pred_eval, y_eval)

        results = metrics.calculate_metrics(users, edge_index_eval, y_pred_eval, y_eval, loss.item())

    return results


def start(model, data, optimizer, loss_fn, writer, settings):
    for epoch in tqdm.tqdm(range(settings['epochs'])):
        print(datetime.datetime.now())

        sampled_data = utils.edge_sampling(data, settings['pos_sampling_rate'], settings['neg_sampling_rate'])

        train_results = train_step(model, sampled_data, optimizer, loss_fn, settings['num_batches'])

        val_results = eval_step(model, sampled_data, loss_fn, EngineSteps.VAL)

        utils.epoch_summary_write(writer, epoch, train_results, val_results)

    test_results = eval_step(model, data, loss_fn, EngineSteps.TEST)

    pprint.pprint(test_results)
