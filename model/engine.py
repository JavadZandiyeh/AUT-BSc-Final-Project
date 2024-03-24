import tqdm
import torch
import datetime
import utils
import metrics
from utils import EngineSteps
import pprint


def train_step(model, data, optimizer, loss_fn, batch_size):
    results = metrics.init_metrics(0)

    model.train()

    edge_index_train = data.edge_index[:, data.edge_mask_train]

    batches = utils.mini_batching(edge_index_train, batch_size)

    """ loop over batches """
    for num_batch, batch in enumerate(batches):
        """ create batch_mask """
        batch = (data.edge_mask_train.nonzero().T.squeeze())[batch]
        batch_mask = torch.zeros_like(data.edge_index[0]).bool()
        batch_mask[batch] = True

        y_pred = model(data)

        y_pred_batch, y_batch = y_pred[batch_mask], data.y[batch_mask]

        loss = loss_fn(y_pred_batch, y_batch)

        """ calculate batch results """
        users = data.node_mask_user.nonzero().squeeze().tolist()
        edge_index_batch = data.edge_index[:, batch_mask]
        batch_results = metrics.calculate_metrics(users, edge_index_batch, y_pred_batch, y_batch, loss.item())

        """ calculate overall results """
        results = {metric: results[metric] + batch_result for metric, batch_result in batch_results.items()}

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    """ average results based on batch_size """
    results = {metric: result / batch_size for metric, result in results.items()}

    return results


def eval_step(model, data, loss_fn, eval_type):
    model.eval()

    with torch.inference_mode():
        edge_mask_eval = data.edge_mask_val if eval_type == EngineSteps.VAL else data.edge_mask_test

        y_pred = model(data)

        y_pred_eval, y_eval = y_pred[edge_mask_eval], data.y[edge_mask_eval]

        loss = loss_fn(y_pred_eval, y_eval)

        """ calculate results """
        users = data.node_mask_user.nonzero().squeeze().tolist()
        edge_index_eval = data.edge_index[:, edge_mask_eval]
        results = metrics.calculate_metrics(users, edge_index_eval, y_pred_eval, y_eval, loss.item())

    return results


def start(model, data, optimizer, loss_fn, epochs, batch_size, pos_sampling_rate, neg_sampling_rate, writer):
    for epoch in tqdm.tqdm(range(epochs)):
        print(datetime.datetime.now())

        sampled_data = utils.edge_sampling(data, pos_sampling_rate, neg_sampling_rate)

        train_results = train_step(model, sampled_data, optimizer, loss_fn, batch_size)

        val_results = eval_step(model, sampled_data, loss_fn, EngineSteps.VAL)

        utils.epoch_summary_write(writer, epoch, train_results, val_results)

    test_results = eval_step(model, data, loss_fn, EngineSteps.TEST)

    pprint.pprint(test_results)
