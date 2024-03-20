import tqdm
import torch
import datetime
import utils
import metrics
from utils import EngineSteps


def train_step(model, data, optimizer, loss_fn, batch_size):
    results = metrics.init_metrics(0)

    model.train()

    batches = utils.mini_batching(data.edge_index[:, data.edge_mask_train], batch_size)

    for num_batch, batch in enumerate(batches):
        """ predict y """
        y_pred = model(data)
        edge_mask_indices = (data.edge_mask_train.nonzero().T.squeeze())[batch]
        y_pred, y = y_pred[edge_mask_indices], data.y[edge_mask_indices]

        loss = loss_fn(y_pred, y)

        """ calculate batch results """
        batch_results = metrics.calculate_metrics(y_pred, y, loss.item(), data, edge_mask_indices)
        for metric, value in batch_results.items():
            results[metric] += value

        optimizer.zero_grad()

        loss.backward(retain_graph=True)  # TODO: retain_graph might be changed

        optimizer.step()

    """ average over batches results """
    for metric, value in results.items():
        results[metric] /= batch_size

    return results


def eval_step(model, data, loss_fn, eval_type, batch_size):
    results = metrics.init_metrics(0)

    model.eval()

    with torch.inference_mode():
        mask = data.edge_mask_val if eval_type == EngineSteps.VAL else data.edge_mask_test

        batches = utils.mini_batching(data.edge_index[:, mask], batch_size)

        for num_batch, batch in enumerate(batches):
            """ predict y """
            y_pred = model(data)
            edge_mask_indices = (mask.nonzero().T.squeeze())[batch]
            y_pred, y = y_pred[edge_mask_indices], data.y[edge_mask_indices]

            loss = loss_fn(y_pred, y)

            """ calculate batch results """
            batch_results = metrics.calculate_metrics(y_pred, y, loss.item(), data, edge_mask_indices)
            for metric, value in batch_results.items():
                results[metric] += value

        """ average over batches results """
        for metric, value in results.items():
            results[metric] /= batch_size

    return results


def start(model, data, optimizer, loss_fn, epochs, batch_size, writer):
    for epoch in tqdm.tqdm(range(epochs)):
        print(datetime.datetime.now())

        sampled_data = utils.edge_sampling(data)

        train_results = train_step(model, sampled_data, optimizer, loss_fn, batch_size)

        val_results = eval_step(model, sampled_data, loss_fn, EngineSteps.VAL, batch_size)

        test_results = eval_step(model, sampled_data, loss_fn, EngineSteps.TEST, batch_size)

        utils.epoch_summary_write(writer, epoch, train_results, val_results, test_results)
