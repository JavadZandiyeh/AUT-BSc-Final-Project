import tqdm
import torch
import datetime
import utils
import metrics
from utils import EngineSteps


def train_step(model, data, optimizer, loss_fn, batch_size):
    results = metrics.init_metrics(0)

    model.train()

    """ batch preparation """
    edge_index_train = data.edge_index[:, data.edge_mask_train]
    batches_mask = utils.mini_batching(edge_index_train, batch_size)
    for i in range(batch_size):
        batch_mask_indices = (data.edge_mask_train.nonzero().T.squeeze())[batches_mask[i]]
        new_batch_mask = torch.zeros_like(data.edge_index[0]).bool()
        new_batch_mask[batch_mask_indices] = True
        batches_mask[i] = new_batch_mask

    """ loop over batches """
    for num_batch, batch_mask in enumerate(batches_mask):
        """ predict y """
        y_pred = model(data)

        y_pred_batch, y_batch = y_pred[batch_mask], data.y[batch_mask]

        loss = loss_fn(y_pred_batch, y_batch)

        """ calculate batch results """
        batch_results = metrics.calculate_metrics(data, batch_mask, y_pred_batch, y_batch, loss)
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
        edge_mask_eval = data.edge_mask_val if eval_type == EngineSteps.VAL else data.edge_mask_test

        """ batch preparation """
        edge_index_eval = data.edge_index[:, edge_mask_eval]
        batches_mask = utils.mini_batching(edge_index_eval, batch_size)
        for i in range(batch_size):
            batch_mask_indices = (edge_mask_eval.nonzero().T.squeeze())[batches_mask[i]]
            new_batch_mask = torch.zeros_like(data.edge_index[0]).bool()
            new_batch_mask[batch_mask_indices] = True
            batches_mask[i] = new_batch_mask

        """ loop over batches """
        for num_batch, batch_mask in enumerate(batches_mask):
            """ predict y """
            y_pred = model(data)

            y_pred_batch, y_batch = y_pred[batch_mask], data.y[batch_mask]

            loss = loss_fn(y_pred_batch, y_batch)

            """ calculate batch results """
            batch_results = metrics.calculate_metrics(data, batch_mask, y_pred_batch, y_batch, loss)
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
