import tqdm
import torch
import utils
import metrics
from utils import EngineSteps
import pprint


def train_step(model, data, optimizer, loss_fn, num_batches, pos_sampling_rate, neg_sampling_rate):
    train_loss = 0

    model.train()

    sampled_data = utils.edge_sampling(data, pos_sampling_rate, neg_sampling_rate)

    edge_index_train = sampled_data.edge_index[:, sampled_data.edge_mask_train]
    indices_batches = utils.mini_batching(edge_index_train, num_batches)

    for num_batch, indices_batch in enumerate(indices_batches):
        # create an edge mask for this batch
        indices_batch = (sampled_data.edge_mask_train.nonzero().T.squeeze())[indices_batch]
        edge_mask_batch = torch.zeros_like(sampled_data.edge_index[0]).bool()
        edge_mask_batch[indices_batch] = True

        h_batch = model(sampled_data)

        y_pred_batch = utils.edge_prediction(h_batch, sampled_data.edge_index[:, edge_mask_batch])
        y_batch = sampled_data.y[edge_mask_batch]

        loss = loss_fn(y_pred_batch, y_batch)

        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss /= num_batches

    return train_loss


def eval_step(model, data, loss_fn, eval_type: EngineSteps, topk=10, calc_results=True):
    model.eval()

    with torch.inference_mode():
        edge_mask_eval = data.edge_mask_val if eval_type == EngineSteps.VAL else data.edge_mask_test

        h_eval = model(data)

        y_pred_eval = utils.edge_prediction(h_eval, data.edge_index[:, edge_mask_eval])
        y_eval = data.y[edge_mask_eval]

        loss = loss_fn(y_pred_eval, y_eval)

        val_loss = loss.item()

        results = metrics.MetricsCalculation(data, h_eval, eval_type, topk).get_results() if calc_results else None

    return val_loss, results


def start(model, data, optimizer, loss_fn, writer, settings):
    for epoch in tqdm.tqdm(range(settings['epochs'])):
        calc_results = (epoch % 10 == 0)  # calculate result every 10 epochs

        train_loss = train_step(
            model=model,
            data=data,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_batches=settings['num_batches'],
            pos_sampling_rate=settings['pos_sampling_rate'],
            neg_sampling_rate=settings['neg_sampling_rate']
        )

        val_loss, val_results = eval_step(
            model=model,
            data=data,
            loss_fn=loss_fn,
            eval_type=EngineSteps.VAL,
            topk=settings['topk'],
            calc_results=calc_results
        )

        calc_results and utils.epoch_summary_write(writer, epoch, train_loss, val_loss, val_results)

    test_loss, test_results = eval_step(
        model=model,
        data=data,
        loss_fn=loss_fn,
        eval_type=EngineSteps.TEST,
        topk=settings['topk']
    )

    pprint.pprint({'loss': test_loss} | test_results)
