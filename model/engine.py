import tqdm
import torch
import datetime
import enum
import utils
from torcheval.metrics import functional


class EvalType(enum.Enum):
    VAL = 'validation'
    TEST = 'test'


def train_step(model, data, optimizer, loss_fn, batch_size):
    results = {
        'loss': 0,
        'accuracy': 0,
    }

    model.train()

    batches = utils.mini_batching(data.edge_index[:, data.edge_mask_train], batch_size)

    for num_batch, batch in enumerate(batches):
        y_pred = model(data)
        edge_mask_indices = (data.edge_mask_train.nonzero().T.squeeze())[batch]
        y_predicted, y_actual = y_pred[edge_mask_indices], data.y[edge_mask_indices]

        loss = loss_fn(y_predicted, y_actual)
        accuracy = functional.r2_score(y_predicted, y_actual)

        results['loss'] += loss.item()
        results['accuracy'] += accuracy.item()

        optimizer.zero_grad()

        loss.backward(retain_graph=True)    # retain_graph might be changed

        optimizer.step()

    results['loss'] /= batch_size
    results['accuracy'] /= batch_size

    return results


def eval_step(model, data, loss_fn, eval_type, batch_size):
    results = {
        'loss': 0,
        'accuracy': 0
    }

    model.eval()

    with torch.inference_mode():
        mask = data.edge_mask_val if eval_type == EvalType.VAL else data.edge_mask_test

        batches = utils.mini_batching(data.edge_index[:, mask], batch_size)

        for num_batch, batch in enumerate(batches):
            y_pred = model(data)
            edge_mask_indices = (mask.nonzero().T.squeeze())[batch]
            y_predicted, y_actual = y_pred[edge_mask_indices], data.y[edge_mask_indices]

            loss = loss_fn(y_predicted, y_actual)
            accuracy = functional.r2_score(y_predicted, y_actual)

            results['loss'] += loss.item()
            results['accuracy'] += accuracy.item()

        results['loss'] /= batch_size
        results['accuracy'] /= batch_size

    return results


def start(model, data, optimizer, loss_fn, epochs, batch_size, writer):

    for epoch in tqdm.tqdm(range(epochs)):
        print(datetime.datetime.now())

        """ positive and/or negative sampling """
        sampled_data = utils.edge_sampling(data)

        """ train """
        train_results = train_step(
            model=model,
            data=sampled_data,
            optimizer=optimizer,
            loss_fn=loss_fn,
            batch_size=batch_size
        )

        """ validation """
        val_results = eval_step(
            model=model,
            data=sampled_data,
            loss_fn=loss_fn,
            eval_type=EvalType.VAL,
            batch_size=batch_size
        )

        """ test """
        test_results = eval_step(
            model=model,
            data=sampled_data,
            loss_fn=loss_fn,
            eval_type=EvalType.TEST,
            batch_size=batch_size
        )

        """ manage results """
        utils.epoch_summary_write(writer, epoch, train_results, val_results, test_results)
