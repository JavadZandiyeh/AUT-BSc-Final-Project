import tqdm
import torch
import datetime
import enum
import utils


class EvalType(enum.Enum):
    VAL = 'validation'
    TEST = 'test'


def train_step(model, data, optimizer, loss_fn, batch_size):
    train_loss = 0

    model.train()

    batches = utils.mini_batching(data.edge_index[:, data.edge_mask_train], batch_size)

    for num_batch, batch in enumerate(batches):
        y_pred = model(data)

        edge_mask_indices = (data.edge_mask_train.nonzero().T.squeeze())[batch]

        loss = loss_fn(y_pred[edge_mask_indices], data.y[edge_mask_indices])  # (predicted, actual)

        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward(retain_graph=True)    # retain_graph might be changed

        optimizer.step()

    train_loss /= batch_size

    val_loss = eval_step(model, data, loss_fn, EvalType.VAL, batch_size)

    return train_loss, val_loss


def eval_step(model, data, loss_fn, eval_type, batch_size):
    eval_loss = 0

    model.eval()

    with torch.inference_mode():
        mask = data.edge_mask_val if eval_type == EvalType.VAL else data.edge_mask_test

        batches = utils.mini_batching(data.edge_index[:, mask], batch_size)

        for num_batch, batch in enumerate(batches):
            y_pred = model(data)

            edge_mask_indices = (mask.nonzero().T.squeeze())[batch]

            loss = loss_fn(y_pred[edge_mask_indices], data.y[edge_mask_indices])  # (predicted, actual)

            eval_loss += loss.item()

    eval_loss /= batch_size

    return eval_loss


def start(model, data, optimizer, loss_fn, epochs, batch_size, writer):

    for epoch in tqdm.tqdm(range(1, epochs + 1)):
        print(datetime.datetime.now())

        """ positive and/or negative sampling """
        sampled_data = utils.edge_sampling(data)

        """ train and validation """
        train_loss, val_loss = train_step(
            model=model,
            data=sampled_data,
            optimizer=optimizer,
            loss_fn=loss_fn,
            batch_size=batch_size
        )

        """ test """
        test_loss = eval_step(
            model=model,
            data=sampled_data,
            loss_fn=loss_fn,
            eval_type=EvalType.TEST,
            batch_size=batch_size
        )

        """ manage results """
        results = {
            'loss': {
                'train': train_loss,
                'val': val_loss,
                'test': test_loss,
            },
        }

        utils.epoch_summary_write(writer, epoch, results)
