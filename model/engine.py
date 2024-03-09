import tqdm
import torch
import datetime
from enum import Enum


class EvalType(Enum):
    VAL = 'validation'
    TEST = 'test'


def train_step(model, data, optimizer, loss_fn):
    train_loss = 0

    model.train()

    y_pred = model(data)

    loss = loss_fn(y_pred[data.edge_mask_train], data.y[data.edge_mask_train])  # (predicted, actual)

    train_loss += loss.item()

    optimizer.zero_grad()

    loss.backward(retain_graph=True)    # it is needed to be changed

    torch.nn.utils.clip_grad_norm_(model.parameters(), 3)

    optimizer.step()

    val_loss = eval_step(model, data, loss_fn, EvalType.VAL)

    return train_loss, val_loss


def eval_step(model, data, loss_fn, eval_type):
    eval_loss = 0

    model.eval()

    with torch.inference_mode():
        y_pred = model(data)

        mask = data.edge_mask_val if eval_type == EvalType.VAL else data.edge_mask_test
        loss = loss_fn(y_pred[mask], data.y[mask])  # (predicted, actual)

        eval_loss += loss.item()

    return eval_loss


def train(model, data, optimizer, loss_fn, epochs):

    for epoch in tqdm.tqdm(range(epochs)):
        print(datetime.datetime.now())

        train_loss, val_loss = train_step(
            model=model,
            data=data,
            optimizer=optimizer,
            loss_fn=loss_fn
        )

        test_loss = eval_step(
            model=model,
            data=data,
            loss_fn=loss_fn,
            eval_type=EvalType.TEST
        )

        print(f'epoch: {epoch+1}, train_loss: {train_loss}, val_loss: {val_loss}, test_loss: {test_loss}')
