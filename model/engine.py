import tqdm
import torch


def train_step(model, data, optimizer, loss_fn):
    train_loss = 0

    model.train()

    y_pred = model(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        edge_mask_ii=data.edge_mask_ii,
        edge_mask_uiu=data.edge_mask_uiu,
        node_mask_item=data.node_mask_item
    )

    loss = loss_fn(y_pred, data.y[data.edge_mask_uiu])  # (predicted, actual)

    train_loss += loss.item()

    optimizer.zero_grad()

    loss.backward(retain_graph=True)

    optimizer.step()

    train_loss /= torch.count_nonzero(data.edge_mask_ii)

    return train_loss


def validation_step():
    pass


def test_step():
    pass


def train(model, data, optimizer, loss_fn, epochs):

    for epoch in tqdm.tqdm(range(epochs)):
        train_loss = train_step(
            model=model,
            data=data,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )

        print(f'epoch: {epoch}, train_loss: {train_loss}')
