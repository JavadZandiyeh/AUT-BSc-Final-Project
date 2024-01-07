import tqdm
import torch


def train_step(model, data_ii, data_ui, optimizer, loss_fn, device):
    train_loss = 0

    model.train()

    out = model(
        data_ii.x, data_ii.edge_index, data_ii.edge_attr,
        data_ui.x, data_ui.edge_index,
        torch.ones_like(data_ui.edge_attr)
    )

    loss = loss_fn(out, data_ui.edge_attr)  # (predicted, actual)

    train_loss += loss.item()

    optimizer.zero_grad()

    loss.backward(retain_graph=True)

    optimizer.step()

    train_loss /= data_ui.edge_index.size(1)

    return train_loss


def validation_step(model, data_ii, data_ui, optimizer, loss_fn, device):
    return 1, 2


def test_step(model, data_ii, data_ui, optimizer, loss_fn, device):
    return 1, 2


def train(model, data_ii, data_ui_train, data_ui_test, data_ui_validation, optimizer, loss_fn, epochs, device):

    for epoch in tqdm.tqdm(range(epochs)):
        train_loss = train_step(
            model=model,
            data_ii=data_ii,
            data_ui=data_ui_train,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device
        )

        # test_loss, test_acc = test_step(
        #     model=model,
        #     data_ii=data_ii,
        #     data_ui=data_ui_test,
        #     optimizer=optimizer,
        #     loss_fn=loss_fn,
        #     device=device
        # )

        # if epoch % 10 == 0:
        print(f'epoch: {epoch}, train_loss: {train_loss}')