import numpy as np
import torch


def get_dataloader(data, *, batch_size, p_test, seed=0):
    idx = np.arange(len(data))

    np.random.default_rng(seed).shuffle(idx)

    max_i_train = int(len(data) * (1.0 - p_test))
    train_data = [data[i.item()] for i in idx[:max_i_train]]
    test_data = [data[i.item()] for i in idx[max_i_train:]]

    return (
        torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test_data, batch_size=batch_size),
    )


def weighted_avg(x):
    y, n = zip(*x)
    return np.sum(np.multiply(y, n)) / np.sum(n)


def loss_batch(model, loss_func, x, y, opt=None, model_args=None, cb=None):
    if model_args is None:
        model_args = []

    loss = loss_func(model(x, *model_args), y)
    if cb:
        cb(loss)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(x)


def fit_epoch(
    model,
    loss_func,
    opt,
    *,
    train_dl,
    test_dl=None,
    model_args=None,
    train_cb=None,
    test_cb=None
):
    model.train()
    loss_train = weighted_avg(
        loss_batch(model, loss_func, xy[:, 0], xy[:, 1:], opt, model_args, cb=train_cb)
        for xy in train_dl
    )

    if test_dl is None:
        return loss_train, None

    model.eval()
    with torch.no_grad():
        loss_test = weighted_avg(
            loss_batch(model, loss_func, xy[:, 0], xy[:, 1:], model_args, cb=test_cb)
            for xy in test_dl
        )

    return loss_train, loss_test
