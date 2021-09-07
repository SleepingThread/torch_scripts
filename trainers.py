import sys

import torch

from .callbacks import CallbacksList


class WeightedSumAggregator(object):
    def __init__(self):
        self.values = dict()
        self.weights = dict()

    @property
    def keys(self):
        return set(self.values.keys())

    def update(self, key, value, weight):
        # weight - batch size
        self.values[key] = self.values.get(key, 0.) + value * weight
        self.weights[key] = self.weights.get(key, 0.) + weight

    def get(self, key=None):
        if key is None:
            return {_k: _v / self.weights[_k] for _k, _v in self.values.items()}

        return self.values[key] / self.weights[key]


class MetricsAggregator(object):
    def __init__(self, metrics_list):
        if isinstance(metrics_list, (list, tuple)):
            def _get_name(metric_fun):
                return metric_fun.__name__

            self.metrics_list = {_get_name(_m): _m for _m in metrics_list}
        else:
            self.metrics_list = metrics_list
        self.wsa = WeightedSumAggregator()

        self._last_batch_size = None

    @property
    def keys(self):
        return self.wsa.keys

    def _get_batch(self, predicted):
        if isinstance(predicted, (tuple, list)):
            return self._get_batch(predicted[0])
        elif isinstance(predicted, dict):
            for _k, _v in predicted.items():
                return self._get_batch(_v)

        self._last_batch_size = int(predicted.shape[0])
        return self._last_batch_size

    @property
    def batch_size(self):
        return self._last_batch_size

    def reset(self):
        self.wsa = WeightedSumAggregator()

    def update(self, predicted, target):
        _bs = self._get_batch(predicted)
        for _metric_key, _m in self.metrics_list.items():
            _val = _m(predicted, target)
            _val = _val.item() if hasattr(_val, "item") else _val
            self.wsa.update(_metric_key, _val, weight=_bs)

    def get(self, key=None):
        return self.wsa.get(key=key)


def train_loop(model, loss_function, loaders, optimizer, epochs, start_epoch=0, device=None,
               metrics=None, logs=None, callbacks=None):
    """
    Parameters
    ==========
    model:
    loss_function:
    loaders: dict with torch.utils.data.DataLoader objects
    optimizer
    epochs: int
    start_epoch: int
    device: str (torch device)
    metrics: list or dict
    logs: list
    callbacks: list of callbacks or None

    Example
    =======
    from torch.nn.functional import mse_loss
    loaders = {"train": DataLoader(dataset, batch_size=32), "test": DataLoader(dataset, batch_size=32)}

    logs = train_loop(model, mse_loss, loaders, opt, 2, metrics=[mse_loss], logs_dir=None)
    """

    class _DummyContext(object):
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    device = "cpu" if device is None else device
    model.to(device)

    logs = list() if logs is None else logs
    metrics = list() if metrics is None else metrics

    ma = MetricsAggregator(metrics)
    cl = CallbacksList(callbacks)

    # on_train_begin
    cl.on_train_begin()

    for _e in range(start_epoch, start_epoch + epochs):

        epoch_logs = dict(epoch=_e)
        logs.append(epoch_logs)

        # on_epoch_begin
        cl.on_epoch_begin(epoch_logs)

        for _li, (_loader_name, _cur_loader) in enumerate(loaders.items()):
            wsa = WeightedSumAggregator()
            ma.reset()

            if _loader_name.startswith("train"):
                model.train()
                ctx = _DummyContext
            else:
                model.eval()
                ctx = torch.no_grad

            with ctx():
                for _bi, (_x, _y) in enumerate(_cur_loader):
                    sys.stdout.write("\r%128s" % "")
                    sys.stdout.write("\rEpoch: %d, Loader: %d, Batch: %d" % (_e, _li, _bi))

                    _x, _y = _x.to(device), _y.to(device)
                    pred = model(_x)

                    # metrics calculation
                    ma.update(pred, _y)

                    # loss calculation
                    loss = loss_function(pred, _y)
                    wsa.update("loss", loss.item(), ma.batch_size)

                    # logs
                    assert not ma.keys.intersection(wsa.keys)
                    epoch_logs[_loader_name] = ma.get()
                    epoch_logs[_loader_name].update(wsa.get())

                    if _loader_name.startswith("train"):
                        # gradient & update processing
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

        # on_epoch_end
        cl.on_epoch_end(epoch_logs)

        if hasattr(model, "stop_training") and getattr(model, "stop_training"):
            break

    # on_train_end
    cl.on_train_end()

    return logs
