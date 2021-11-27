import os
import typing
import functools

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


class TBLogsWriter(object):
    def __init__(self, logs_dir, writers_keys=None, histograms=None):
        self.logs_dir = logs_dir
        self.histograms = dict() if histograms is None else histograms

        self.writers_keys = writers_keys
        if writers_keys is not None:
            self.writers = {_k: SummaryWriter(os.path.join(logs_dir, _k)) for _k in writers_keys}
            self.writers.update({"info": SummaryWriter(os.path.join(logs_dir, "info"))})
        else:
            self.writers = SummaryWriter(logs_dir)

    def add_scalars(self, logs, epoch):
        if isinstance(self.writers, dict):
            for _wk, _wv in logs.items():
                if isinstance(_wv, dict):
                    for _k, _v in _wv.items():
                        self.writers[_wk].add_scalar(_k, _v, epoch)
                elif _wk != "epoch":
                    self.writers["info"].add_scalar(_wk, _wv, epoch)
        else:
            for _k, _v in logs.items():
                self.writers.add_scalar(_k, _v, epoch)

    def _add_summary(self, summary):
        if self.writers_keys is not None:
            self.writers["info"].file_writer.add_summary(summary)
        else:
            self.writers.file_writer.add_summary(summary)

    def _add_histogram(self, name, weights, epoch):
        """
        weights: callable, list/tuple of weights, weight
        """
        _writer = self.writers["info"] if self.writers_keys is not None else self.writers

        with torch.no_grad():
            if callable(weights):
                weights = weights()

            if not isinstance(weights, (list, tuple)):
                weights = [weights]

            weights = [_w.cpu().numpy() if isinstance(_w, torch.Tensor) else _w for _w in weights]
            weights = np.concatenate([_w.reshape((-1,)) for _w in weights])
            _writer.add_histogram(name, weights, epoch)

    def add_info(self, info_dict):
        exp, ssi, sei = hparams(info_dict, {"_": ""}, None)
        self._add_summary(exp)
        self._add_summary(ssi)
        self._add_summary(sei)
        self.flush()

    def on_epoch_end(self, epoch_logs):
        self.add_scalars(epoch_logs, epoch_logs["epoch"])

        # process histograms, if needed
        for _k, _v in self.histograms.items():
            self._add_histogram(_k, _v, epoch_logs["epoch"])

        self.flush()

    def flush(self):
        if isinstance(self.writers, dict):
            for _writer in self.writers.values():
                _writer.flush()
        else:
            self.writers.flush()


class CallbacksList(object):
    def __init__(self, callbacks_list: typing.Optional[typing.List[typing.Any]]):
        self.callbacks_list = [] if callbacks_list is None else callbacks_list

    def _callback_method_call(self, cm_name, *args, **kwargs):
        for _c in self.callbacks_list:
            if hasattr(_c, cm_name):
                getattr(_c, cm_name)(*args, **kwargs)

    def __getattr__(self, item):
        return functools.partial(self._callback_method_call, item)


class LRSchedulerLinear(object):
    def __init__(self, optimizer, max_lr=2.0e-3, min_lr=1.0e-3, n_epochs=100):
        self.optimizer = optimizer

        self.max_lr = max_lr
        self.min_lr = min_lr
        self.n_epochs = n_epochs

    def on_epoch_begin(self, logs):
        _e = logs.get("epoch_lr", logs["epoch"]) % self.n_epochs
        _lr = self.max_lr - self.min_lr * min(_e / 100., 1.0)
        for _pg in self.optimizer.param_groups:
            _pg["lr"] = _lr
        logs["lr"] = _lr


class LRSchedulerCosine(object):
    def __init__(self, optimizer, max_lr=2.0e-3, n_epochs=200, cycle_epochs=200):
        self.optimizer = optimizer

        self.max_lr = max_lr
        self.n_epochs = n_epochs
        self.cycle_epochs = cycle_epochs

    def on_epoch_begin(self, logs):
        _e = logs.get("epoch_lr", logs["epoch"]) % self.cycle_epochs
        _lr = (1 + np.cos(_e * np.pi / self.n_epochs)) * 0.5 * self.max_lr if _e > 0 else self.max_lr
        for _pg in self.optimizer.param_groups:
            _pg["lr"] = _lr
        logs["lr"] = _lr
