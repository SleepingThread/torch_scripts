import numpy as np

import torch

from .utils import Masker


class Pruner(Masker):
    def __init__(self, model, modules, optimizer,
                 n_epochs_to_train=200,
                 perc_to_prune=10,
                 target_n_weights=100000.,
                 storage=None):
        super(Pruner, self).__init__(model, modules)

        self.optimizer = optimizer
        self.storage = storage

        self.n_epochs_to_train = n_epochs_to_train
        self.perc_to_prune = perc_to_prune
        self.target_n_weights = target_n_weights

        self._last_prune_epoch = 0
        self._optimizer_state_dict = self.optimizer.state_dict()

    def time_to_prune(self, epoch):
        return epoch % self.n_epochs_to_train == 0

    def _save_model(self):
        if self.storage is not None:
            self.storage.save(self.model, update=False)

    def prune(self):
        weights = self.get_weights()
        masks = self.get_masks()
        with torch.no_grad():
            shifts = [0]
            for _m in masks:
                shifts.append(shifts[-1] + _m.sum().item())
            w_values = [_w[_m].cpu().numpy() for _w, _m in zip(weights, masks)]
            w_values = np.abs(np.concatenate(w_values))
            w_to_prune = w_values > np.percentile(w_values, self.perc_to_prune)
            for _i in range(len(shifts) - 1):
                _m = masks[_i]
                _m[_m.clone()] = torch.from_numpy(w_to_prune[shifts[_i]:shifts[_i + 1]]).to(_m.device)

    def on_train_begin(self):
        self._optimizer_state_dict = self.optimizer.state_dict()

    def on_epoch_begin(self, epoch_logs):
        if self.time_to_prune(epoch_logs["epoch"]):
            if self.count_nonzero_weights() <= self.target_n_weights:
                self.model.stop_training = True
                return

            self._last_prune_epoch = epoch_logs["epoch"]
            self._save_model()
            self.optimizer.load_state_dict(self._optimizer_state_dict)
            self.prune()
        epoch_logs["epoch_lr"] = epoch_logs["epoch"] - self._last_prune_epoch

    def on_train_end(self):
        self._save_model()

    def on_epoch_end(self, epoch_logs):
        self.model.info["quality"] = epoch_logs.get("test", dict()).get("top_1", None)
        _nz = self.count_nonzero_weights()
        _all = self.count_weights()
        self.model.info["nonzero_weights"] = _nz
        self.model.info["total_weights"] = _all
        epoch_logs["nonzero_weights"] = _nz
        epoch_logs["nonzero_perc"] = _nz / _all
