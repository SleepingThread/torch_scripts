import numpy as np

import torch
from .utils import Masker


class WeakL2Masker(Masker):
    def __init__(self, model, modules, candle_epochs=10, l2_coef=None):
        super(WeakL2Masker, self).__init__(model, modules)

        self.candle_epochs = candle_epochs
        self.l2_coef = l2_coef

        self._candle_start_epoch = None
        self._cur_candle = None

        self.candles = []
        self.histograms = []

    def prehook(self, module, _inputs):
        _cfg = self.modules_dict[module]
        for _w_name in self.get_config_weight_names(_cfg):
            # set weight to be weight_orig
            _weight = getattr(module, _w_name + "_orig")
            setattr(module, _w_name, _weight)

    def set_l2_coef(self, value):
        self.l2_coef = value

    def get_l2(self):
        return self.l2_coef * torch.stack([
            torch.square(_w[torch.logical_not(_m)]).sum()
            for _w, _m in zip(self.get_weights(), self.get_masks())
        ]).sum()

    def decay_l2(self):
        with torch.no_grad():
            for _w, _m in zip(self.get_weights(), self.get_masks()):
                _not_m = torch.logical_not(_m)
                _w[_not_m] -= _w[_not_m]*self.l2_coef

    def _init_candle(self):
        self._cur_candle = ([_w.detach().clone() for _w in self.get_weights()],
                            [_w.detach().clone() for _w in self.get_weights()])

    def _update_candle(self):
        _candle_min = self._cur_candle[0]
        _candle_max = self._cur_candle[1]

        weights = [_w.detach() for _w in self.get_weights()]
        _candle_min = [torch.minimum(_e, _w, out=_e) for _e, _w in zip(_candle_min, weights)]
        _candle_max = [torch.maximum(_e, _w, out=_e) for _e, _w in zip(_candle_max, weights)]
        self._cur_candle = (_candle_min, _candle_max)

    def _save_candle(self):
        _candle_min = self._cur_candle[0]
        _candle_max = self._cur_candle[1]

        _candle_min = [_e.cpu().numpy() for _e in _candle_min]
        _candle_max = [_e.cpu().numpy() for _e in _candle_max]

        self.candles.append((_candle_min, _candle_max))

    def _new_candle(self):
        self._save_candle()
        self._cur_candle = None
        self._init_candle()

    def _build_histogram(self):
        res = np.concatenate([_w[torch.logical_not(_m)].detach().cpu().numpy()
                              for _w, _m in zip(self.get_weights(), self.get_masks())],
                             axis=-1)
        self.histograms.append(np.histogram(res, bins=100))

    def clear_stats(self):
        self._candle_start_epoch = None
        self._cur_candle = None

        self.candles = []
        self.histograms = []

    def on_train_begin(self):
        self._init_candle()
        self._new_candle()

        self._build_histogram()

    def on_train_end(self):
        self._save_candle()
        self._cur_candle = None

    def on_epoch_end(self, epoch_logs):
        _nz = self.count_nonzero_weights()
        _all = self.count_weights()
        self.model.info["nonzero_weights"] = _nz
        self.model.info["total_weights"] = _all
        epoch_logs["nonzero_weights"] = _nz
        epoch_logs["nonzero_perc"] = _nz / _all
        epoch_logs["l2_coef"] = self.l2_coef

        _e = epoch_logs["epoch"]
        if self._candle_start_epoch is None:
            self._candle_start_epoch = _e

        # build candles
        if _e - self._candle_start_epoch >= self.candle_epochs:
            self._candle_start_epoch = _e
            self._new_candle()
        else:
            self._update_candle()

        self._build_histogram()
