import numpy as np

import torch

from ..utils import Modifiers


class BaseMasker(object):
    def __init__(self, model, modules):
        self.model = model
        self.modules = modules

        self.modules_dict = dict()
        self.w_name_list = []
        self.w_cfg_list = []
        for _m, _cfg in self.modules:
            _cfg = dict() if _cfg is None else _cfg
            self.modules_dict[_m] = _cfg
            _cfg = [_cfg] if not isinstance(_cfg, (tuple, list)) else _cfg
            for _w_cfg in _cfg:
                self.w_name_list.append((_m, _w_cfg.get("weight", "weight")))
                self.w_cfg_list.append(_w_cfg)

        self._forward_hooks = list()
        self._forward_pre_hooks = list()

        if hasattr(model, "modifiers"):
            if not isinstance(model.modifiers, Modifiers):
                raise ValueError("model.modifiers should be instance of utils.Modifiers")
            old_masker = None
            for _k, _v in model.modifiers.elems.items():
                if isinstance(_v, BaseMasker):
                    old_masker = _v
                    del model.modifiers.elems[_k]
                    break

            if old_masker is not None:
                old_masker.remove()
                old_weights = set(old_masker.w_name_list)
                new_weights = set(self.w_name_list)
                if not old_weights.issubset(new_weights):
                    raise ValueError("You should specify configuration for all weights: %s" %
                                     str(old_weights - new_weights))
        else:
            model.modifiers = Modifiers()

        model.modifiers.elems[self.__class__.__name__] = self

    @staticmethod
    def get_config_weight_names(cfg):
        if isinstance(cfg, dict):
            return [cfg.get("weight", "weight")]
        assert isinstance(cfg, (tuple, list))
        return [_c.get("weight", "weight") for _c in cfg]

    def remove(self):
        for _hook in self._forward_pre_hooks + self._forward_hooks:
            _hook.remove()

    def get_weights(self):
        return [getattr(_m, _w_name + "_orig") for _m, _w_name in self.w_name_list]

    def get_masks(self):
        return [getattr(_m, _w_name + "_mask") for _m, _w_name in self.w_name_list]

    def count_weights(self):
        _all = 0.
        for _mask in self.get_masks():
            _all += np.prod(_mask.shape)

        return _all

    def count_nonzero_weights(self):
        _nonzero = 0.
        with torch.no_grad():
            for _mask in self.get_masks():
                _nonzero += torch.sum(_mask).item()

        return _nonzero


class Masker(BaseMasker):
    def __init__(self, model, modules):
        super(Masker, self).__init__(model, modules)

        self._build()

    def _build(self):
        """
        Add prehook and hook for all modules
        """
        for _m, _cfg in self.modules_dict.items():
            for _w_name in self.get_config_weight_names(_cfg):
                if not hasattr(_m, _w_name + "_orig"):
                    _w = getattr(_m, _w_name)
                    delattr(_m, _w_name)
                    _m.register_parameter(_w_name + "_orig", _w)

                    if not hasattr(_m, _w_name + "_mask"):
                        _m.register_buffer(_w_name + "_mask", torch.ones(*_w.shape, dtype=torch.bool))

            self._forward_pre_hooks.append(_m.register_forward_pre_hook(self.prehook))

    def prehook(self, module, _inputs):
        _cfg = self.modules_dict[module]
        for _w_name in self.get_config_weight_names(_cfg):

            # calculate masked weights
            _mask = getattr(module, _w_name + "_mask")
            _weight = getattr(module, _w_name + "_orig") * _mask
            setattr(module, _w_name, _weight)
