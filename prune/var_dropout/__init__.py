import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

from ..utils import BaseMasker


class BaseVariationalDropout(BaseMasker):
    """
    We can treat VariationalDropout as hooks container

    Parameters
    ==========
    modules: list of (module, <dict with config>)

    Example
    =======
    vd = VariationalDropout([(model.linear, None)])  # all specified modules support vd
    """

    def __init__(self, model, modules, vd_lambda=None, deterministic=False):
        super(BaseVariationalDropout, self).__init__(model, modules)
        self.deterministic = deterministic

        self.vd_lambda = vd_lambda

    def set_vd_lambda(self, vd_lambda):
        self.vd_lambda = vd_lambda

    def remove(self):
        for _hook in self._forward_pre_hooks + self._forward_hooks:
            _hook.remove()

    def get_supported_layers(self):
        _prehook = set()
        _hook = set()
        for _el in dir(self):
            if _el.startswith("_prehook_"):
                _lr_name = _el[len("_prehook_"):]
                _prehook.add(_lr_name)
            elif _el.startswith("_hook_"):
                _lr_name = _el[len("_hook_"):]
                _hook.add(_lr_name)
        return list(_prehook.intersection(_hook))

    def prehook(self, module, inp):
        _method_name = "_prehook_" + module.__class__.__name__.lower()
        return getattr(self, _method_name)(module, inp)

    def hook(self, module, inp, output):
        if not self.deterministic and module.training:
            _method_name = "_hook_" + module.__class__.__name__.lower()
            return getattr(self, _method_name)(module, inp, output)
        return output


class VariationalDropout(BaseVariationalDropout):
    def __init__(self, model, modules, vd_lambda=None, normal_stddev=1., initial_logalpha=-8., logalpha_threshold=3.,
                 logalpha_clip_values=(-8., 8.), deterministic=False):
        super(VariationalDropout, self).__init__(model, modules, vd_lambda=vd_lambda, deterministic=deterministic)

        self.normal_stddev = normal_stddev
        self.initial_logalpha = initial_logalpha
        self.logalpha_threshold = logalpha_threshold
        self.logalpha_clip_values = logalpha_clip_values

        self._build()

    def _build(self):
        """
        Add prehook and hook for all modules
        """
        for _m, _cfg in self.modules_dict.items():
            _w_name = _cfg.get("weight", "weight")

            _w = getattr(_m, _w_name)
            delattr(_m, _w_name)
            if not hasattr(_m, _w_name + "_orig"):
                _m.register_parameter(_w_name + "_orig", _w)
            if not hasattr(_m, _w_name + "_logalpha"):
                _la = Parameter(torch.full(_w.shape, _cfg.get("init_logalpha", self.initial_logalpha)))
                _m.register_parameter(_w_name + "_logalpha", _la)
            if not hasattr(_m, _w_name + "_mask"):
                _m.register_buffer(_w_name + "_mask", torch.ones(*_w.shape, dtype=torch.bool))

            self._forward_pre_hooks.append(_m.register_forward_pre_hook(self.prehook))
            self._forward_hooks.append(_m.register_forward_hook(self.hook))

    def _base_prehook(self, module, _inputs):
        _cfg = self.modules_dict[module]
        _w_name = _cfg.get("weight", "weight")

        # calculate masked weight
        _mask = getattr(module, _w_name + "_mask")
        _la = getattr(module, _w_name + "_logalpha")
        with torch.no_grad():
            _mask[:] = _la < self.logalpha_threshold

        _weight = getattr(module, _w_name + "_orig") * _mask
        setattr(module, _w_name, _weight)

    def _base_hook(self, module, inputs, outputs):
        pass

    def _prehook_linear(self, module, inputs):
        return self._base_prehook(module, inputs)

    def _hook_linear(self, module, inputs, outputs):
        _inp = inputs[0]
        _w = module.weight
        _la = torch.clamp(module.weight_logalpha, min=self.logalpha_clip_values[0], max=self.logalpha_clip_values[1])

        _vd_add = torch.sqrt((_inp * _inp) @ (torch.exp(_la) * _w * _w).t() + 1.0e-14)
        _rand = torch.normal(0., self.normal_stddev, _vd_add.shape, device=_vd_add.device)
        _vd_add = _rand * _vd_add

        return outputs + _vd_add

    def _prehook_conv2d(self, module, inputs):
        return self._base_prehook(module, inputs)

    def _hook_conv2d(self, module, inputs, outputs):
        _inp = inputs[0]
        _w = module.weight
        _la = torch.clamp(module.weight_logalpha, min=self.logalpha_clip_values[0], max=self.logalpha_clip_values[1])

        # convolve _inp*_inp with torch.exp(_la)*_w*_w, replace bias with None
        _inp = _inp * _inp
        _w = torch.exp(_la) * _w * _w
        if module.padding_mode != 'zeros':
            _vd_add = F.conv2d(F.pad(_inp, module._padding_repeated_twice, mode=module.padding_mode),
                               _w, None, module.stride,
                               torch.utils._pair(0), module.dilation, module.groups)
        else:
            _vd_add = F.conv2d(_inp, _w, None, module.stride,
                               module.padding, module.dilation, module.groups)

        _vd_add = torch.sqrt(_vd_add + 1.0e-14)
        _rand = torch.normal(0., self.normal_stddev, _vd_add.shape, device=_vd_add.device)
        _vd_add = _rand * _vd_add

        return outputs + _vd_add

    def get_dkl(self):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        c = -k1

        _res = 0.
        for _m, _cfg in self.modules_dict.items():
            _la = getattr(_m, _cfg.get("weight", "weight") + "_logalpha")
            _la = torch.clamp(_la, min=self.logalpha_clip_values[0], max=self.logalpha_clip_values[1])
            mdkl = k1 * torch.sigmoid(k2 + k3 * _la) - 0.5 * torch.log1p(torch.exp(-_la)) + c
            _res += -torch.sum(mdkl)

        return self.vd_lambda * _res


class VariationalDropoutLogsigma2(BaseVariationalDropout):
    def __init__(self, model, modules, vd_lambda=None, normal_stddev=1., initial_logsigma2=-10., logalpha_threshold=3.,
                 logalpha_clip_values=(-8., 8.), deterministic=False):
        super(VariationalDropoutLogsigma2, self).__init__(model, modules, vd_lambda=vd_lambda, deterministic=deterministic)

        self.normal_stddev = normal_stddev
        self.initial_logsigma2 = initial_logsigma2
        self.logalpha_threshold = logalpha_threshold
        self.logalpha_clip_values = logalpha_clip_values

        self._build()

    def _build(self):
        """
        Add prehook and hook for all modules
        """
        for _m, _cfg in self.modules_dict.items():
            _w_name = _cfg.get("weight", "weight")

            _w = getattr(_m, _w_name)
            delattr(_m, _w_name)
            if not hasattr(_m, _w_name + "_orig"):
                _m.register_parameter(_w_name + "_orig", _w)
            if not hasattr(_m, _w_name + "_logsigma2"):
                _ls = Parameter(torch.full(_w.shape, _cfg.get("init_logsigma2", self.initial_logsigma2)))
                _m.register_parameter(_w_name + "_logsigma2", _ls)
            if not hasattr(_m, _w_name + "_mask"):
                _m.register_buffer(_w_name + "_mask", torch.ones(*_w.shape, dtype=torch.bool))

            self._forward_pre_hooks.append(_m.register_forward_pre_hook(self.prehook))
            self._forward_hooks.append(_m.register_forward_hook(self.hook))

    def _base_prehook(self, module, _inputs):
        _cfg = self.modules_dict[module]
        _w_name = _cfg.get("weight", "weight")

        # calculate masked weight
        _mask = getattr(module, _w_name + "_mask")
        _w_orig = getattr(module, _w_name + "_orig")
        _ls = getattr(module, _w_name + "_logsigma2")
        with torch.no_grad():
            _la = _ls - torch.log(torch.square(_w_orig) + 1.0e-8)
            _mask[:] = _la < self.logalpha_threshold

        _weight = getattr(module, _w_name + "_orig") * _mask
        setattr(module, _w_name, _weight)

    def _base_hook(self, module, inputs, outputs):
        pass

    def _prehook_linear(self, module, inputs):
        return self._base_prehook(module, inputs)

    def _hook_linear(self, module, inputs, outputs):
        _inp = inputs[0]
        _w = module.weight
        _la = module.weight_logsigma2 - torch.log(torch.square(module.weight_orig) + 1.0e-8)
        _la = torch.clamp(_la, min=self.logalpha_clip_values[0], max=self.logalpha_clip_values[1])

        _vd_add = torch.sqrt((_inp * _inp) @ (torch.exp(_la) * _w * _w).t() + 1.0e-14)
        _rand = torch.normal(0., self.normal_stddev, _vd_add.shape, device=_vd_add.device)
        _vd_add = _rand * _vd_add

        return outputs + _vd_add

    def _prehook_conv2d(self, module, inputs):
        return self._base_prehook(module, inputs)

    def _hook_conv2d(self, module, inputs, outputs):
        _inp = inputs[0]
        _w = module.weight
        _la = module.weight_logsigma2 - torch.log(torch.square(module.weight_orig) + 1.0e-8)
        _la = torch.clamp(_la, min=self.logalpha_clip_values[0], max=self.logalpha_clip_values[1])

        # convolve _inp*_inp with torch.exp(_la)*_w*_w, replace bias with None
        _inp = _inp * _inp
        _w = torch.exp(_la) * _w * _w
        if module.padding_mode != 'zeros':
            _vd_add = F.conv2d(F.pad(_inp, module._padding_repeated_twice, mode=module.padding_mode),
                               _w, None, module.stride,
                               torch.utils._pair(0), module.dilation, module.groups)
        else:
            _vd_add = F.conv2d(_inp, _w, None, module.stride,
                               module.padding, module.dilation, module.groups)

        _vd_add = torch.sqrt(_vd_add + 1.0e-14)
        _rand = torch.normal(0., self.normal_stddev, _vd_add.shape, device=_vd_add.device)
        _vd_add = _rand * _vd_add

        return outputs + _vd_add

    def get_dkl(self):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        c = -k1

        _res = 0.
        for _m, _cfg in self.modules_dict.items():
            _w_orig = getattr(_m, _cfg.get("weight", "weight") + "_orig")
            _ls = getattr(_m, _cfg.get("weight", "weight") + "_logsigma2")
            _la = _ls - torch.log(torch.square(_w_orig) + 1.0e-8)
            _la = torch.clamp(_la, min=self.logalpha_clip_values[0], max=self.logalpha_clip_values[1])
            mdkl = k1 * torch.sigmoid(k2 + k3 * _la) - 0.5 * torch.log1p(torch.exp(-_la)) + c
            _res += -torch.sum(mdkl)

        return self.vd_lambda * _res
