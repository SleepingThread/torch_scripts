import os
import sys
import random
import pickle
import functools

import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
import torchmetrics
import torchvision
from torchvision import transforms

from torch_prune.models.var_dropout_vgglike import VGGLike
from torch_prune.trainers import train_loop, TBLogsWriter


class VariationalDropout(object):
    def __init__(self, modules, vd_lambda=None, normal_stddev=1., initial_logalpha=-8., logalpha_threshold=3.,
                 logalpha_clip_values=(-8., 8.), deterministic=False):
        """
        We can treat VariationalDropout as hooks container
        ??: How it will be saved/loaded with torch.save/torch.load
        It should be all OK, because function/object - there is no big difference
        The other question - how pickle treat the same object <self> while loading.
        Will it create multiple VariationalDropout objects? - it's not good at all.

        Parameters
        ==========
        modules: list of (module, <dict with config>)

        Example
        =======
        vd = VariationalDropout([(model.linear, None)])  # all specified modules support vd
        """
        self.modules = modules
        self.normal_stddev = normal_stddev
        self.initial_logalpha = initial_logalpha
        self.logalpha_threshold = logalpha_threshold
        self.logalpha_clip_values = logalpha_clip_values

        self.deterministic = deterministic

        self.vd_lambda = vd_lambda

        self._modules_dict = None
        self._forward_hooks = list()
        self._forward_pre_hooks = list()

        self._build()

    def _build(self):
        """
        Add prehook and hook for all modules
        """
        self._modules_dict = dict()
        for _m, _cfg in self.modules:
            _cfg = _cfg if _cfg is not None else dict()
            self._modules_dict[_m] = _cfg
            _w_name = _cfg.get("weight", "weight")

            _w = getattr(_m, _w_name)
            delattr(_m, _w_name)
            _m.register_parameter(_w_name + "_orig", _w)
            _la = Parameter(torch.full(_w.shape, _cfg.get("init_logalpha", self.initial_logalpha)))
            _m.register_parameter(_w_name + "_logalpha", _la)
            _m.register_buffer(_w_name + "_mask", torch.zeros(*_w.shape, dtype=torch.bool))

            self._forward_pre_hooks.append(_m.register_forward_pre_hook(self.prehook))
            self._forward_hooks.append(_m.register_forward_hook(self.hook))

    def set_vd_lambda(self, vd_lambda):
        self.vd_lambda = vd_lambda

    def _base_prehook(self, module, _inputs):
        _cfg = self._modules_dict[module]
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
        C = -k1

        _res = 0.
        for _m, _cfg in self._modules_dict.items():
            _la = getattr(_m, _cfg.get("weight", "weight") + "_logalpha")
            _la = torch.clamp(_la, min=self.logalpha_clip_values[0], max=self.logalpha_clip_values[1])
            mdkl = k1 * torch.sigmoid(k2 + k3 * _la) - 0.5 * torch.log1p(torch.exp(-_la)) + C
            _res += -torch.sum(mdkl)

        return self.vd_lambda * _res

    def get_nonzero_weights(self):
        _nonzero = 0.
        _all = 0.
        for _m, _cfg in self._modules_dict.items():
            _w_name = _cfg.get("weight", "weight")
            _mask = getattr(_m, _w_name + "_mask")
            _nonzero = torch.sum(_mask).item()
            _all = np.prod(_mask.shape)

        return _nonzero / _all

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

    def prehook(self, module, input):
        _method_name = "_prehook_" + module.__class__.__name__.lower()
        return getattr(self, _method_name)(module, input)

    def hook(self, module, input, output):
        if not self.deterministic:
            _method_name = "_hook_" + module.__class__.__name__.lower()
            return getattr(self, _method_name)(module, input, output)
        return output


class ZCA(object):
    def __init__(self, dtype=None, regularization=1.0e-5):
        self.regularization = regularization
        self.dtype = dtype

        self.mean = None
        self.ZCA_mat = None

    def fit(self, x):
        sys.stdout.write("Fitting ZCA ...")
        x = x.astype("float64")
        x = x.reshape((x.shape[0], np.prod(x.shape[1:])))
        self.mean = np.mean(x, axis=0)
        x -= self.mean

        sigma = np.dot(x.T, x) / x.shape[0]
        U, S, V = np.linalg.svd(sigma)

        tmp = np.dot(U, np.diag(1. / np.sqrt(S + self.regularization)))
        tmp2 = np.dot(U, np.diag(np.sqrt(S + self.regularization)))
        self.ZCA_mat = np.dot(tmp, U.T)
        sys.stdout.write("\rFitting ZCA done\n")
        return self

    def transform(self, x):
        _shape = x.shape
        x = x.astype("float64").reshape((x.shape[0], np.prod(x.shape[1:])))
        x = np.dot(x - self.mean, self.ZCA_mat)
        x = x.reshape(_shape)
        return x if self.dtype is None else x.astype(self.dtype)


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, transform=None, target_transform=None):
        self.data = data
        self.target = target

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def get_CIFAR10_ZCA(root, train_transform=None, test_transform=None, train_limit=None, test_limit=None):
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, transform=None, download=True)
    testset = torchvision.datasets.CIFAR10(root=root, train=False, transform=None)

    train_data = trainset.data
    test_data = testset.data

    zca = ZCA(dtype="float32")
    train_data = zca.fit(train_data).transform(train_data)
    test_data = zca.transform(test_data)

    train_limit = train_limit if train_limit is not None else len(train_data)
    test_limit = test_limit if test_limit is not None else len(test_data)

    trainset = NumpyDataset(train_data[:train_limit], trainset.targets[:train_limit], transform=train_transform)
    testset = NumpyDataset(test_data[:test_limit], testset.targets[:test_limit], transform=test_transform)

    return trainset, testset


class LRScheduler(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def on_epoch_begin(self, logs):
        _lr = 2.0e-5 - 1.0e-5 * min(logs["epoch"] / 100., 1.0)
        for _pg in self.optimizer.param_groups:
            _pg["lr"] = _lr
        logs["lr"] = _lr


class VDLambdaScheduler(object):
    def __init__(self, var_dropout, max_value):
        self.var_dropout = var_dropout
        self.max_value = max_value

    def on_epoch_begin(self, logs):
        epoch = logs["epoch"]
        vd_lambda = self.max_value * min(max(0, epoch - 5)/15., 1.0)
        self.var_dropout.set_vd_lambda(vd_lambda)
        logs["vd_lambda"] = vd_lambda
        logs["nonzero_weights"] = self.var_dropout.get_nonzero_weights()


seed_value = 0

random.seed(seed_value)
torch.manual_seed(seed_value)
np.random.seed(seed_value)

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = VGGLike(10, 1, use_dropout=False).to(device)

modules = functools.reduce(lambda _a, _x: _a+_x,
                           [[_cbr.conv for _cbr in _blk.conv_bn_rectify] for _blk in model.blocks]) + \
          [model.final[0], model.final[3]]
modules = list(zip(modules, [dict() for _i in range(len(modules))]))
vd = VariationalDropout(modules)

train_transform = transforms.Compose([
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

epochs = 200
train_limit = test_limit = None

if os.environ.get("TEST_MODE", "FALSE").upper() == "TRUE":
    epochs = 200
    train_limit=30
    test_limit=10

trainset, testset = get_CIFAR10_ZCA(os.path.join(os.environ["DATA"], "cifar10"), train_transform=train_transform,
                                    test_transform=test_transform, train_limit=train_limit, test_limit=test_limit)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


train_g = torch.Generator()
train_g.manual_seed(seed_value)
test_g = torch.Generator()
test_g.manual_seed(seed_value)
loaders = {"train": DataLoader(trainset, batch_size=100, generator=train_g, worker_init_fn=seed_worker),
           "test": DataLoader(testset, batch_size=100, generator=test_g, worker_init_fn=seed_worker)}
decay = [_p for _n, _p in model.named_parameters() if not _n.endswith(".bias") and _p.requires_grad]
no_decay = [_p for _n, _p in model.named_parameters() if _n.endswith(".bias") and _p.requires_grad]
opt = torch.optim.Adam([{'params': no_decay, 'weight_decay': 0.},
                        {'params': decay, 'weight_decay': 1.0e-5}], lr=2.0e-5)

ce_loss = torch.nn.CrossEntropyLoss()
ce_loss.__name__ = "cross_entropy"

def loss(output, target):
    return ce_loss(output, target) + vd.get_dkl()

top_1 = torchmetrics.Accuracy(top_k=1).to(device)
top_1.__name__ = "top_1"

logs_writer = TBLogsWriter("./", writers_keys=list(loaders.keys()),
                           histograms={"logalphas": [_p for _n, _p in model.named_parameters()
                                                     if _n.endswith("logalpha")]})
logs_writer.add_info({"model": "Molchanov's VGGLike(10, 1, use_dropout=False)",
                      "comment": "Reproduce simple VD(with logalpha)"})

logs = train_loop(model, loss, loaders, opt, epochs, device=device, metrics=[ce_loss, top_1],
                  callbacks=[LRScheduler(opt), VDLambdaScheduler(vd, 1./50000.), logs_writer])

with open("logs.pkl", "wb") as logs_file:
    pickle.dump(logs, logs_file)

torch.save(model, "model.torch")
