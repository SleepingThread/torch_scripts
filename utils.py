import os
import random
import pickle
import time

import numpy as np

import torch
from torch import nn
import torch.backends.cudnn


def initialize_torch(seed_value, deterministic=True,
                     visible_gpus=None, cublas_workspace_config=":4096:8"):
    if visible_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus

    random.seed(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


class Modifiers(object):
    def __init__(self):
        self.__dict__["elems"] = dict()

    def __setattr__(self, name, value):
        if name in self.elems:
            raise KeyError("%s key is already in use" % name)
        self.elems[name] = value

    def __delattr__(self, name):
        if name in self.elems:
            del self.elems[name]
        else:
            super(Modifiers, self).__delattr__(name)

    def __getattr__(self, name):
        if name != "elems" and name in self.elems:
            return self.elems[name]
        raise AttributeError

    def __repr__(self):
        return str(self.elems)


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()

        self.modifiers = Modifiers()
        self.logs_dir = None
        self.storage_id = None
        self.storage_id_history = []
        self.info = dict()

    def initialize_logs_dir(self, root, prefix, suffix=None):
        """
        logs_dir = <root>/<prefix>[_suffix][_<unique_id>]
        suffix = None | <string for strftime>
        """
        _name = prefix

        if suffix is not None:
            _suffix = time.strftime(suffix)
            _name += "_" + _suffix

        root = os.path.join(os.path.abspath(root), _name)

        _logs_dir = root
        unique_id = 1
        while True:
            if not os.path.exists(_logs_dir):
                try:
                    os.makedirs(_logs_dir)
                    break
                except FileExistsError:
                    pass
            if unique_id >= 1000:
                raise ValueError("Can't find unique_id")

            _logs_dir = "%s_%03d" % (root, unique_id)
            unique_id += 1

        self.logs_dir = _logs_dir

    def save(self, name=None, obj=None):
        assert self.logs_dir is not None

        if (obj is None) != (name is None):
            raise ValueError("name and obj should be set to None or specified at the same time")

        if not os.path.isdir(self.logs_dir):
            os.makedirs(self.logs_dir)

        if name is not None:
            if name.endswith(".torch"):
                torch.save(obj, os.path.join(self.logs_dir, name))
            elif name.endswith(".pkl"):
                with open(os.path.join(self.logs_dir, name), "wb") as f:
                    pickle.dump(obj, f)
            else:
                ValueError("Unknown name extension: should be .torch or .pkl")
        else:
            torch.save(self, os.path.join(self.logs_dir, "model.torch"))

    @staticmethod
    def load_model(logs_dir):
        return torch.load(os.path.join(logs_dir, "model.torch"))

    def load(self, name):
        assert self.logs_dir is not None

        if name.endswith(".torch"):
            return torch.load(os.path.join(self.logs_dir, name))
        elif name.endswith(".pkl"):
            with open(os.path.join(self.logs_dir, name), "rb") as f:
                obj = pickle.load(f)
            return obj
        else:
            ValueError("Unknown name extension: should be .torch or .pkl")

    def dropout_to(self, values):
        dropout_layers = [_e for _e in self.modules() if isinstance(_e, nn.Dropout)]
        if not isinstance(values, (tuple, list)):
            values = [values] * len(dropout_layers)

        for _drop, _p in zip(dropout_layers, values):
            _drop.p = _p
