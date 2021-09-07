import os
import random

import numpy as np

import torch
from torch import nn
import torch.backends.cudnn


def initialize_torch(seed_value, deterministic=True):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)

    if deterministic:
        # os.environ["CUBLAS"]
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


class Modifiers(object):
    def __init__(self):
        self.__dict__["elems"] = dict()

    def __setattr__(self, name, value):
        if name in self.elems:
            raise KeyError()
        self.elems[name] = value

    def __getattr__(self, name):
        return self.elems[name]

    def __repr__(self):
        return str(self.elems)


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()

        self.modifiers = Modifiers()
        self.logs_dir = None

    def dropout_to(self, values):
        dropout_layers = [_e for _e in self.modules() if isinstance(_e, nn.Dropout)]
        if not isinstance(values, (tuple, list)):
            values = [values] * len(dropout_layers)

        for _drop, _p in zip(dropout_layers, values):
            _drop.p = _p
