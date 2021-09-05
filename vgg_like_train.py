import os
import sys
import random
import pickle

import numpy as np

import torch
from torch.utils.data import DataLoader
import torchmetrics
import torchvision
from torchvision import transforms

from torch_prune.models.var_dropout_vgglike import VGGLike
from torch_prune.trainers import train_loop, TBLogsWriter


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


seed_value = 0

random.seed(seed_value)
torch.manual_seed(seed_value)
np.random.seed(seed_value)

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = VGGLike(10, 1, use_dropout=True).to(device)

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

top_1 = torchmetrics.Accuracy(top_k=1).to(device)
top_1.__name__ = "top_1"

logs_writer = TBLogsWriter("./", writers_keys=list(loaders.keys()))
logs_writer.add_info({"model": "Molchanov's VGGLike(10, 1, use_dropout=True)",
                      "comment": "Reproduce training"})

logs = train_loop(model, ce_loss, loaders, opt, epochs, device=device, metrics=[ce_loss, top_1],
                  callbacks=[LRScheduler(opt), logs_writer])

with open("logs.pkl", "wb") as logs_file:
    pickle.dump(logs, logs_file)

torch.save(model, "model.torch")
