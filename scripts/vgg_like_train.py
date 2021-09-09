import os
import random

import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchmetrics
# from torch_scripts.data.cifar10 import get_cifar10_zca
from torch_scripts.models.vgg import VGG
from torch_scripts.trainers import train_loop
from torch_scripts.callbacks import TBLogsWriter
from torch_scripts.utils import initialize_torch
from torch_scripts.storage import Storage

storage = Storage(os.environ["STORAGE"]).create_storage()

seed_value = 0
initialize_torch(seed_value)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = VGG('VGG16')
storage.save(model)
model.initialize_logs_dir(os.path.join(os.path.dirname(os.environ["STORAGE"]), "experiments"), "%d" % model.storage_id)

epochs = 200
train_limit = test_limit = None

# if os.environ.get("TEST_MODE", "FALSE").upper() == "TRUE":
#     epochs = 200
#     train_limit=30
#     test_limit=10
#
#trainset, testset = get_CIFAR10_ZCA(os.path.join(os.environ["DATA"], "cifar10"), train_transform=train_transform,
#                                    test_transform=test_transform, train_limit=train_limit, test_limit=test_limit)

train_transform = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
])

trainset = torchvision.datasets.CIFAR10(root=os.path.join(os.environ["DATA"], "cifar10"),
                                        train=True, transform=train_transform, download=True)
testset = torchvision.datasets.CIFAR10(root=os.path.join(os.environ["DATA"], "cifar10"),
                                       train=False, transform=test_transform)


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

opt = torch.optim.Adam(model.parameters())

ce_loss = torch.nn.CrossEntropyLoss()
ce_loss.__name__ = "cross_entropy"

top_1 = torchmetrics.Accuracy(top_k=1).to(device)
top_1.__name__ = "top_1"

model.l2_weight = 2.0e-5


def loss_l2(output, target):
    w = [_p for _n, _p in model.named_parameters() if _n.endswith(".weight")]
    reg = torch.stack([torch.square(_p).sum() for _p in w]).sum()
    return ce_loss(output, target) + model.l2_weight*reg


model.info = {
    "model": "VGG16",
    "data": "cifar10",
    "comment": "train, Adam + l2(2e-5), linear lr scheduler"
}

logs_writer = TBLogsWriter(model.logs_dir, writers_keys=list(loaders.keys()))
logs_writer.add_info(model.info)


class LRScheduler(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def on_epoch_begin(self, logs):
        _lr = 2.0e-5 - 1.0e-5 * min(logs["epoch"] / 100., 1.0)
        for _pg in self.optimizer.param_groups:
            _pg["lr"] = _lr
        logs["lr"] = _lr


logs = train_loop(model, loss_l2, loaders, opt, epochs, device=device, metrics=[loss_l2, ce_loss, top_1],
                  callbacks=[LRScheduler(opt), logs_writer])

model.save("logs.pkl", logs)
model.save()
