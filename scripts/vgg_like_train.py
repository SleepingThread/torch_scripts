# Good configurations:
# 11: 93.12, crop+flip, Adam + l2(2e-5) + drop(0.2) + cosine lr
# 9: 92.61, crop+flip, Adam + l2(1e-4) + drop(0.2) + cosine lr
# 2: 93.18, crop+flip, SGD + l2(2.5e-4) + cosine lr

import os
import random

import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchmetrics
from torch_scripts.models.vgg import VGG
from torch_scripts.trainers import train_loop
from torch_scripts.callbacks import TBLogsWriter, LRSchedulerCosine
from torch_scripts.utils import initialize_torch
from torch_scripts.storage import Storage

storage = Storage(os.environ["STORAGE"]).create_storage()

seed_value = 0
initialize_torch(seed_value)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = VGG('VGG16')
model.dropout_to(0.2)
storage.save(model)
model.initialize_logs_dir(os.path.join(os.path.dirname(os.environ["STORAGE"]), "experiments"),
                          "%03d" % model.storage_id)
print("Logs dir: ", model.logs_dir)

epochs = 200

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
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
loaders = {"train": DataLoader(trainset, batch_size=100, generator=train_g, worker_init_fn=seed_worker),
           "test": DataLoader(testset, batch_size=512, shuffle=False)}

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
    "comment": "train, Adam + l2(2e-5) + drop(0.2), cosine lr scheduler",
    "status": "running"
}

logs_writer = TBLogsWriter(model.logs_dir, writers_keys=list(loaders.keys()))
logs_writer.add_info(model.info)

logs = train_loop(model, loss_l2, loaders, opt, epochs, device=device, metrics=[loss_l2, ce_loss, top_1],
                  callbacks=[LRSchedulerCosine(opt), logs_writer])

model.info["status"] = "finished"
model.info["quality"] = logs[-1]["test"]["top_1"]
model.save("logs.pkl", logs)
model.save()
storage.save(model)
