import sys

import numpy as np

from torch.utils.data import Dataset as TorchDataset
import torchvision


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


class NumpyDataset(TorchDataset):
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


def get_cifar10_zca(root, train_transform=None, test_transform=None, train_limit=None, test_limit=None):
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
