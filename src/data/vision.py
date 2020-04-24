import os
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from typeguard import typechecked
from typing import Tuple


@typechecked
def get_loaders_mnist(root: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Returns mnist dataloaders
    :param mnist_root:
    :param batch_size:
    :return:
    """

    mnist_transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

    train_loader = DataLoader(
        datasets.MNIST(root, train=True, download=True, transform=mnist_transform),
        batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(
        datasets.MNIST(root, train=False, transform=mnist_transform),
        batch_size=1000, shuffle=False)

    return train_loader, test_loader

@typechecked
def get_loaders_cifar10(root: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Returns cifar10 dataloaders
    :param mnist_root:
    :param use_cuda:
    :param batch_size:
    :return:
    """
    stats = {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)}

    transform_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**stats),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root, train=True, download=True, transform=transform_cifar10),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root, train=False, transform=transform_cifar10),
        batch_size=1000, shuffle=False)

    return train_loader, test_loader

@typechecked
def get_loaders_cifar100(root: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Returns cifar100 dataloaders
    :param mnist_root:
    :param batch_size:
    :return:
    """
    stats = {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)}

    transform_cifar100 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**stats),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root, train=True, download=True, transform=transform_cifar100),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root, train=False, transform=transform_cifar100),
        batch_size=1000, shuffle=False)

    return train_loader, test_loader