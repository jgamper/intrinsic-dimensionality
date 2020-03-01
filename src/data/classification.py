import os
import torch
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder

def get_loaders_mnist(root, use_cuda, batch_size, stats):
    """

    :param mnist_root:
    :param use_cuda:
    :param batch_size:
    :return:
    """
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    mnist_transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(**stats)
                       ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root, train=True, download=True, transform=mnist_transform),
        batch_size=batch_size, shuffle=True, **loader_kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root, train=False, transform=mnist_transform),
        batch_size=1000, shuffle=False, **loader_kwargs)

    return train_loader, test_loader

def get_loaders_cifar10(root, use_cuda, batch_size, stats):
    """

    :param mnist_root:
    :param use_cuda:
    :param batch_size:
    :return:
    """
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**stats),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root, train=True, download=True, transform=transform_cifar10),
        batch_size=batch_size, shuffle=True, **loader_kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root, train=False, transform=transform_cifar10),
        batch_size=1000, shuffle=False, **loader_kwargs)

    return train_loader, test_loader

def get_loaders_cifar100(root, use_cuda, batch_size, stats):
    """

    :param mnist_root:
    :param use_cuda:
    :param batch_size:
    :return:
    """
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform_cifar100 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**stats),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root, train=True, download=True, transform=transform_cifar100),
        batch_size=batch_size, shuffle=True, **loader_kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root, train=False, transform=transform_cifar100),
        batch_size=1000, shuffle=False, **loader_kwargs)

    return train_loader, test_loader

def get_loaders_custom(root_train, root_valid, use_cuda, batch_size=32, stats=None):
    """

    :param root_train:
    :param root_val:
    :param use_cuda:
    :param batch_size:
    :param means: normalising stats
    :return:
    """
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if stats:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(**stats),
        ])

    train_loader = torch.utils.data.DataLoader(
        ImageFolder(root_train, transform=transform if stats else None),
        batch_size=batch_size, shuffle=True, **loader_kwargs)

    test_loader = torch.utils.data.DataLoader(
        ImageFolder(root_valid, transform=transform if stats else None),
        batch_size=1000, shuffle=False, **loader_kwargs)

    return train_loader, test_loader

def get_loaders(root,  name, use_cuda, batch_size=32, stats=None):
    """

    :param root:
    :param use_cuda:
    :param batch_size:
    :param means:
    :param name:
    :return:
    """
    available = {'MNIST': get_loaders_mnist,
                 'CIFAR10': get_loaders_cifar10,
                'CIFAR100': get_loaders_cifar100,
                 'BREAST': get_loaders_custom,
                 'COLON': get_loaders_custom,
                 'LYMPHOMA': get_loaders_custom}

    assert any(name == dts for dts in list(available.keys())), 'Selected data should be '+' '.join(available)

    func = available[name]

    if name in ['MNIST', 'CIFAR10', 'CIFAR100']:

        train_loader, test_loader = func(root, use_cuda, batch_size, stats)

    else:
        print(name)
        root_train = os.path.join(root, 'train')
        root_valid= os.path.join(root, 'test')
        train_loader, test_loader = func(root_train, root_valid, use_cuda, batch_size, stats)

    return train_loader, test_loader