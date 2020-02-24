import torch
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder

def get_loaders_mnist(root, use_cuda, batch_size=32,
                      means=((0.1307,), (0.3081,))):
    """

    :param mnist_root:
    :param use_cuda:
    :param batch_size:
    :return:
    """
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    mnist_transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(**means)
                       ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root, train=True, download=True, transform=mnist_transform),
        batch_size=batch_size, shuffle=True, **loader_kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root, train=False, transform=mnist_transform),
        batch_size=1000, shuffle=False, **loader_kwargs)

    return train_loader, test_loader

def get_loaders_cifar10(root, use_cuda, batch_size=32,
                        means=((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))):
    """

    :param mnist_root:
    :param use_cuda:
    :param batch_size:
    :return:
    """
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**means),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root, train=True, download=True, transform=transform_cifar10),
        batch_size=batch_size, shuffle=True, **loader_kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root, train=False, transform=transform_cifar10),
        batch_size=1000, shuffle=False, **loader_kwargs)

    return train_loader, test_loader

def get_loaders_cifar100(root, use_cuda, batch_size=32,
                         means=((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))):
    """

    :param mnist_root:
    :param use_cuda:
    :param batch_size:
    :return:
    """
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform_cifar100 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**means),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root, train=True, download=True, transform=transform_cifar100),
        batch_size=batch_size, shuffle=True, **loader_kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root, train=False, transform=transform_cifar100),
        batch_size=1000, shuffle=False, **loader_kwargs)

    return train_loader, test_loader

def get_loaders_custom(root, use_cuda, batch_size=32, means=None):
    """

    :param mnist_root:
    :param use_cuda:
    :param batch_size:
    :param means: normalising stats
    :return:
    """
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if means:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**means),
        ])

    train_loader = torch.utils.data.DataLoader(
        ImageFolder(root, transform=transform if means else None),
        batch_size=batch_size, shuffle=True, **loader_kwargs)

    test_loader = torch.utils.data.DataLoader(
        ImageFolder(root, transform=transform if means else None),
        batch_size=1000, shuffle=False, **loader_kwargs)

    return train_loader, test_loader

def get_loaders(root,  name, use_cuda, batch_size=32, means=None):
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
                 'CUSTOM': get_loaders_custom}

    assert any(name == dts for dts in list(available.keys())), 'Selected data should be '+' '.join(available)

    func = available[name]

    train_loader, test_loader = func(root, use_cuda, batch_size, means)

    return train_loader, test_loader