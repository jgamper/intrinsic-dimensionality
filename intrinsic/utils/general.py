import torch
from torchvision import transforms
from torchvision import datasets

def get_loaders(mnist_root, use_cuda, batch_size=32):
    """

    :param mnist_root:
    :param use_cuda:
    :param batch_size:
    :return:
    """
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    mnist_transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(mnist_root, train=True, download=True, transform=mnist_transform),
        batch_size=batch_size, shuffle=True, **loader_kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(mnist_root, train=False, transform=mnist_transform),
        batch_size=1000, shuffle=False, **loader_kwargs)

    return train_loader, test_loader

def parameter_count(model):
    """
    Counts the number of model parameters that require grad update
    :param model:
    :return:
    """
    param_tot=0
    for name, param in model.named_parameters():
        if param.requires_grad:
            #print(name, param.data.size(), v_size)
            param_size = 1
            for d in list(param.data.size()):
                param_size *= d
            param_tot += param_size
    return param_tot