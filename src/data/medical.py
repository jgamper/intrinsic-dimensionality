
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from typeguard import typechecked
from typing import Tuple
from src.data.utils import split

@typechecked
def get_loaders(root: str, batch_size: int, seed: int, stats=None) -> Tuple[DataLoader, DataLoader]:
    """
    Reads the dataset and splits into training and validation
    :param root_train:
    :param batch_size:
    :param seed:
    :param stats: Stats to normalise images
    :return:
    """
    if stats:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(**stats),
        ])

    dts = ImageFolder(root, transform=transform if stats else None)
    train, test = split(dts, split_fraction=0.3, return_test=False, seed=seed)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=1000, shuffle=False, pin_memory=True, num_workers=0)

    return train_loader, test_loader