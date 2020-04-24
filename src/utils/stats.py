"""Usage:
          stats.py --root <path>

Computes mean and standard deviation of the dataset per channel
Options:
    --root   path to the dataset
"""
from docopt import docopt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    ])

def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for data, y in tqdm(loader):

        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

def compute_stats(root):
    """
    Computes dataset statistics
    :param root; path to the dataset
    """
    dataset = datasets.ImageFolder(root,
                     transform=transform)
    loader = DataLoader(dataset,
                             batch_size=10,
                             num_workers=0,
                             shuffle=False)

    mean, std = online_mean_and_sd(loader)
    print(mean)
    print('\n')
    print(std)

if __name__ == "__main__":

    arguments = docopt(__doc__)
    root = arguments['<path>']
    compute_stats(root)