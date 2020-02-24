from src.utils.train import get_stats_for, parameter_count, get_exponential_range
from src.models.mnist import RegularCNNModel, FCNAsInPAper
from src.models.resnet import get_resnet
from src.data.classification import (get_loaders_cifar10,
                                    get_loaders_cifar100,
                                    get_loaders_mnist,
                                    get_loaders_custom,
                                     get_loaders)