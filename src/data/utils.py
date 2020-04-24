from torch.utils.data import Dataset
from torchbearer.cv_utils import DatasetValidationSplitter
from typeguard import typechecked
from typing import Tuple


@typechecked
def split(dataset: Dataset,
        split_fraction: float,
        seed: int,
        return_test: bool = True) -> Tuple[Dataset, ...]:
    """
    Splits the dataset into training and validation (and testing if return_test is true)
    :param dataset: Dataset object
    :param split_fraction: Fraction of the whole dataset to be used for validation
    :param seed: Seed used for splitting
    :param return_test: if should split into three parts
    :return:
    """

    splitter = DatasetValidationSplitter(len(dataset), split_fraction, shuffle_seed=seed)

    trainset = splitter.get_train_dataset(dataset)
    valset = splitter.get_val_dataset(dataset)

    # Split the valset into test and validation
    if return_test:
        # Set split_fraction to low value such that testset
        valset, testset = split(valset, split_fraction=0.70, seed=seed, return_test=False)
        return trainset, valset, testset
    else:
        return trainset, valset