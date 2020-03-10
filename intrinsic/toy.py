from typing import Union, Tuple
import torch
from torch import nn
from intrinsic.dense import WrapDense
from intrinsic.fastfood import WrapFastfood
import numpy as np
from tqdm import tqdm


class ToyProblem(nn.Module):

    def __init__(self):
        """
        Toy model for intrinsic dimensionality demonstration
        https://arxiv.org/abs/1804.08838 in Section 2
        """
        super(ToyProblem, self).__init__()
        self.param = nn.Parameter(torch.zeros(1000,))

    def forward(self, x):
        """
        X is just a dummy variable here
        :param x:
        :return: Will return loss value
        """
        loss = 0
        for lbl, j in enumerate(range(100, 1000 + 1, 100)):
            i = j - 100
            subset = self.param[i:j - 1]
            param_sum = torch.sum(subset)
            loss += (param_sum - (lbl + 1)) ** 2
        return loss


def optimize_toy(wrapper: Union[WrapFastfood, WrapDense],
                             intrinsic_dimension: int=10, num_iter: int=10000, lr: float=3e-4) -> Tuple[list, nn.Module]:
    """
    Will optimize ToyProblem model
    :param wrapper: Either WrapFastfood or WrapDense
    :param intrinsic_dimension:
    :param num_iter:
    :param lr:
    :return:
    """
    losses = []
    epoch_no_improvement = 0

    # Initialise toy model for the toy problem and wrap using one of the wrappers
    model = wrapper(ToyProblem().cuda(), intrinsic_dimension=intrinsic_dimension)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    iterator = tqdm(range(num_iter), desc='Loss ', leave=True)
    for i in iterator:

        model.train()
        optimizer.zero_grad()
        loss = model.forward(1)
        loss.backward()
        optimizer.step()

        iterator.set_description("Loss {}".format(loss.item()))
        iterator.refresh()

        losses.append(loss.item())

        if i == 0:
            prev_loss = loss.item()
        if np.isclose(loss.item(), 0.1):
            iterator.close()
            break
        if prev_loss <= loss.item():
            epoch_no_improvement += 1
        if epoch_no_improvement > 1000:
            iterator.close()
            break

        prev_loss = loss.item()
    return losses, model

def evaluate(model: Union[WrapFastfood, WrapDense]) -> Tuple[list, float]:
    """
    Evaluates toy problem
    :param params:
    :return:
    """
    vals = []
    total = 0
    for lbl, j in enumerate(range(100, 1000 + 1, 100)):
        i = j - 100
        with torch.no_grad():
            val = model.m[0].param[i:j - 1].sum().item()
            total += val
            vals.append(val)

    return np.array(vals), total

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    target_sum = np.sum([i for i in range(11)])

    dims = [1, 3, 5, 7, 8, 9, 10, 11, 12, 15, 17, 20]

    performances = []

    for int_dim in dims:

        losses, model = optimize_toy(WrapDense,
                                     intrinsic_dimension=int_dim,
                                     num_iter=100000, lr=0.001)

        vals, total = evaluate(model)

        performances.append(total)