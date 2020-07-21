* All contributions are welcome! Please raise an issue for a bug, feature or pull request!

* Give this repo a star! :star:

<p align="center">
    <img src="https://raw.githubusercontent.com/jgamper/intrinsic-dimensionality/master/assets/intrinsic_star.png" width="600"/>
<p>

# About

This package includes fastfood and dense transformation wrappers for pytorch modules, primarily to reproduce results from
[Li, Chunyuan, et al. "Measuring the intrinsic dimension of objective landscapes." arXiv preprint arXiv:1804.08838 (2018)](https://arxiv.org/abs/1804.08838) - see below for info.

# Install

`pip install intrinsic-dimensionality`

# Quick start on your classification task!

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_NUM
import torch
from torch import nn
import torchvision.models as models
from intrinsic import FastFoodWrap

class Classifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, n_classes)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.maxpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

def get_resnet(encoder_name, num_classes, pretrained=False):
    assert encoder_name in ["resnet18", "resnet50"], "{} is a wrong encoder name!".format(encoder_name)
    if encoder_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        latent_dim = 512
    else:
        model = models.resnet50(pretrained=pretrained)
        latent_dim = 2048
    children = (list(model.children())[:-2] + [Classifier(latent_dim, num_classes)])
    model = torch.nn.Sequential(*children)
    return model

# Get model and wrap it in fastfood
model = get_resnet("resnet18", num_classes=YOUR_NUMBER_OF_CLASSES).cuda()
model = FastFoodWrap(model, intrinsic_dimension=100, device=DEVICE_NUM)
```

# Reproducing experiments from the paper

Full thread about reproducibility results is available [here](https://twitter.com/brutforcimag/status/1240335205807816705?s=20).
Note that some hyper-parameters were not listed in the paper - I raised issues on Uber's Github repo [here](https://github.com/uber-research/intrinsic-dimension/issues/5).

I am able to reproduce their MNIST results with LR=0.0003, batch size 32 for both dense and fastfood transformations
using FCN (fcn-dense, fcn-fastfood). However, not for LeNet (cnn-dense, cnn-fastfood).

<p align="center">
    <img src="https://raw.githubusercontent.com/jgamper/intrinsic-dimensionality/master/assets/mnist_reproduction.png" width="600"/>
<p>

For CIFAR-10, with far larger resnet (Resnet-18 11mil param) vs 280k 20-layer resnet used in the paper,
results appear to be similar. FCN results in appendix (Fig S7) suggest some variation is to be expected.

<p align="center">
    <img src="https://raw.githubusercontent.com/jgamper/intrinsic-dimensionality/master/assets/cifar10.png" width="600"/>
<p>

# Cite

```
@misc{jgamper2020intrinsic,
  title   = "Intrinsic-dimensionality Pytorch",
  author  = "Gamper, Jevgenij",
  year    = "2020",
  url     = "https://github.com/jgamper/intrinsic-dimensionality"
}

@article{li2018measuring,
  title={Measuring the intrinsic dimension of objective landscapes},
  author={Li, Chunyuan and Farkhoor, Heerad and Liu, Rosanne and Yosinski, Jason},
  journal={arXiv preprint arXiv:1804.08838},
  year={2018}
}
```
