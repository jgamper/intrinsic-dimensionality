<p align="center">
    <a href="https://github.com/jgamper/intrinsic-dimensionality/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/jgamper/intrinsic-dimensionality.svg?color=blue">
    </a>
    <a href="https://github.com/jgamper/intrinsic-dimensionality/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/v/release/jgamper/intrinsic-dimensionality?include_prereleases">
    </a>
</p>

* All contributions are welcome! Please raise an issue for a bug, feature or pull request!

* <a href="https://twitter.com/share?" class="twitter-share-button" data-text="Check this out!" data-url="https://github.com/jgamper/intrinsic-dimensionality" data-show-count="false">Tweet</a> about this repo!

* Give this repo a star! :star:

<p align="center">
    <img src="https://raw.githubusercontent.com/jgamper/intrinsic-dimensionality/master/docs/source/imgs/star_syntax.png?token=ADDZO4PH6CJSK5XTSC2ZLXK6ZPXRY" width="600"/>
<p>

# Install

`pip install compay-syntax==0.4.0`

# Quick start on your classification task!

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_NUM
import torch
import torchvision.models as models
from intrinsic import FastFoodWrap

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
<p align="center">
    <img src="https://raw.githubusercontent.com/jgamper/intrinsic-dimensionality/refactor/assets/mnist_reproduction.png?token=ADDZO4IHL3PNENJD6CP25GC7ECUBW" width="600"/>
<p>