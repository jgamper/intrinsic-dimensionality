from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torch

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Classifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Classifier, self).__init__()

        self.fc = nn.Linear(input_dim, n_classes)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.maxpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def get_resnet(encoder_name, num_classes, pretrained=True):
    """

    :param encoder_name:
    :param num_classes:
    :param pretrained:
    :return:
    """

    if encoder_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        latent_dim = 512
    else:
        model = models.resnet50(pretrained=pretrained)
        latent_dim = 2048

    children = (list(model.children())[:-2] + [Classifier(latent_dim, num_classes)])
    model = torch.nn.Sequential(*children)
    return model