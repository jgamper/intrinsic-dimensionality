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

def get_resnet_mean_var(encoder_name, num_classes):
    """
    Returns model that is intialised according to mean and var of Imagenet weights
    Mean Var Init from:
    @article{raghu2019transfusion,
      title={Transfusion: Understanding transfer learning with applications to medical imaging},
      author={Raghu, Maithra and Zhang, Chiyuan and Kleinberg, Jon and Bengio, Samy},
      journal={arXiv preprint arXiv:1902.07208},
      year={2019}
    }
    :param encoder_name:
    :param num_classes:
    :return:
    """
    if encoder_name == "resnet18":
        imagenet = models.resnet18(pretrained=True)
        latent_dim = 512
    else:
        imagenet = models.resnet50(pretrained=True)
        latent_dim = 2048

    mean_var = get_resnet(encoder_name, num_classes, pretrained=False)

    mean_var_dict = mean_var.state_dict()
    mean_var_keys = list(mean_var.state_dict().keys())
    
    with torch.no_grad():
        for i, (key, param) in enumerate(imagenet.state_dict().items()):
            if "num_batches_tracked" not in key:
                mean, std = torch.mean(param), torch.std(param)
                mean_var_key = mean_var_keys[i]
                param_mean_var = mean_var_dict[mean_var_key]
                mean_var_dict[mean_var_key] = torch.nn.init.normal_(torch.empty_like(param_mean_var),
                                                                    mean=mean, std=std)
    mean_var.load_state_dict(mean_var_dict)

    return mean_var