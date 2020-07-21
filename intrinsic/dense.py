import torch
from torch import nn


class DenseWrap(nn.Module):
    def __init__(self, module, intrinsic_dimension, device=0):
        """
        Wrapper to estimate the intrinsic dimensionality of the
        objective landscape for a specific task given a specific model
        :param module: pytorch nn.Module
        :param intrinsic_dimension: dimensionality within which we search for solution
        :param device: cuda device id
        """
        super(DenseWrap, self).__init__()

        # Hide this from inspection by get_parameters()
        self.m = [module]

        self.name_base_localname = []

        # Stores the initial value: \theta_{0}^{D}
        self.initial_value = dict()

        # Stores the randomly generated projection matrix P
        self.random_matrix = dict()

        # Parameter vector that is updated, initialised with zeros as per text: \theta^{d}
        V = nn.Parameter(torch.zeros((intrinsic_dimension, 1)).to(device))
        self.register_parameter("V", V)
        v_size = (intrinsic_dimension,)

        # Iterates over layers in the Neural Network
        for name, param in module.named_parameters():
            # If the parameter requires gradient update
            if param.requires_grad:

                # Saves the initial values of the initialised parameters from param.data and sets them to no grad.
                # (initial values are the 'origin' of the search)
                self.initial_value[name] = v0 = (
                    param.clone().detach().requires_grad_(False).to(device)
                )

                # If v0.size() is [4, 3], then below operation makes it [4, 3, v_size]
                matrix_size = v0.size() + v_size

                # Generates random projection matrices P, sets them to no grad
                self.random_matrix[name] = (
                    torch.randn(matrix_size, requires_grad=False).to(device)
                    / intrinsic_dimension ** 0.5
                )

                # NOTE!: lines below are not clear!
                base, localname = module, name
                while "." in localname:
                    prefix, localname = localname.split(".", 1)
                    base = base.__getattr__(prefix)
                self.name_base_localname.append((name, base, localname))

        for name, base, localname in self.name_base_localname:
            delattr(base, localname)

    def forward(self, x):
        # Iterate over the layers
        for name, base, localname in self.name_base_localname:

            # Product between matrix P and \theta^{d}
            ray = torch.matmul(self.random_matrix[name], self.V)

            # Add the \theta_{0}^{D} to P \dot \theta^{d}
            param = self.initial_value[name] + torch.squeeze(ray, -1)

            setattr(base, localname, param)

        # Pass through the model, by getting the module from a list self.m
        module = self.m[0]
        x = module(x)
        return x
