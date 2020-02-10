import torch
from torch import nn
import numpy as np
from intrinsic.utils.fastfood import fastfood_torched, fastfood_vars

class WrapFastfood(nn.Module):

    def __init__(self, module, intrinsic_dimension, device=0):
        """

        :param module:
        :param intrinsic_dimension:
        :param device:
        """
        super(WrapFastfood, self).__init__()

        # Hide this from inspection by get_parameters()
        self.m = [module]

        self.name_base_localname = []

        # Stores the initial value: \theta_{0}^{D}
        self.initial_value = dict()

        # Fastfood parameters
        self.fastfood_params = {}

        # Parameter vector that is updated
        # Initialised with zeros as per text: \theta^{d}
        V = nn.Parameter(torch.zeros((intrinsic_dimension)).to(device))
        self.register_parameter('V', V)
        v_size = (intrinsic_dimension, )

        # Iterate over layers in the module
        for name, param in module.named_parameters():
            # If param requires grad update
            if param.requires_grad:

                # Saves the initial values of the initialised parameters from param.data and sets them to no grad.
                # (initial values are the 'origin' of the search)
                self.initial_value[name] = v0 = param.clone().detach().requires_grad_(False).to(device)

                # Generate fastfood parameters
                DD = np.prod(v0.size())
                self.fastfood_params[name] = fastfood_vars(DD, device)

                base, localname = module, name
                while '.' in localname:
                    prefix, localname = localname.split('.', 1)
                    base = base.__getattr__(prefix)
                self.name_base_localname.append((name, base, localname))

        for name, base, localname in self.name_base_localname:
            delattr(base, localname)

    def forward(self, x):
        # Iterate over layers
        for name, base, localname in self.name_base_localname:

            init_shape = self.initial_value[name].size()
            DD = np.prod(init_shape)

            # Fastfood transform te replace dence P
            ray = fastfood_torched(self.V, DD, self.fastfood_params[name]).view(init_shape)

            param = self.initial_value[name] + ray

            setattr(base, localname, param)

        # Pass through the model, by getting hte module from a list self.m
        module = self.m[0]
        x = module(x)
        return x