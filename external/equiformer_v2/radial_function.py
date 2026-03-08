import torch
import torch.nn as nn


class RadialFunction(nn.Module):
    def __init__(self, channels_list):
        super().__init__()
        modules = []
        input_channels = channels_list[0]
        for i in range(1, len(channels_list)):
            modules.append(nn.Linear(input_channels, channels_list[i], bias=True))
            input_channels = channels_list[i]
            if i == len(channels_list) - 1:
                break
            modules.append(nn.LayerNorm(channels_list[i]))
            modules.append(torch.nn.SiLU())
        self.net = nn.Sequential(*modules)

    def forward(self, inputs):
        return self.net(inputs)
