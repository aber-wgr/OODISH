import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import math


class SplitAutoencoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        input_size = kwargs["input_shape"] #xy
        x = input_size[1]
        y = input_size[0]
        conv_scale = kwargs["convolutions"] #conv at first level
        flattened_size = int((x/8)*(y/8)*(conv_scale*4))
        self.encoder = nn.Sequential( 
            nn.Conv2d(in_channels=1, out_channels=conv_scale, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # x/2 x y/2 x conv
            nn.Conv2d(in_channels=conv_scale, out_channels=conv_scale*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # x/4 x y/4 x conv*2
            nn.Conv2d(in_channels=conv_scale*2, out_channels=conv_scale*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # x/8 x y/8 x conv*4
            nn.Flatten(), # x/8*y/8*conv*4
            nn.Linear(in_features=flattened_size,out_features=kwargs["code_size"]),
            nn.ReLU()
        )
        # result (encoding) is code_size x 1
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features=kwargs["code_size"], out_features=flattened_size), 
            nn.Unflatten(1,(int(conv_scale*4),int(x/8),int(y/8))),
            nn.Conv2d(in_channels=conv_scale*4, out_channels=conv_scale*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'), 
            nn.Conv2d(in_channels=conv_scale*2, out_channels=conv_scale, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=conv_scale, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='bilinear')
            #nn.Sigmoid()
        )
        
    def forward(self, features):
        code = self.encoder(features)
        out = self.decoder(code)
        return out

