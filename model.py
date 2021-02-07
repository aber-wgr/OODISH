import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import collections
from collections import OrderedDict
import torchvision
import math

class ExtensibleEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_shape = kwargs["input_shape"] #xy
        self.conv_scale = kwargs["convolutions"] #conv at first level
        self.code_size = kwargs["code_size"]
        self.dropout_chance = kwargs["dropout_chance"]
        
        encoderPlanBase = OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=self.conv_scale, kernel_size=3, stride=1, padding=1)),
            ('dropout1', nn.Dropout2d(self.dropout_chance / 4,True)),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2)), # x/2 x y/2 x conv
            ('conv2', nn.Conv2d(in_channels=self.conv_scale, out_channels=self.conv_scale**2, kernel_size=3, stride=1, padding=1)),
            ('dropout2', nn.Dropout2d(self.dropout_chance / 2,True)),
            ('relu2', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2)), # x/4 x y/4 x conv^2
            ('conv3', nn.Conv2d(in_channels=self.conv_scale**2, out_channels=self.conv_scale**3, kernel_size=3, stride=1, padding=1)),
            ('dropout3', nn.Dropout2d(self.dropout_chance,True)),
            ('relu3', nn.ReLU()),
            ('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2)), # x/8 x y/8 x conv^3
            ('flatten', nn.Flatten()) #x/8*y/8*conv^3 
        ])
        
        self.cnnStage = nn.Sequential(encoderPlanBase)        
        self.rebuild_fc_layers(self.input_shape)
    
    def reconstruct_to(self, scale):
        for p in self.cnnStage.parameters():
            p.requires_grad = False
        self.input_shape = scale
        self.rebuild_fc_layers(scale)
    
    def rebuild_fc_layers(self, scale): 
        x = scale[1]
        y = scale[0]
        flattened_size = int((x/8)*(y/8)*(self.conv_scale**3))        
        self.fc1 = nn.Linear(in_features=flattened_size,out_features=self.code_size)
                                     
    def forward(self, features):
        cnnOutput = self.cnnStage(features)
        code = self.fc1(cnnOutput)
        return code

class ExtensibleDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_shape = kwargs["input_shape"] #xy
        self.conv_scale = kwargs["convolutions"] #conv at first level
        self.code_size = kwargs["code_size"]
        self.dropout_chance = kwargs["dropout_chance"]
        
        decoderPlanBase = OrderedDict([
            ('conv3', nn.Conv2d(in_channels=self.conv_scale**3, out_channels=self.conv_scale**2, kernel_size=3, stride=1, padding=1)),
            ('dropout3', nn.Dropout2d(self.dropout_chance,True)),
            ('relu3', nn.ReLU()),
            ('upsample3', nn.Upsample(scale_factor=2,mode='bilinear')),
            ('conv2', nn.Conv2d(in_channels=self.conv_scale**2, out_channels=self.conv_scale, kernel_size=3, stride=1, padding=1)),
            ('dropout2', nn.Dropout2d(self.dropout_chance / 2,True)),
            ('relu2', nn.ReLU()),
            ('upsample2', nn.Upsample(scale_factor=2,mode='bilinear')),
            ('conv1', nn.Conv2d(in_channels=self.conv_scale, out_channels=1, kernel_size=3, stride=1, padding=1)),
            ('dropout1', nn.Dropout2d(self.dropout_chance / 4,True)),
            ('relu1', nn.ReLU()),
            ('upsample1', nn.Upsample(scale_factor=2,mode='bilinear'))
        ])
        
        self.cnnStage = nn.Sequential(decoderPlanBase)        
        self.rebuild_fc_layers(self.input_shape)
    
    def reconstruct_to(self, scale):           
        for p in self.cnnStage.parameters():
            p.requires_grad = False
        self.input_stage = scale    
        self.rebuild_fc_layers(scale)
    
    def rebuild_fc_layers(self, scale):
        x = scale[1]
        y = scale[0]
        flattened_size = int((x/8)*(y/8)*(self.conv_scale**3))
        
        self.fc1 = nn.Linear(in_features=self.code_size,out_features=flattened_size)
        self.unflatten = nn.Unflatten(1,(int(self.conv_scale**3),int(x/8),int(y/8)))
                                     
    def forward(self, features):
        fc_out = self.fc1(features)
        unflattened = self.unflatten(fc_out)
        out = self.cnnStage(unflattened)
        return out    
    
class SplitAutoencoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_shape = kwargs["input_shape"] #xy
        self.conv_scale = kwargs["convolutions"] #conv at first level
        self.code_size = kwargs["code_size"]
        self.dropout_chance = kwargs["dropout_chance"]
        
        self.encoder = ExtensibleEncoder(input_shape=self.input_shape,code_size=self.code_size,convolutions=self.conv_scale, dropout_chance=self.dropout_chance)
        self.decoder = ExtensibleDecoder(input_shape=self.input_shape,code_size=self.code_size,convolutions=self.conv_scale, dropout_chance=self.dropout_chance)
    
    def reconstruct_to(self, scale):
        self.input_shape = scale
        self.encoder.reconstruct_to(scale)
        self.decoder.reconstruct_to(scale)
                                
    def forward(self, features):
        code = self.encoder(features)
        out = self.decoder(code)
        return out

