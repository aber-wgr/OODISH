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
        #self.conv_scale = kwargs["convolutions"] #conv at first level
        self.conv_scale = kwargs.pop('convolutions', 32)
        self.code_size = kwargs.pop('code_size', 100)
        self.dropout_chance = kwargs.pop('dropout_chance', 0.0)
        
        encoderPlanBase = OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=self.conv_scale, kernel_size=3, stride=1, padding=1)),
            ('dropout1', nn.Dropout2d(self.dropout_chance / 4,True)),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2)), # x/2 x y/2 x conv
            ('conv2', nn.Conv2d(in_channels=self.conv_scale, out_channels=self.conv_scale * 2, kernel_size=3, stride=1, padding=1)),
            ('dropout2', nn.Dropout2d(self.dropout_chance / 2,True)),
            ('relu2', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2)), # x/4 x y/4 x conv*2
            ('conv3', nn.Conv2d(in_channels=self.conv_scale  * 2, out_channels=self.conv_scale * 4, kernel_size=3, stride=1, padding=1)),
            ('dropout3', nn.Dropout2d(self.dropout_chance,True)),
            ('relu3', nn.ReLU()),
            ('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2)), # x/8 x y/8 x conv*3
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
        flattened_size = int((x/8)*(y/8)*(self.conv_scale * 4))        
        self.fc1 = nn.Linear(in_features=flattened_size,out_features=self.code_size)
                                     
    def forward(self, features):
        cnnOutput = self.cnnStage(features)
        code = self.fc1(cnnOutput)
        return code

class ExtensibleDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_shape = kwargs["input_shape"] #xy
        self.conv_scale = kwargs.pop('convolutions', 32)
        self.code_size = kwargs.pop('code_size', 100)
        self.dropout_chance = kwargs.pop('dropout_chance', 0.0)
        
        decoderPlanBase = OrderedDict([
            ('conv3', nn.Conv2d(in_channels=self.conv_scale * 4, out_channels=self.conv_scale * 2, kernel_size=3, stride=1, padding=1)),
            ('dropout3', nn.Dropout2d(self.dropout_chance,True)),
            ('relu3', nn.ReLU()),
            ('upsample3', nn.Upsample(scale_factor=2,mode='bilinear')),
            ('conv2', nn.Conv2d(in_channels=self.conv_scale*2, out_channels=self.conv_scale, kernel_size=3, stride=1, padding=1)),
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
        flattened_size = int((x/8)*(y/8)*(self.conv_scale * 4))
        
        self.fc1 = nn.Linear(in_features=self.code_size,out_features=flattened_size)
        self.unflatten = nn.Unflatten(1,(int(self.conv_scale*4),int(x/8),int(y/8)))
        self.sigmoid = nn.Sigmoid()
                                     
    def forward(self, features):
        fc_out = self.fc1(features)
        unflattened = self.unflatten(fc_out)
        sigmo = self.sigmoid(unflattened)
        out = self.cnnStage(sigmo)
        return out    
    
class SplitAutoencoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_shape = kwargs["input_shape"] #xy
        self.conv_scale = kwargs.pop('convolutions', 32)
        self.code_size = kwargs.pop('code_size', 100)
        self.dropout_chance = kwargs.pop('dropout_chance', 0.0)
        
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

class OldSplitAutoencoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.code_size = kwargs["code_size"]
        self.encoder = nn.Sequential( 
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 128x128x32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64x64x64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x32x128
            nn.Flatten(), # 131072x1
            nn.Linear(in_features=32*32*128,out_features=self.code_size),
            nn.ReLU()
        )
        # result (encoding) is code_size x 1
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.code_size, out_features=32*32*128), #131072x1
            nn.Unflatten(1,(128,32,32)), # 32x32x128
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'), # 64x64x64
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'), # 128x128x32
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='bilinear'), #256x256x1
            #nn.Sigmoid()
        )
        
    def forward(self, features):
        code = self.encoder(features)
        out = self.decoder(code)
        return out

#VAE version. Note that we cannot access the encodings here (as they're rather complex) so we don't bother with spliting the structure directly.
#This code is adapted from the VAE example for Pytorch, found at https://github.com/pytorch/examples/tree/master/vae

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.input_shape = kwargs["input_shape"] #xy
        self.code_size = kwargs.pop('code_size', 100)
        self.hidden_layer_size = kwargs.pop('hidden_layer', 400)
        #self.dropout_chance = kwargs.pop('dropout_chance', 0.0)
        hidden_layer_size = 100
        self.fc1 = nn.Linear(in_features=kwargs["input_shape"], out_features=hidden_layer_size)
        self.fc21 = nn.Linear(in_features=hidden_layer_size, out_features=self.code_size)
        self.fc22 = nn.Linear(in_features=hidden_layer_size, out_features=self.code_size)
        self.fc3 = nn.Linear(in_features=self.code_size, out_features=hidden_layer_size)
        self.fc4 = nn.Linear(in_features=hidden_layer_size, out_features=kwargs["input_shape"]
                             
        #self.dropout1 = nn.

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar