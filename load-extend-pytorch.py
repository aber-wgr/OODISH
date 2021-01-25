#!/usr/bin/env python
# coding: utf-8

# Collecting And Extending Autoencoder Networks in Pytorch
# ===
# 
# This is adapted from the workbook provided alongside the article "Implementing an Autoencoder in Pytorch" which can be found [here](https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1). The purpose of this workbook is to load the generated networks, read the parameters from the network, then extend the network to work with larger images.
# 

# ## Setup
# 
# We begin by importing our dependencies.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as  optim
import torchvision
import math

import collections

from model import SplitAutoencoder,ExtensibleEncoder,ExtensibleDecoder

from torch.utils.data.sampler import SubsetRandomSampler


# Set our seed and other configurations for reproducibility.

# In[ ]:


seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

platform = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 1, 'pin_memory': True} if platform=='cuda' else {}


if torch.cuda.is_available():
    platform = "cuda"
else:
    platform = "cpu"
#platform = "cpu"
print(platform)


# We set the batch size, the number of training epochs, and the learning rate. Batch size has to be reasonably low as we can't fit a huge number of these images into VRAM on my laptop.
# 
# Image size can be set here as I'm automatically resizing the images in my extraction code.

# In[ ]:

base_width = 256
base_height = 256

extended_width = 2048
extended_height = 2048

image_size = extended_width * extended_height

batch_size = 8
epochs = 20
learning_rate = 1e-4

code_sides = [16]

model_path = "../../Data/OPTIMAM_NEW/model0.pt"

#image_count = 500
image_count = -1

validation_split = 0.9


# ## Dataset
# 
# ImageFolder is now used to load the 2048x2048 versions of the images. This version of the DataLoader setup is designed to not batch or shuffle the images as we load them sequentially.

# In[ ]:


from torchvision.datasets import ImageFolder

from torchvision.transforms import ToTensor,Grayscale
transform = torchvision.transforms.Compose([
     torchvision.transforms.Grayscale(),
#     torchvision.transforms.Resize((height,width)),
     torchvision.transforms.ToTensor()
    ])

root_dir = "../../Data/OPTIMAM_NEW/png_images/casewise/ScreeningMammography/2048"
train_dataset = torchvision.datasets.ImageFolder(root=root_dir, transform=transform)
if (image_count == -1):
    train_dataset_subset = train_dataset
    image_count = len(train_dataset)
    print("setting image count to " + str(image_count))
else:
    train_dataset_subset = torch.utils.data.Subset(train_dataset, numpy.random.choice(len(train_dataset), image_count, replace=False))

dataset_len = len(train_dataset_subset)
indices = list(range(dataset_len))

# Randomly splitting indices:
val_len = int(np.floor((1.0 - validation_split) * dataset_len))

dataset_size = len(train_dataset_subset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if 1 :
    np.random.seed(1337)
    np.random.shuffle(indices)
train_indices, valid_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)
    
train_loader = torch.utils.data.DataLoader(
    train_dataset_subset, batch_size=batch_size, sampler = train_sampler
)

valid_loader = torch.utils.data.DataLoader(
    train_dataset_subset, batch_size=batch_size, sampler = valid_sampler
)

data_loaders = {"train": train_loader, "val": valid_loader}
data_lengths = {"train": split, "val": val_len}
print(split)
print(val_len)


# In[ ]:


#  use gpu if available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(platform)

# reload the saved model

code_size = code_sides[0] * code_sides[0]

# mean-squared error loss
criterion = nn.MSELoss()
#criterion = nn.BCELoss()


# Now load the saved 256x256 model to CPU-side (we'll load the extended version to GPU).

# In[ ]:


model = torch.load(model_path,map_location=torch.device("cuda"))
model.train()


# And adapt it to 2048x2048 mode.

# In[ ]:


model.reconstruct_to((extended_height,extended_width))
model.to(device)


# In[ ]:


print(model)


# Now run the reconstructed model for our selected number of epochs:

# In[ ]:


best_model_dicts = []
# populate with fake best models
for i in range(len(code_sides)):
    best_model_dicts.append((1.0,None))

models = [model]
optimizers = [optim.Adam(models[i].parameters(), lr=learning_rate)]
    
train_losses = []
val_losses = []
    
i=0
print("==================")
print("Running for code size:" + str(code_sides[i] * code_sides[i]))

train_losses.append([])
val_losses.append([])

for epoch in range(epochs):
    losses = {'train':0.0, 'val':0.0}

    for phase in ['train', 'val']:
        if phase == 'train':
            models[i].train()  # Set model to training mode
        else:
            models[i].eval()  # Set model to evaluate mode

        for batch_features, labels in data_loaders[phase]:
            # load it to the active device
            batch_features = batch_features.to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizers[i].zero_grad()

            # compute reconstructions
            codes = models[i].encoder(batch_features)
            outputs = models[i].decoder(codes)

            # compute training reconstruction loss
            local_loss = criterion(outputs,batch_features)

            if phase == 'train':
                # compute accumulated gradients
                local_loss.backward()

                # perform parameter update based on current gradients
                optimizers[i].step()

            # add the mini-batch training loss to epoch loss
            losses[phase] += local_loss.item()

    # compute the epoch training loss
    #losses['train'] = losses['train'] / data_lengths['train']
    #losses['val'] = losses['val'] / data_lengths['val']

    losses['train'] = losses['train'] / len(data_loaders['train'])
    losses['val'] = losses['val'] / len(data_loaders['val'])

    #check if best model
    if(losses['val'] < best_model_dicts[i][0]):
        best_model_dicts[i] = (losses['val'],models[i].state_dict())

    train_losses.append(losses['train'])
    val_losses.append(losses['val'])

    # display the epoch training loss
    print("epoch : {}/{}, train loss = {:.8f}, validation loss = {:.8f}".format(epoch + 1, epochs, losses['train'],losses['val']))
    


# In[ ]:




