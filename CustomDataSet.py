import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import collections
from collections import OrderedDict
import torchvision
import math
from torch.utils.data import Dataset
from natsort import natsorted
from PIL import Image
import os
from torchvision.transforms import ToTensor,Grayscale

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("F")
        tensor_image = self.transform(image)
        return tensor_image
    
class CustomDataSetWithError(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)
        self.criterion = nn.MSELoss()
        
        self.plain_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.0,65535.0)
        ])

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("F")
        base_image = self.plain_transform(image)
        transformed_image = self.transform(image)
        error = self.criterion(transformed_image,base_image)
        return transformed_image,error