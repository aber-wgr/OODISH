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