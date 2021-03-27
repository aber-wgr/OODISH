#basics taken from https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745/2
#clipped to rescale to stated min/max

import torch

class AddGaussianNoiseAndRescale(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noise = noise / torch.max(noise)
        noisy = tensor + noise
        
        noisy = noisy - torch.min(noisy)
        noisy = noisy / torch.max(noisy)
        return noisy
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
class Rescale(object):
    def __init__(self):
        mean = 0.5
        
    def __call__(self, tensor):
        tensor = tensor - torch.min(tensor)
        tensor = tensor / torch.max(tensor)
        return tensor