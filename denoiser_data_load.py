import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import Compose, RandomRotation, InterpolationMode, RandomApply
from tqdm import tqdm
from tqdm import trange

from torchvision.transforms import Compose, RandomRotation, InterpolationMode, RandomApply, RandomHorizontalFlip, RandomVerticalFlip, functional

class RandomHorizontalFlipTx(object):
    """Random horizontal flip with probability p."""
    def __init__(self, p):
        self.p = p
    
    def __call__(self, sample):
        noisy, gt = sample['noisy'], sample['gt']
        # Random horizontal flipping
        if random.random() < self.p:
            noisy = functional.hflip(noisy)
            gt = functional.hflip(gt)
        return {'noisy': noisy, 'gt': gt}


class RandomVerticalFlipTx(object):
    """Random vertical flip with probability p."""
    def __init__(self, p):
        self.p = p
        
    def __call__(self, sample):
        noisy, gt = sample['noisy'], sample['gt']
        # Random horizontal flipping
        if random.random() < self.p:
            noisy = functional.vflip(noisy)
            gt = functional.vflip(gt)
        return {'noisy': noisy, 'gt': gt}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        noisy, gt = sample['noisy'], sample['gt']
        return {'noisy': torch.FloatTensor(noisy).cuda(), 'gt': torch.FloatTensor(gt).cuda()}


class RandomRotationTx(object):
    """Random rotation transform with provided list of angles."""
    def __init__(self, angles):
        self.angles = angles
        
    def __call__(self, sample):
        angle = float(np.random.choice(self.angles, 1))
        noisy, gt = sample['noisy'], sample['gt']
        return {'noisy': functional.rotate(noisy, angle), 'gt': functional.rotate(gt, angle)}


def gauss_noise_tensor(sample):
    img = sample['noisy']
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)
    sigma = np.random.uniform(low=std_low, high=std_high)
    
    out = img + sigma * torch.randn_like(img)
    if out.dtype != dtype:
        out = out.to(dtype)
        
    return {'noisy': img, 'gt': sample['gt']}


class Dataset(Dataset):
    """Train dataset."""
    def __init__(self, data_train, data_gt, rot_p=0.5, rot_angles=[0,90,180,270]):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_train = data_train
        self.data_gt = data_gt
        
        self.transform = Compose([
        ToTensor(), RandomApply([RandomHorizontalFlipTx(p=rot_p), RandomVerticalFlipTx(p=rot_p),
                                 RandomRotationTx(angles=rot_angles)], p=1.0)])

    def __len__(self):
        return self.data_train.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'noisy': self.data_train[idx][None, ...],
                  'gt': self.data_gt[idx][None, ...]}

        if self.transform:
            sample = self.transform(sample)

        return sample