"""Data loading utilities for DnCNN denoiser pre-training.

Provides paired (noisy, ground-truth) sample transforms and a PyTorch
Dataset that applies them on-the-fly during training.
"""

import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset as _TorchDataset, DataLoader  # noqa: F401 — DataLoader re-exported
from torchvision.transforms import Compose, RandomApply, functional as TF


# ---------------------------------------------------------------------------
# Per-sample spatial augmentation transforms
# ---------------------------------------------------------------------------

class RandomHorizontalFlipTx:
    """Randomly flip both images in a sample horizontally.

    Args:
        p: Probability of applying the flip. Default is 0.5.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            sample: Dict with keys 'noisy' and 'gt', each a float tensor.

        Returns:
            Sample with both tensors horizontally flipped with probability p.
        """
        if random.random() < self.p:
            return {'noisy': TF.hflip(sample['noisy']), 'gt': TF.hflip(sample['gt'])}
        return sample


class RandomVerticalFlipTx:
    """Randomly flip both images in a sample vertically.

    Args:
        p: Probability of applying the flip. Default is 0.5.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            sample: Dict with keys 'noisy' and 'gt', each a float tensor.

        Returns:
            Sample with both tensors vertically flipped with probability p.
        """
        if random.random() < self.p:
            return {'noisy': TF.vflip(sample['noisy']), 'gt': TF.vflip(sample['gt'])}
        return sample


class ToTensor:
    """Convert numpy arrays in a sample dict to CUDA float tensors."""

    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """
        Args:
            sample: Dict with keys 'noisy' and 'gt', each a numpy array.

        Returns:
            Dict with both arrays converted to float32 CUDA tensors.
        """
        return {
            'noisy': torch.tensor(sample['noisy'], dtype=torch.float32, device='cuda'),
            'gt': torch.tensor(sample['gt'], dtype=torch.float32, device='cuda'),
        }


class RandomRotationTx:
    """Randomly rotate both images in a sample by one of the given angles.

    Args:
        angles: List of candidate rotation angles in degrees.
    """

    def __init__(self, angles: list[float]) -> None:
        self.angles = angles

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            sample: Dict with keys 'noisy' and 'gt', each a float tensor.

        Returns:
            Sample with both tensors rotated by a randomly chosen angle.
        """
        angle = float(np.random.choice(self.angles))
        return {
            'noisy': TF.rotate(sample['noisy'], angle),
            'gt': TF.rotate(sample['gt'], angle),
        }


# ---------------------------------------------------------------------------
# Paired dataset
# ---------------------------------------------------------------------------

class Dataset(_TorchDataset):
    """Paired (noisy, ground-truth) dataset with on-the-fly spatial augmentation.

    Each sample is a dict with keys 'noisy' and 'gt'. The transform pipeline
    converts arrays to CUDA float tensors and applies random horizontal flip,
    vertical flip, and rotation with the given probabilities.

    Args:
        data_train: Array of noisy training images, shape (N, H, W).
        data_gt: Array of ground-truth images, shape (N, H, W).
        rot_p: Probability of applying each spatial augmentation. Default 0.5.
        rot_angles: Candidate rotation angles in degrees.
            Defaults to [0, 90, 180, 270].
    """

    def __init__(
        self,
        data_train: np.ndarray,
        data_gt: np.ndarray,
        rot_p: float = 0.5,
        rot_angles: Optional[list[float]] = None,
    ) -> None:
        if rot_angles is None:
            rot_angles = [0, 90, 180, 270]
        self.data_train = data_train
        self.data_gt = data_gt
        self.transform = Compose([
            ToTensor(),
            RandomApply([
                RandomHorizontalFlipTx(p=rot_p),
                RandomVerticalFlipTx(p=rot_p),
                RandomRotationTx(angles=rot_angles),
            ], p=1.0),
        ])

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.data_train.shape[0]

    def __getitem__(self, idx: int | list[int]) -> dict[str, torch.Tensor]:
        """Return the sample at index idx after applying the transform pipeline.

        Args:
            idx: Integer index or list of indices.

        Returns:
            Dict with keys 'noisy' and 'gt', each a float32 CUDA tensor of
            shape (1, H, W).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {
            'noisy': self.data_train[idx][None, ...],
            'gt': self.data_gt[idx][None, ...],
        }
        return self.transform(sample)
