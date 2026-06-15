"""Miscellaneous utilities for RSR-NF.

Covers: coordinate grid generation, object loading, image-quality metrics,
model utilities, and TV regularization losses.
"""

import numpy as np
import torch

from skimage.data import shepp_logan_phantom
from skimage.transform import rescale


# ---------------------------------------------------------------------------
# Coordinate grid helpers
# ---------------------------------------------------------------------------

def _create_yx_grid(grid_size: tuple[int, int]) -> np.ndarray:
    """Create a 2-D mesh grid of normalised pixel coordinates (matrix indexing).

    Args:
        grid_size: (H, W) spatial dimensions of the grid.

    Returns:
        Float32 array of shape (H, W, 2), where the last axis holds
        (row, col) coordinates normalised to [0, 1).
    """
    h, w = grid_size
    coords_i = np.linspace(0, 1, h, endpoint=False)
    coords_j = np.linspace(0, 1, w, endpoint=False)
    return np.stack(
        np.meshgrid(coords_i, coords_j, indexing='ij'), axis=-1
    ).astype('float32')


def _create_yxt_grid(grid_size: tuple[int, int, int]) -> np.ndarray:
    """Create a 3-D mesh grid of normalised spatiotemporal coordinates (matrix indexing).

    Args:
        grid_size: (H, W, P) dimensions of the grid.

    Returns:
        Float32 array of shape (H, W, P, 3), where the last axis holds
        (row, col, time) coordinates normalised to [0, 1).
    """
    h, w, P = grid_size
    coords_i = np.linspace(0, 1, h, endpoint=False)
    coords_j = np.linspace(0, 1, w, endpoint=False)
    coords_t = np.linspace(0, 1, P, endpoint=False)
    return np.stack(
        np.meshgrid(coords_i, coords_j, coords_t, indexing='ij'), axis=-1
    ).astype('float32')


# ---------------------------------------------------------------------------
# Object loading
# ---------------------------------------------------------------------------

def load_shepp_logan(size: int = 256) -> np.ndarray:
    """Load and rescale the Shepp-Logan phantom to a given spatial resolution.

    Args:
        size: Target side length in pixels. The phantom is originally 400x400
            and is rescaled by size/400.

    Returns:
        Phantom array of shape (size, size, 1).
    """
    return rescale(
        shepp_logan_phantom(),
        scale=size / 400,
        mode='reflect',
        channel_axis=None,
    )[..., None]


def load_f(obj_type: str, motion: str, spatial_dim: int, P: int) -> np.ndarray:
    """Load the ground-truth dynamic object from disk.

    File paths are hardcoded per object type. For object types that only
    exist at a minimum of 32 frames, sub-sampling is applied when P < 32.

    Args:
        obj_type: Dataset identifier. Supported values:
            'walnut', 'hydro', 'cardiac*', 'material',
            'LLNL', 'LLNL_S03', 'LLNL_S12',
            'pde*' (P must be 256 or 128),
            'polymer_binary', 'polymer_binary_subint',
            'polymer_binary_subint_hardest',
            'polymer_subint', 'polymer_subint_hardest'.
        motion: Motion type string; used in the filename for walnut/hydro/cardiac/pde.
        spatial_dim: Spatial side length (used in the filename for walnut/hydro/cardiac/pde).
        P: Number of time frames to load.

    Returns:
        Object array f, shape (H, W, P).

    Raises:
        ValueError: If obj_type is not recognised or P is unsupported for 'pde*'.
    """
    base = '/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/obj_domain_psm/data/true_objects'
    nitin = '/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/nitin_files/phantoms'

    if obj_type in ['walnut', 'hydro'] or 'cardiac' in obj_type:
        p_load = max(P, 32)
        f = np.load(
            f'{base}/{obj_type}/f_{obj_type}_{motion}_spatial_dim_{spatial_dim}_P_{p_load}.npy')
        if P < 32:
            f = f[..., ::32 // P]

    elif obj_type == 'material':
        f = np.load(f'{base}/material/{P}/f_materials.npy').transpose(1, 2, 0)[:, :, :P] / 255
        f[f < 0.3] = 0

    elif obj_type in ['LLNL_S03', 'LLNL']:
        f = np.load(f'{base}/LLNL/{P}/f_S03_008.npy')[:, :, :P]

    elif obj_type == 'LLNL_S12':
        f = np.load(f'{base}/LLNL/{P}/f_S12_001.npy')[:, :, :P]

    elif 'pde' in obj_type:
        if P == 256:
            f = np.load(
                f'{base}/{obj_type}/f_{obj_type}_{motion}_spatial_dim_{spatial_dim}_P_{P}.npy')
        elif P == 128:
            f = np.load(
                f'{base}/{obj_type}/f_{obj_type}_{motion}_spatial_dim_{spatial_dim}_P_{P}_2nd_half.npy')
        else:
            raise ValueError(f"pde objects only support P=256 or P=128; got P={P}.")

    elif obj_type == 'polymer_binary':
        f = np.load(f'{nitin}/48_binary_/obj_{P}_full.npy')

    elif obj_type == 'polymer_binary_subint':
        p_load = max(P, 32)
        f = np.load(f'{nitin}/48_binary_/obj_int_len_{p_load}_int_no_1.npy')
        if P < 32:
            f = f[..., ::32 // P]

    elif obj_type == 'polymer_binary_subint_hardest':
        p_load = max(P, 128)
        f = np.load(f'{nitin}/48_binary_/obj_int_len_{p_load}_int_no_5.npy')
        if P < 128:
            f = f[..., ::128 // P]

    elif obj_type == 'polymer_subint':
        p_load = max(P, 32)
        f = np.load(f'{nitin}/48_/obj_int_len_{p_load}_int_no_1.npy')
        if P < 32:
            f = f[..., ::32 // P]

    elif obj_type == 'polymer_subint_hardest':
        p_load = max(P, 128)
        f = np.load(f'{nitin}/48_/obj_int_len_{p_load}_int_no_5.npy')
        if P < 128:
            f = f[..., ::128 // P]

    else:
        raise ValueError(f"Unknown obj_type '{obj_type}'.")

    return f


# ---------------------------------------------------------------------------
# Image-quality metrics
# ---------------------------------------------------------------------------

def compute_psnr(x_gt: np.ndarray, x_rec: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio between a reconstruction and ground truth.

    Uses the dynamic range of x_gt (max - min) as the peak signal value.

    Args:
        x_gt: Ground-truth array, any shape.
        x_rec: Reconstructed array, same shape as x_gt.

    Returns:
        PSNR in dB.
    """
    mse = np.mean((x_rec - x_gt) ** 2)
    return 20 * np.log10((np.max(x_gt) - np.min(x_gt)) / np.sqrt(mse))


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def get_params(net: torch.nn.Module) -> list[torch.Tensor]:
    """Return all parameters of a network as a flat list.

    Args:
        net: PyTorch module.

    Returns:
        List of parameter tensors, suitable for passing to an optimiser.
    """
    return list(net.parameters())


def count_parameters(model: torch.nn.Module) -> int:
    """Count the total number of trainable parameters in a model.

    Args:
        model: PyTorch module.

    Returns:
        Total trainable parameter count.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_grad_norm(model: torch.nn.Module) -> float:
    """Compute the L2 norm of the gradient across all trainable parameters.

    Args:
        model: PyTorch module after a backward pass (gradients must be populated).

    Returns:
        L2 gradient norm scalar.
    """
    total_sq_norm = sum(
        p.grad.detach().data.norm(2).item() ** 2
        for p in model.parameters()
    )
    return total_sq_norm ** 0.5


# ---------------------------------------------------------------------------
# TV regularization losses
# ---------------------------------------------------------------------------

def total_variation_loss(img: torch.Tensor, weight: float) -> torch.Tensor:
    """Isotropic spatial total variation loss, averaged over all voxels.

    Note: this function expects the time axis first — shape (P, H, W) —
    which differs from the (H, W, P) convention used in loss.py.

    Args:
        img: Input tensor, shape (P, H, W).
        weight: Scalar multiplier applied to the loss.

    Returns:
        Weighted, normalised TV scalar: weight * (TV_h + TV_w) / (P * H * W).
    """
    P, h_img, w_img = img.size()
    tv_h = torch.abs(img[:, 1:, :] - img[:, :-1, :]).sum()
    tv_w = torch.abs(img[:, :, 1:] - img[:, :, :-1]).sum()
    return weight * (tv_h + tv_w) / (h_img * w_img * P)


def total_variation_spatiotemp_loss(
    img: torch.Tensor,
    weight_spatial: float,
    weight_temp: float,
) -> torch.Tensor:
    """Spatiotemporal total variation loss with separate spatial and temporal weights.

    Note: this function expects the time axis first — shape (P, H, W).

    Args:
        img: Input tensor, shape (P, H, W).
        weight_spatial: Scalar multiplier for the spatial (H, W) TV terms.
        weight_temp: Scalar multiplier for the temporal (P) TV term.

    Returns:
        Normalised combined loss:
        (weight_spatial * (TV_h + TV_w) + weight_temp * TV_t) / (P * H * W).
    """
    P, h_img, w_img = img.size()
    tv_h = torch.abs(img[:, 1:, :] - img[:, :-1, :]).sum()
    tv_w = torch.abs(img[:, :, 1:] - img[:, :, :-1]).sum()
    tv_t = torch.abs(img[1:, :, :] - img[:-1, :, :]).sum()
    return (weight_spatial * (tv_h + tv_w) + weight_temp * tv_t) / (h_img * w_img * P)


def total_variation_temp_loss(img: torch.Tensor, weight: float) -> torch.Tensor:
    """First-order temporal total variation loss, averaged over all voxels.

    Note: this function expects the time axis first — shape (P, H, W).

    Args:
        img: Input tensor, shape (P, H, W).
        weight: Scalar multiplier applied to the loss.

    Returns:
        Weighted, normalised temporal TV scalar: weight * TV_t / (P * H * W).
    """
    P, h_img, w_img = img.size()
    tv_t = torch.abs(img[1:, :, :] - img[:-1, :, :]).sum()
    return weight * tv_t / (h_img * w_img * P)
