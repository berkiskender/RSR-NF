"""Utility functions for RSR-NF.

Covers: data loading, Radon forward operators, measurement simulation,
positional encodings, image-quality metrics, and model I/O helpers.
"""

from pathlib import Path
from typing import Callable, Optional
import os
import sys

import numpy as np
from skimage.transform import radon, iradon
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_wavelet
from scipy import ndimage, sparse as sp

import torch
from torch import nn

import tomosipo as ts
from ts_algorithms import sirt, tv_min2d, nag_ls

import models_denoiser
sys.modules['red_psm_models'] = models_denoiser  # saved checkpoints reference the old module name
import models

np.set_printoptions(precision=2)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def mask_fov_object(f: np.ndarray, spatial_dim: int) -> np.ndarray:
    """Zero-out voxels outside the circular field-of-view for all time frames.

    Args:
        f: Object array, shape (H, W, P).
        spatial_dim: Side length of the square spatial grid (H == W == spatial_dim).

    Returns:
        f with all voxels outside the inscribed circle set to zero, in place.
    """
    ii, jj = np.meshgrid(
        np.arange(spatial_dim), np.arange(spatial_dim), indexing='ij')
    outside_fov = (
        (ii - spatial_dim // 2) ** 2 + (jj - spatial_dim // 2) ** 2
        >= (spatial_dim // 2) ** 2
    )
    f[outside_fov, :] = 0
    return f


def _decimal_to_binary(n: int) -> str:
    """Return the binary representation of n as a string without the '0b' prefix.

    Args:
        n: Non-negative integer.

    Returns:
        Binary string, e.g. _decimal_to_binary(6) == '110'.
    """
    return bin(n).replace("0b", "")


def _bit_reversal(x: int, N: int) -> int:
    """Reverse the bits of integer x within a log2(N)-bit field.

    Used internally by generate_theta to produce the bit-reversal angle schedule.

    Args:
        x: Integer whose bits are to be reversed (0 <= x < N).
        N: Total number of angles; must be a power of two.

    Returns:
        Integer obtained by reversing the binary digits of x.
    """
    num_digit = 0
    while N // 2:
        N = N // 2
        num_digit += 1
    digits = list(_decimal_to_binary(x))
    while len(digits) < num_digit:
        digits = [0] + digits
    digits.reverse()
    return int("".join(str(d) for d in digits), 2)


# ---------------------------------------------------------------------------
# Projection / measurement routines
# ---------------------------------------------------------------------------

def obtain_projections(f: np.ndarray, theta: np.ndarray, P: int) -> np.ndarray:
    """Compute one time-sequential projection per frame (dynamic undersampling).

    For each time frame p, the Radon transform is evaluated at a single view
    angle theta[p], giving one column of the sinogram per frame.

    Args:
        f: Time-varying object, shape (H, W, P).
        theta: View angles in radians, shape (P,). One angle per frame.
        P: Total number of time frames (must equal f.shape[-1]).

    Returns:
        Sinogram g, shape (H, P), where g[:, p] is the projection at theta[p].
    """
    g = np.zeros([f.shape[0], P])
    for p in range(P):
        g[:, p] = radon(f[:, :, p], theta=[360 * theta[p] / (2 * np.pi)])[:, 0]
    return g


def obtain_all_projections(f: np.ndarray, theta: np.ndarray, P: int) -> np.ndarray:
    """Compute the full sinogram for every time frame over all view angles.

    Args:
        f: Time-varying object, shape (H, W, P).
        theta: View angles in radians, shape (num_angles,).
        P: Total number of time frames (must equal f.shape[-1]).

    Returns:
        Full sinogram, shape (H, num_angles, P).
    """
    g = np.zeros([f.shape[0], len(theta), P])
    for p in range(P):
        g[:, :, p] = radon(f[:, :, p], theta=360 * theta / (2 * np.pi))
    return g


def static_recon(g: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Filtered back-projection reconstruction from a static sinogram.

    Args:
        g: Sinogram, shape (H, num_angles).
        theta: View angles in radians, shape (num_angles,).

    Returns:
        FBP reconstruction, shape (H, H).
    """
    return iradon(g, theta=360 * theta / (2 * np.pi), filter_name='ramp')


def generate_theta(
    P: int,
    ang_range: float = 2 * np.pi,
    period: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Compute view-angle schedules for dynamic tomographic acquisition.

    Four schemes are returned as a dict; the caller selects one by name.
    Each scheme produces P angles in [0, ang_range) by tiling a base pattern
    of length `period`.

    Args:
        P: Total number of view angles (one per time frame).
        ang_range: Angular range in radians; np.pi for pi-symmetric acquisition,
            2*np.pi for full rotation.
        period: Length of the base pattern before tiling. Defaults to P
            (no tiling).

    Returns:
        Dict with keys 'linear', 'random', 'bit_reversal', 'golden_angle',
        each mapping to an angle array of shape (P,) in radians.
    """
    period = P if period is None else period
    tile = lambda x: np.tile(x, P // period)

    theta_linear = tile(np.linspace(0, ang_range, period, endpoint=False))
    theta_random = tile(np.random.uniform(0, ang_range, size=[period]))
    theta_bit_reversal = tile(np.array([
        (ang_range / period) * _bit_reversal(p, period) for p in range(period)
    ]))
    theta_golden_angle = tile(np.array([
        ((p * (111.25 / 360) * 2 * np.pi) % (2 * np.pi)) for p in range(period)
    ]))
    return {
        'linear': theta_linear,
        'random': theta_random,
        'bit_reversal': theta_bit_reversal,
        'golden_angle': theta_golden_angle,
    }


def construct_pi_symm_g(
    g_radon: np.ndarray,
    g_radon_pi_symm: np.ndarray,
    ang_range: float,
) -> np.ndarray:
    """Concatenate pi-symmetric measurement pairs into an extended sinogram.

    When ang_range == pi, the pi-symmetric projections (acquired at angles
    theta + pi) are stacked below the original projections to form a sinogram
    of twice the detector-row count, exploiting the redundancy for better
    conditioning.

    Args:
        g_radon: Original sinogram, shape (H, P).
        g_radon_pi_symm: Pi-symmetric sinogram (at angles + pi), shape (H, P).
        ang_range: Acquisition angular range; np.pi triggers concatenation.

    Returns:
        Extended sinogram of shape (2*H, P) when ang_range == pi, or
        (g_radon, g_radon) tuple otherwise (2*pi case, no concatenation needed).
    """
    if ang_range == np.pi:
        g_symm_long = np.zeros([2 * g_radon.shape[0], g_radon.shape[1]])
        g_symm_long[:g_radon.shape[0], :] = g_radon
        g_symm_long[g_radon.shape[0]:, :] = g_radon_pi_symm
    else:
        g_symm_long = g_radon, g_radon
    return g_symm_long


def generate_f_pol(
    f: np.ndarray,
    P: int,
    spatial_dim: int,
    num_instances: int,
    theta_exp: np.ndarray,
    obj_type: str,
    period: Optional[int] = None,
    path: Optional[str] = None,
    save: bool = True,
    add_path: Optional[str] = None,
    rep: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the full sinogram and FBP reconstruction for each time frame.

    Args:
        f: Time-varying object, shape (H, W, P).
        P: Number of time frames.
        spatial_dim: Spatial side length (sinogram height = spatial_dim).
        num_instances: Number of sinogram instances to compute.
        theta_exp: View angles in radians, shape (rep*P,).
        obj_type: Object identifier string, used to build the save filename.
        period: Number of unique view angles per period (for filename suffix).
        path: Directory to save outputs. Required if save=True.
        save: Whether to write arrays to disk as .npy files.
        add_path: Optional filename suffix for material-specific variants.
        rep: Number of simultaneous views per time frame.

    Returns:
        f_pol_s: Full sinogram, shape (spatial_dim, rep*P, num_instances).
        f_true_recon: Per-frame FBP reconstruction, shape (H, W, P).
    """
    f_pol_s = np.zeros([spatial_dim, rep * P, num_instances])
    f_true_recon = np.zeros(f.shape)
    for t in range(num_instances):
        f_pol_s[..., t] = radon(
            f[..., t * rep * P // num_instances],
            theta=360 * theta_exp / (2 * np.pi))
        f_true_recon[..., t] = iradon(
            f_pol_s[..., t * rep * P // num_instances],
            theta=360 * theta_exp / (2 * np.pi))
    if save:
        str_period = '' if period is None else '_' + str(period)
        str_rep = '' if rep == 1 else '_rep_%d' % rep
        np.save(path + 'f_pol_s_%s%s_%d%s%s.npy' % (
            obj_type, add_path, P, str_period, str_rep), f_pol_s)
        np.save(path + 'f_true_recon_%s%s_%d%s%s.npy' % (
            obj_type, add_path, P, str_period, str_rep), f_true_recon)
    return f_pol_s, f_true_recon


def add_meas_noise(
    g_radon: np.ndarray,
    g_radon_pi_symm_long: np.ndarray,
    noise_std: float,
    g_max: float,
    ang_range: float,
    obj_type: str,
    P: int,
    period: Optional[int],
    path: str,
    save: bool = False,
    add_path: Optional[str] = None,
    rep: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Add zero-mean Gaussian noise to the time-sequential sinogram.

    Noise standard deviation is noise_std * g_max (relative to the sinogram
    maximum), matching the noise model used during denoiser pre-training.
    The pi-symmetric rows receive the flipped noise realisation from the
    original projections to preserve pi-symmetry.

    Args:
        g_radon: Clean sinogram, shape (H, P).
        g_radon_pi_symm_long: Extended (pi-symmetric) sinogram, shape (2*H, P).
        noise_std: Relative noise standard deviation.
        g_max: Maximum sinogram value used to scale the noise.
        ang_range: Acquisition angular range (np.pi triggers saving).
        obj_type: Object identifier, used to build the save filename.
        P: Number of time frames.
        period: Number of unique view angles per period (for filename suffix).
        path: Directory to save the noisy sinogram.
        save: Whether to write the noisy sinogram to disk.
        add_path: Optional filename suffix for material-specific variants.
        rep: Number of simultaneous views per frame (for filename suffix).

    Returns:
        g_radon_noisy: Noisy version of g_radon, shape (H, P).
        g_symm_long_noisy: Noisy extended sinogram, shape (2*H, P).
    """
    g_radon_noisy = g_radon + np.random.normal(
        loc=0.0, scale=noise_std * g_max, size=g_radon.shape)
    g_symm_long_noisy = np.zeros(g_radon_pi_symm_long.shape)
    g_symm_long_noisy[:g_radon.shape[0]] = g_radon_noisy
    g_symm_long_noisy[g_radon.shape[0]:] = (
        g_radon_pi_symm_long[g_radon.shape[0]:]
        + np.flipud(g_radon_noisy - g_radon)
    )
    if save and ang_range == np.pi:
        str_period = '' if period is None else '_' + str(period)
        str_rep = '' if rep == 1 else '_rep_%d' % rep
        np.save(
            path + 'g_radon_symm_long_noisy_%s%s_%d%s_noise_std_%.2e%s.npy' % (
                obj_type, add_path, P, str_period, noise_std, str_rep),
            g_symm_long_noisy)
    return g_radon_noisy, g_symm_long_noisy


# ---------------------------------------------------------------------------
# Forward operator
# ---------------------------------------------------------------------------

def load_radon_op(
    pi_symm: bool,
    spatial_dim: int,
    P: int,
    period: Optional[int] = None,
    path: Optional[str] = None,
) -> tuple[np.ndarray, torch.Tensor]:
    """Load the pre-computed differentiable Radon forward operator from disk.

    Prefers a compressed sparse float32 file (<stem>.sparse.npz) when available;
    falls back to the original dense float64 .npy file. The operator has shape
    (num_angles, spatial_dim, spatial_dim²); when P > period it is tiled.

    Args:
        pi_symm: If True, loads the pi-symmetric operator (half-scan angles).
        spatial_dim: Spatial side length d; reconstruction is d x d.
        P: Total number of view angles to cover.
        period: Number of unique angles in the stored operator. Defaults to P.
        path: Directory containing the operator file.

    Returns:
        R: Radon operator array, shape (P, spatial_dim, spatial_dim²).
        R_cuda: Same as R but as a CUDA FloatTensor for GPU use.
    """
    period = P if period is None else period
    stem = (
        'A_radon_spatial_dim_%d_P_%d_bit_reversal_pi_symm' % (spatial_dim, period)
        if pi_symm else
        'A_radon_spatial_dim_%d_P_%d_bit_reversal' % (spatial_dim, period)
    )
    R = sp.load_npz(path + stem + '.sparse.npz').toarray().reshape(period, spatial_dim, -1).astype(np.float64)

    R = np.tile(R, (P // period, 1, 1))
    R_cuda = torch.cuda.FloatTensor(R)
    return R, R_cuda


# ---------------------------------------------------------------------------
# Object loading
# ---------------------------------------------------------------------------

def load_f(obj_type: str, motion: str, spatial_dim: int, P: int) -> np.ndarray:
    """Load the ground-truth dynamic object data.

    File paths are hardcoded per object type. For object types that only exist
    at a minimum of 32 frames, sub-sampling is applied when P < 32.

    Args:
        obj_type: Dataset identifier. Supported values:
            'walnut',
            'polymer_binary', 'polymer_binary_subint',
            'polymer_binary_subint_hardest',
            'polymer_subint', 'polymer_subint_hardest'.
        motion: Motion type string.
        spatial_dim: Spatial side length.
        P: Number of time frames to load.

    Returns:
        Object array f, shape (H, W, P).

    Raises:
        ValueError: If obj_type is not recognized.
    """
    walnut_path = Path('data') / 'walnut'
    polymer_path = Path('data') / 'polymer'

    if obj_type == 'walnut':
        p_load = max(P, 32)
        f = np.load(
            walnut_path / f'f_{obj_type}_{motion}_spatial_dim_{spatial_dim}_P_{p_load}.npy')
        if P < 32:
            f = f[..., ::32 // P]

    elif obj_type == 'polymer_binary':
        f = np.load(polymer_path / '48_binary_' / f'obj_{P}_full.npy')

    elif obj_type == 'polymer_binary_subint':
        p_load = max(P, 32)
        f = np.load(polymer_path / '48_binary_' / f'obj_int_len_{p_load}_int_no_1.npy')
        if P < 32:
            f = f[..., ::32 // P]

    elif obj_type == 'polymer_binary_subint_hardest':
        p_load = max(P, 128)
        f = np.load(polymer_path / '48_binary_' / f'obj_int_len_{p_load}_int_no_5.npy')
        if P < 128:
            f = f[..., ::128 // P]

    elif obj_type == 'polymer_subint':
        p_load = max(P, 32)
        f = np.load(polymer_path / '48_' / f'obj_int_len_{p_load}_int_no_1.npy')
        if P < 32:
            f = f[..., ::32 // P]

    elif obj_type == 'polymer_subint_hardest':
        p_load = max(P, 128)
        f = np.load(polymer_path / '48_' / f'obj_int_len_{p_load}_int_no_5.npy')
        if P < 128:
            f = f[..., ::128 // P]

    else:
        raise ValueError(f"Unknown obj_type '{obj_type}'.")

    return f


# ---------------------------------------------------------------------------
# Denoiser loading
# ---------------------------------------------------------------------------

class Patchifier(nn.Module):
    """Spatially patchify an image volume and fold patches back.

    Wraps PyTorch's Unfold/Fold pair to extract non-overlapping (or strided)
    square patches from each frame and reconstruct the image from patches.
    """

    def __init__(self, patchSize: int, patchStride: int, spatial_dim: int, bs: int):
        """
        Args:
            patchSize: Side length of each square patch.
            patchStride: Stride between adjacent patches.
            spatial_dim: Spatial side length of the full image.
            bs: Batch size used during patch folding.
        """
        super().__init__()
        self.patchSize = patchSize
        self.patchStride = patchStride
        self.bs = bs
        fold_params = dict(kernel_size=[patchSize, patchSize], stride=patchStride)
        self.fold = nn.Fold(output_size=[spatial_dim, spatial_dim], **fold_params)
        self.unfold = nn.Unfold(**fold_params)

    def merge(self, patches: torch.Tensor) -> torch.Tensor:
        """Fold extracted patches back into a spatial image.

        Args:
            patches: Patch tensor, shape (bs * num_patches, 1, patchSize, patchSize).

        Returns:
            Reconstructed image, shape (bs, 1, spatial_dim, spatial_dim),
            normalised by the patch overlap count.
        """
        patches = patches.contiguous().view(
            self.bs, patches.shape[0] // self.bs,
            self.patchSize * self.patchSize).permute(0, 2, 1)
        x = self.fold(patches)
        return x / (self.patchSize ** 2 / self.patchStride ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patches from a spatial image.

        Args:
            x: Image tensor, shape (bs, 1, spatial_dim, spatial_dim).

        Returns:
            Patch tensor, shape (bs * num_patches, 1, patchSize, patchSize).
        """
        patches = self.unfold(x).permute(0, 2, 1)
        patches = patches.contiguous().view(
            self.bs * patches.shape[1], 1, self.patchSize, self.patchSize)
        return patches


# kept as module-level alias so existing call sites using the lowercase name still work
patchifier = Patchifier


def denoising_network_loader(
    train_type: str,
    denoiser_type: str,
    pSize: int,
    pStride: int,
    num_layers: int,
    spatial_dim: int,
    obj_type: str,
    num_channels: int,
    noise_est_type: str = 'direct',
    epochs: int = 500,
    noise_std: float = 5e-2,
    lambda_jr: Optional[float] = None,
) -> tuple[Callable, Optional[Patchifier]]:
    """Load a pretrained denoiser network from disk and freeze its parameters.

    Constructs the appropriate architecture, resolves the checkpoint path from
    the provided hyperparameters, loads weights, and moves the model to GPU.

    Args:
        train_type: Denoiser architecture / training variant. Supported values:
            'dncnn', 'dncnn_jreg', 'dncnn_oracle', 'dncnn_deblur',
            'dncnn_limited', 'dncnn_limited_deblur', 'unet', 'wavelet'.
        denoiser_type: Input handling variant: 'full_img', 'full_img_plinear',
            or 'patch_based_patchloss'.
        pSize: Patch size (used for patch-based variants and filename).
        pStride: Patch stride (used for patch-based variants and filename).
        num_layers: Number of denoiser layers.
        spatial_dim: Spatial side length of the input image.
        obj_type: Object type string used to resolve the checkpoint filename.
        num_channels: Number of feature channels in the denoiser.
        noise_est_type: Noise estimation mode ('direct' or 'residual').
        epochs: Number of pre-training epochs used to resolve the checkpoint.
        noise_std: Noise standard deviation used to resolve the checkpoint.
        lambda_jr: Jacobian regularization weight (only for 'dncnn_jreg').

    Returns:
        model_denoiser: Frozen denoiser callable. An nn.Module for network-based
            denoisers; a plain callable for the 'wavelet' variant.
        patchifier_red: Patchifier instance for patch-based denoisers, else None.

    Raises:
        NotImplementedError: If train_type or denoiser_type is not supported.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filter_size = 3
    denoiser_root = (
        '/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography'
        '/obj_domain_psm/data/denoiser'
    )
    print('Denoiser info:', train_type, obj_type, noise_est_type,
          epochs, num_layers, num_channels, '%.1e' % noise_std)
    if lambda_jr is not None:
        print('lambda_jr: %.1e' % lambda_jr)

    _network_types = [
        'unet', 'unet_deblur', 'unet_limited',
        'dncnn', 'dncnn_oracle', 'dncnn_jreg',
        'dncnn_deblur', 'dncnn_limited', 'dncnn_limited_deblur',
    ]

    if train_type in _network_types:
        if denoiser_type == 'full_img':
            model_name = train_type
            lambda_jr_str = (
                '_lambdajr_%.1e' % lambda_jr
                if train_type == 'dncnn_jreg' and lambda_jr is not None
                else ''
            )
            needs_noise_std = (
                'polymer_path' in obj_type
                or 'deblur' in train_type or 'limited' in train_type
            )
            if needs_noise_std:
                fname = (
                    f'{model_name}_model_{obj_type}_{noise_est_type}_epochs_{epochs}'
                    f'_num_layers_{num_layers}_num_ch_{num_channels}'
                    f'_noise_std_{noise_std:.1e}{lambda_jr_str}.pt'
                )
            else:
                fname = (
                    f'{model_name}_model_{obj_type}_{noise_est_type}_epochs_{epochs}'
                    f'_num_layers_{num_layers}_num_ch_{num_channels}{lambda_jr_str}.pt'
                )
            model_path = os.path.join(denoiser_root, fname)

            if train_type == 'unet':
                model_denoiser = models_denoiser.UNet(1, 1, 32).cuda()
            elif train_type in [
                'dncnn', 'dncnn_jreg', 'dncnn_oracle',
                'dncnn_deblur', 'dncnn_limited', 'dncnn_limited_deblur',
            ]:
                model_denoiser = models_denoiser.dncnn(
                    num_layers, num_channels, filter_size, noise_est_type).cuda()
            else:
                raise NotImplementedError(f"Unsupported train_type '{train_type}'.")
            patchifier_red = None

        elif denoiser_type == 'full_img_plinear':
            model_name = 'plinear_dncnn'
            fname = (
                f'{model_name}_model_{obj_type}_{noise_est_type}_epochs_{epochs}'
                f'_num_layers_{num_layers}_num_ch_{num_channels}.pt'
            )
            model_path = os.path.join(denoiser_root, fname)
            model_denoiser = models_denoiser.dncnn_plinear(
                num_layers, num_channels, filter_size, noise_est_type).cuda()
            patchifier_red = None

        elif denoiser_type == 'patch_based_patchloss':
            model_name = 'dncnn_patchbased_patchloss'
            fname = (
                f'{model_name}_model_{obj_type}_{noise_est_type}'
                f'_num_layers_{num_layers}_patch_size_{pSize}'
                f'_patch_stride_{pStride}_num_ch_{num_channels}_epochs_{epochs}.pt'
            )
            model_path = os.path.join(denoiser_root, fname)
            model_denoiser = models_denoiser.dncnnPatchBased_patchLoss(
                num_layers, num_channels, filter_size, noise_est_type).cuda()
            patchifier_red = Patchifier(pSize, pStride, spatial_dim, 1)

        else:
            raise NotImplementedError(f"Unsupported denoiser_type '{denoiser_type}'.")

        model_denoiser = torch.load(model_path)
        model_denoiser.eval()
        for _, v in model_denoiser.named_parameters():
            v.requires_grad = False
        model_denoiser = model_denoiser.to(device)
        n_params = sum(p.numel() for p in model_denoiser.parameters())
        print(f'{train_type} {denoiser_type}  path: {{{model_path}}}  params: {n_params}')

    elif train_type == 'wavelet':
        def model_denoiser(x: torch.Tensor) -> torch.Tensor:
            denoised = denoise_wavelet(
                x.squeeze().detach().cpu().numpy(),
                channel_axis=-1,
                method='BayesShrink',
                mode='soft',
                rescale_sigma=True,
            )
            return torch.tensor(
                np.nan_to_num(denoised), dtype=torch.float32, device=device)
        patchifier_red = None

    else:
        raise NotImplementedError(f"Unsupported train_type '{train_type}'.")

    return model_denoiser, patchifier_red


# ---------------------------------------------------------------------------
# Tomosipo operator and reconstruction
# ---------------------------------------------------------------------------

def tomo_op(
    spatial_dim: int,
    P: int,
    P_static: int,
    angles: np.ndarray,
    pixel_size: float = 1.0,
    rot_axis_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple:
    """Build a tomosipo parallel-beam geometry and operator.

    Args:
        spatial_dim: Spatial side length of the reconstruction volume.
        P: Number of detector columns (time frames).
        P_static: Unused; reserved for API consistency with caller.
        angles: Rotation angles in radians, shape (num_angles,).
        pixel_size: Physical size of each detector pixel and voxel.
        rot_axis_pos: (x, y, z) position of the rotation axis.

    Returns:
        A: Tomosipo forward operator.
        pg: Parallel-beam projection geometry.
        vg: Rotated volume geometry.
    """
    detector_shape = (spatial_dim, P)
    volume_shape = np.array([spatial_dim, spatial_dim, P])
    voxel_size = np.array([1.0, 1.0, 1.0])

    pg = ts.parallel_vec(
        shape=detector_shape,
        ray_dir=(0, 1, 0),
        det_pos=(0, 0, 0),
        det_v=(pixel_size, 0, 0),
        det_u=(0, 0, pixel_size),
    )
    vg0 = ts.volume(shape=volume_shape, pos=(0, 0, 0), size=volume_shape * voxel_size)
    R = ts.rotate(pos=rot_axis_pos, axis=(0, 0, 1), angles=angles)
    vg = R * vg0.to_vec()

    A = ts.operator(vg, pg)
    return A, pg, vg


def tomo_recon(
    R,
    g: torch.Tensor,
    num_iter: int,
    l2_reg: float,
    mode: str,
) -> torch.Tensor:
    """Run an iterative tomographic reconstruction algorithm.

    Args:
        R: Tomosipo forward operator.
        g: Sinogram tensor.
        num_iter: Number of iterations.
        l2_reg: L2 regularization weight (used only by 'nag_ls').
        mode: Algorithm name — 'SIRT', 'TV_min', or 'nag_ls'.

    Returns:
        Reconstructed volume tensor.

    Raises:
        NotImplementedError: If mode is not one of the supported algorithms.
    """
    if mode == 'SIRT':
        return sirt(R, g, num_iterations=num_iter, min_constraint=0)
    elif mode == 'TV_min':
        return tv_min2d(R, g, 0.0001, num_iterations=num_iter, min_constraint=0)
    elif mode == 'nag_ls':
        return nag_ls(R, g, num_iterations=num_iter, min_constraint=0,
                      l2_regularization=l2_reg)
    else:
        raise NotImplementedError(f"Unsupported reconstruction mode '{mode}'.")


def obtain_full_recon_cpu(
    g: torch.Tensor,
    theta: np.ndarray,
    spatial_dim: int,
    P: int,
    filt: str = 'ramp',
    interp: str = 'cubic',
) -> np.ndarray:
    """FBP reconstruction of a full sinogram stack on CPU.

    Args:
        g: Sinogram tensor (on any device), shape (spatial_dim, spatial_dim, P).
        theta: View angles in radians, shape (num_angles,).
        spatial_dim: Spatial side length of the reconstruction.
        P: Number of time frames.
        filt: Filter name passed to skimage's iradon (default 'ramp').
        interp: Interpolation method passed to iradon (default 'cubic').

    Returns:
        Per-frame FBP reconstructions, shape (spatial_dim, spatial_dim, P).
    """
    g_np = g.detach().cpu().numpy()
    f_rec = np.zeros([spatial_dim, spatial_dim, P])
    for t in range(P):
        f_rec[..., t] = iradon(
            g_np[..., t], theta=360 * theta / (2 * np.pi),
            filter_name=filt, interpolation=interp)
    return f_rec


# ---------------------------------------------------------------------------
# Coordinate grid
# ---------------------------------------------------------------------------

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
# Positional encoding
# ---------------------------------------------------------------------------

def pos_enc(
    xyt_grid: torch.Tensor,
    pos_enc_type: str,
    spatial_dim: int,
    mapping_size: int,
    scale: float,
    t_freq_ratio: float,
) -> tuple[torch.Tensor, int, Optional[nn.Module]]:
    """Apply a positional encoding to the raw spatiotemporal coordinate grid.

    Args:
        xyt_grid: Raw coordinate grid, shape (1, H, W, P, 3).
        pos_enc_type: Encoding type — 'fourier', 'gaussian', 'sine', or 'none'.
        spatial_dim: Spatial side length (used by the sine encoding).
        mapping_size: Number of Fourier features.
        scale: Frequency scale for the Gaussian/Fourier encoding.
        t_freq_ratio: Temporal-to-spatial frequency ratio for the Fourier encoding.

    Returns:
        xyt_grid_enc: Encoded coordinate tensor.
        input_ch: Number of channels in the encoded output (= input_dim of the NF model).
        model_enc: The encoding module (nn.Module), or None for 'none'.
    """
    if pos_enc_type == 'sine':
        model_enc = models.SineEncoding(
            input_dim=3, image_size=spatial_dim).cuda()
        xyt_grid_enc = model_enc(xyt_grid)
        input_ch = 6
    elif pos_enc_type == 'gaussian':
        model_enc = models.GaussianFourierEncoding3D(
            input_dim=3, image_size=spatial_dim,
            mapping_size=mapping_size, scale=scale).cuda()
        xyt_grid_enc = model_enc(xyt_grid)
        input_ch = 2 * mapping_size
    elif pos_enc_type == 'fourier':
        model_enc = models.FourierEncoding2D_t(
            input_dim=3, mapping_size=mapping_size,
            t_freq_ratio=t_freq_ratio).cuda()
        xyt_grid_enc = model_enc(xyt_grid)
        input_ch = 6 * mapping_size
    elif pos_enc_type == 'none':
        model_enc = None
        xyt_grid_enc = xyt_grid
        input_ch = 3
    else:
        raise ValueError(f"Unsupported pos_enc_type '{pos_enc_type}'.")
    return xyt_grid_enc, input_ch, model_enc


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


def compute_mae(x_gt: np.ndarray, x_rec: np.ndarray) -> float:
    """Mean Absolute Error between a reconstruction and ground truth.

    Args:
        x_gt: Ground-truth array, any shape.
        x_rec: Reconstructed array, same shape as x_gt.

    Returns:
        MAE scalar.
    """
    return np.mean(np.abs(x_rec - x_gt))


def compute_ssim(x_gt: np.ndarray, x_rec: np.ndarray) -> float:
    """Structural Similarity Index between a reconstruction and ground truth.

    Data range is derived from x_rec.

    Args:
        x_gt: Ground-truth 2-D or 3-D array.
        x_rec: Reconstructed array, same shape as x_gt.

    Returns:
        SSIM scalar in [-1, 1].
    """
    return ssim(x_gt, x_rec, data_range=x_rec.max() - x_rec.min())


def compute_hfen(f: np.ndarray, f_est: np.ndarray) -> float:
    """High-Frequency Error Norm between a reconstruction and ground truth.

    Applies a Laplacian-of-Gaussian filter (sigma=1.5) to the residual of
    each frame and returns the Frobenius norm of the result.

    Args:
        f: Ground-truth array, shape (H, W) or (H, W, P).
        f_est: Reconstructed array, same shape as f.

    Returns:
        HFEN scalar (lower is better).
    """
    log_filter = lambda x, y: ndimage.gaussian_laplace(x - y, sigma=1.5)

    if f.ndim == 2:
        f = f[..., None]
        f_est = f_est[..., None]

    hf_f = np.zeros(f.shape)
    for t in range(f.shape[-1]):
        hf_f[..., t] = log_filter(f[..., t], f_est[..., t])
    return np.linalg.norm(hf_f)


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a model.

    Args:
        model: PyTorch module.

    Returns:
        Total trainable parameter count.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_grad_norm(model: nn.Module) -> float:
    """Compute the L2 norm of the gradient across all trainable parameters.

    Args:
        model: PyTorch module after a backward pass (gradients must be populated).

    Returns:
        L2 gradient norm scalar.
    """
    total_sq_norm = sum(
        p.grad.detach().data.norm(2).item() ** 2
        for p in model.parameters()
        if p.requires_grad
    )
    return total_sq_norm ** 0.5


def get_params(net: nn.Module) -> list[torch.Tensor]:
    """Return all parameters of a network as a flat list.

    Args:
        net: PyTorch module.

    Returns:
        List of parameter tensors, suitable for passing to an optimiser.
    """
    return list(net.parameters())


# ---------------------------------------------------------------------------
# Model save / load
# ---------------------------------------------------------------------------

def save_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss: torch.Tensor,
    obj_type: str,
    num_layers: int,
    num_channels: int,
    pos_enc_type: str,
    mapping_size: int,
    scale: float,
    P: int,
    epoch: int,
    embed_model: str,
    t_freq_ratio: float,
) -> None:
    """Save a neural field model checkpoint to disk.

    Args:
        model: Neural field model.
        optimizer: Optimiser state.
        scheduler: LR scheduler state.
        loss: Most recent loss value.
        obj_type: Object type string (used in filename).
        num_layers: Number of network layers (filename).
        num_channels: Number of feature channels (filename).
        pos_enc_type: Positional encoding type (filename).
        mapping_size: Fourier mapping size (filename).
        scale: Frequency scale (filename).
        P: Number of time frames (filename).
        epoch: Current epoch number.
        embed_model: Embedding model identifier (filename).
        t_freq_ratio: Temporal frequency ratio (filename).
    """
    path = Path('data/oracle_init/model/')
    path.mkdir(parents=True, exist_ok=True)
    fname = path / (
        f'{obj_type}_nlyr_{num_layers}_nch_{num_channels}_enc_{pos_enc_type}'
        f'_mapS_{mapping_size}_sc_{scale}_P_{P}_{embed_model}_tfr_{t_freq_ratio:.3e}.pt'
    )
    torch.save({'epoch': epoch, 'model': model, 'optimizer': optimizer,
                'scheduler': scheduler, 'loss': loss}, fname)


def save_gauss_enc_model(
    model_enc: nn.Module,
    obj_type: str,
    num_layers: int,
    num_channels: int,
    pos_enc_type: str,
    mapping_size: int,
    scale: float,
    P: int,
    epoch: int,
    embed_model: str,
    t_freq_ratio: float,
) -> None:
    """Save a Gaussian Fourier encoding module's state dict to disk.

    Args:
        model_enc: Encoding module whose state dict is saved.
        obj_type: Object type string (used in filename).
        num_layers: Number of network layers (filename).
        num_channels: Number of feature channels (filename).
        pos_enc_type: Positional encoding type (filename).
        mapping_size: Fourier mapping size (filename).
        scale: Frequency scale (filename).
        P: Number of time frames (filename).
        epoch: Current epoch number (unused in filename; reserved).
        embed_model: Embedding model identifier (filename).
        t_freq_ratio: Temporal frequency ratio (filename).
    """
    path = Path('data/oracle_init/model_enc/')
    path.mkdir(parents=True, exist_ok=True)
    fname = path / (
        f'{obj_type}_nlyr_{num_layers}_nch_{num_channels}_enc_{pos_enc_type}'
        f'_mapS_{mapping_size}_sc_{scale}_P_{P}_{embed_model}_tfr_{t_freq_ratio:.3e}.pt'
    )
    torch.save({'model_state_dict': model_enc.state_dict()}, fname)


def load_model(
    input_ch: int,
    num_layers: int,
    num_channels: int,
    P: int,
    obj_type: str,
    pos_enc_type: str,
    mapping_size: int,
    scale: float,
    embed_model: str,
    t_freq_ratio: float,
) -> nn.Module:
    """Load a neural field model checkpoint from the oracle-init directory.

    Args:
        input_ch: Number of input channels for the model.
        num_layers: Number of network layers.
        num_channels: Number of feature channels.
        P: Number of time frames.
        obj_type: Object type string (used to resolve filename).
        pos_enc_type: Positional encoding type (filename).
        mapping_size: Fourier mapping size (filename).
        scale: Frequency scale (filename).
        embed_model: Embedding model identifier (filename).
        t_freq_ratio: Temporal frequency ratio (filename).

    Returns:
        Neural field model with weights loaded from checkpoint.
    """
    model = models.NeuralFieldModel3D_fc(
        input_ch=input_ch, output_dim=1,
        num_layers=num_layers, num_channels=num_channels, P=P).cuda()
    path = Path('data/oracle_init/model/')
    path.mkdir(parents=True, exist_ok=True)
    fname = path / (
        f'{obj_type}_nlyr_{num_layers}_nch_{num_channels}_enc_{pos_enc_type}'
        f'_mapS_{mapping_size}_sc_{scale}_P_{P}_{embed_model}_tfr_{t_freq_ratio:.3e}.pt'
    )
    print('NF Model path:', fname)
    checkpoint = torch.load(fname)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def load_model_enc(
    image_size: int,
    mapping_size: int,
    scale: float,
    obj_type: str,
    num_layers: int,
    num_channels: int,
    pos_enc_type: str,
    P: int,
    embed_model: str,
    t_freq_ratio: float,
) -> nn.Module:
    """Load a Gaussian Fourier encoding module from the oracle-init directory.

    Args:
        image_size: Spatial side length the encoding was built for.
        mapping_size: Number of Fourier features.
        scale: Frequency scale.
        obj_type: Object type string (filename).
        num_layers: Number of network layers (filename).
        num_channels: Number of feature channels (filename).
        pos_enc_type: Positional encoding type (filename).
        P: Number of time frames (filename).
        embed_model: Embedding model identifier (filename).
        t_freq_ratio: Temporal frequency ratio (filename).

    Returns:
        Encoding module with weights loaded from checkpoint.
    """
    model_enc = models.GaussianFourierEncoding3D(
        input_dim=3, image_size=image_size,
        mapping_size=mapping_size, scale=scale)
    path = Path('data/oracle_init/model_enc/')
    path.mkdir(parents=True, exist_ok=True)
    fname = path / (
        f'{obj_type}_nlyr_{num_layers}_nch_{num_channels}_enc_{pos_enc_type}'
        f'_mapS_{mapping_size}_sc_{scale}_P_{P}_{embed_model}_tfr_{t_freq_ratio:.3e}.pt'
    )
    checkpoint = torch.load(fname)
    model_enc.load_state_dict(checkpoint['model_state_dict'])
    return model_enc


def save_projection_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    obj_type: str,
    num_layers: int,
    num_channels: int,
    pos_enc_type: str,
    mapping_size: int,
    scale: float,
    P: int,
    epoch: int,
    embed_model: str,
    t_freq_ratio: float,
) -> None:
    """Save a projection-domain neural field model checkpoint to disk.

    Args:
        model: Projection-domain neural field model.
        optimizer: Optimiser state.
        loss: Most recent loss value.
        obj_type: Object type string (filename).
        num_layers: Number of network layers (filename).
        num_channels: Number of feature channels (filename).
        pos_enc_type: Positional encoding type (filename).
        mapping_size: Fourier mapping size (filename).
        scale: Frequency scale (filename).
        P: Number of time frames (filename).
        epoch: Current epoch number.
        embed_model: Embedding model identifier (filename).
        t_freq_ratio: Temporal frequency ratio (filename).
    """
    path = Path('data/oracle_init/model_projection/')
    path.mkdir(parents=True, exist_ok=True)
    fname = path / (
        f'{obj_type}_nlyr_{num_layers}_nch_{num_channels}_enc_{pos_enc_type}'
        f'_mapS_{mapping_size}_sc_{scale}_P_{P}_{embed_model}_tfr_{t_freq_ratio:.3e}.pt'
    )
    torch.save({'epoch': epoch, 'model': model,
                'optimizer': optimizer, 'loss': loss}, fname)


def load_projection_model(
    input_ch: int,
    num_layers: int,
    num_channels: int,
    P: int,
    obj_type: str,
    pos_enc_type: str,
    mapping_size: int,
    scale: float,
    embed_model: str,
    t_freq_ratio: float,
    P_static: int,
) -> nn.Module:
    """Load a projection-domain neural field model from the oracle-init directory.

    Args:
        input_ch: Number of input channels.
        num_layers: Number of network layers.
        num_channels: Number of feature channels.
        P: Number of time frames.
        obj_type: Object type string (filename).
        pos_enc_type: Positional encoding type (filename).
        mapping_size: Fourier mapping size (filename).
        scale: Frequency scale (filename).
        embed_model: Embedding model identifier (filename).
        t_freq_ratio: Temporal frequency ratio (filename).
        P_static: Number of static frames used during training.

    Returns:
        Projection-domain neural field model loaded from checkpoint.
    """
    model = models.NF3DProjDomain(
        input_ch=input_ch, output_dim=1,
        num_layers=num_layers, num_channels=num_channels,
        spatial_dim=128, P=P, act_type='relu',
        P_static=P_static, init=embed_model)
    path = Path('data/oracle_init/model_projection/')
    path.mkdir(parents=True, exist_ok=True)
    fname = path / (
        f'{obj_type}_nlyr_{num_layers}_nch_{num_channels}_enc_{pos_enc_type}'
        f'_mapS_{mapping_size}_sc_{scale}_P_{P}_{embed_model}_tfr_{t_freq_ratio:.3e}.pt'
    )
    checkpoint = torch.load(fname)
    return checkpoint['model']


def denoising_network_loader_proj(
    train_type: str,
    denoiser_type: str,
    pSize: int,
    pStride: int,
    num_layers: int,
    spatial_dim: int,
    obj_type: str,
    num_channels: int,
    noise_est_type: str = 'direct',
    epochs: int = 500,
    P_static: int = 128,
    noise_std_max: float = 5e-2,
) -> tuple[nn.Module, Optional[Patchifier]]:
    """Load a projection-domain DnCNN denoiser from disk.

    Args:
        train_type: Denoiser architecture (currently only 'dncnn' supported).
        denoiser_type: Input variant (currently only 'full_img' supported).
        pSize: Patch size (for filename / patch-based variants).
        pStride: Patch stride (for filename / patch-based variants).
        num_layers: Number of denoiser layers.
        spatial_dim: Spatial side length of the input sinogram.
        obj_type: Object type string (filename).
        num_channels: Number of feature channels.
        noise_est_type: Noise estimation mode ('direct' or 'residual').
        epochs: Pre-training epochs (filename).
        P_static: Number of static projection frames (filename).
        noise_std_max: Maximum noise level used in pre-training (filename).

    Returns:
        model_dncnn: Frozen DnCNN denoiser on device.
        patchifier_red: Patchifier instance or None.

    Raises:
        NotImplementedError: If train_type or denoiser_type is unsupported.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filter_size = 3
    denoiser_root = (
        '/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography'
        '/2D_neural_fields/own_imp/data/denoiser'
    )

    if train_type == 'dncnn':
        if denoiser_type == 'full_img':
            fname = (
                f'dncnn_g_model_{obj_type}_{noise_est_type}_epochs_{epochs}'
                f'_P_static_{P_static}_num_layers_{num_layers}'
                f'_num_ch_{num_channels}_noise_std_max_{noise_std_max:.2e}.pt'
            )
            model_path = os.path.join(denoiser_root, fname)
            models_denoiser.dncnn(
                num_layers, num_channels, filter_size, noise_est_type).cuda()
            patchifier_red = None
        else:
            raise NotImplementedError(f"Unsupported denoiser_type '{denoiser_type}'.")
    else:
        raise NotImplementedError(f"Unsupported train_type '{train_type}'.")

    model_dncnn = torch.load(model_path)
    model_dncnn.eval()
    for _, v in model_dncnn.named_parameters():
        v.requires_grad = False
    model_dncnn = model_dncnn.to(device)
    n_params = sum(p.numel() for p in model_dncnn.parameters())
    print(f'DnCNN {denoiser_type}  path: {{{model_path}}}  params: {n_params}')
    return model_dncnn, patchifier_red
