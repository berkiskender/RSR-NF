"""
Reconstruction quality metric tracking for RSR-NF.
"""

from typing import Optional

import numpy as np
import torch.nn as nn

import utils


def update_metrics(
    f: np.ndarray,
    f_est: np.ndarray,
    f_nf_est: np.ndarray,
    metrics: dict[str, list[float]],
    rep: int,
    best_psnr_f_est: float,
    best_f_est: Optional[np.ndarray],
    model: nn.Module,
) -> tuple[dict[str, list[float]], float, np.ndarray]:
    """Append image-quality metrics for both the f-step and NF estimates.

    Computes PSNR, MAE, SSIM, and HFEN against the ground truth for both
    the ADMM f-step estimate (f_est) and the neural field estimate (f_nf_est),
    and appends the gradient norm of the model parameters. Updates the
    best-so-far NF estimate if the latest PSNR_nf exceeds the running maximum.

    Args:
        f: Ground-truth object, shape (H, W, P//rep).
        f_est: Current ADMM f-step estimate, shape (H, W, P//rep).
        f_nf_est: Current neural field estimate, shape (H, W, P//rep).
        metrics: Running history dict with list-valued entries for each metric key.
            Expected keys: 'PSNR_f', 'PSNR_nf', 'MAE_f', 'MAE_nf',
            'SSIM_f', 'SSIM_nf', 'HFEN_f', 'HFEN_nf', 'grad_nf'.
        rep: View repetition factor (unused here; retained for API consistency).
        best_psnr_f_est: Best PSNR_nf observed so far.
        best_f_est: Neural field estimate corresponding to best_psnr_f_est.
        model: Neural field model; used to compute the gradient norm.

    Returns:
        metrics: Updated metrics dict with one new entry per key.
        best_psnr_f_est: Updated best PSNR_nf (unchanged if current is not higher).
        best_f_est: Neural field estimate at the new best PSNR_nf, or unchanged.
    """
    metrics['PSNR_f'].append(utils.compute_psnr(f, f_est))
    metrics['PSNR_nf'].append(utils.compute_psnr(f, f_nf_est))
    metrics['MAE_f'].append(utils.compute_mae(f, f_est))
    metrics['MAE_nf'].append(utils.compute_mae(f, f_nf_est))
    metrics['SSIM_f'].append(utils.compute_ssim(f, f_est))
    metrics['SSIM_nf'].append(utils.compute_ssim(f, f_nf_est))
    metrics['HFEN_f'].append(utils.compute_hfen(f, f_est))
    metrics['HFEN_nf'].append(utils.compute_hfen(f, f_nf_est))
    metrics['grad_nf'].append(utils.compute_grad_norm(model))

    if best_psnr_f_est <= metrics['PSNR_nf'][-1]:
        best_psnr_f_est = metrics['PSNR_nf'][-1]
        best_f_est = f_nf_est

    return metrics, best_psnr_f_est, best_f_est


def update_static_metrics(
    f: np.ndarray,
    f_est: np.ndarray,
    z_est: np.ndarray,
    metrics: dict[str, list[float]],
    best_psnr_f_est: float,
    best_f_est: Optional[np.ndarray],
) -> tuple[dict[str, list[float]], float, np.ndarray]:
    """Append image-quality metrics for the static variable-splitting formulation.

    Tracks PSNR, MAE, and SSIM for both the neural field estimate (f_est) and
    the denoiser z-step estimate (z_est). Updates the best-so-far estimate
    based on PSNR_z (the denoiser output is the primary reconstruction target
    in the static formulation).

    Args:
        f: Ground-truth object, shape (H, W).
        f_est: Current neural field estimate, shape (H, W).
        z_est: Current denoiser z-step estimate, shape (H, W).
        metrics: Running history dict with list-valued entries for each metric key.
            Expected keys: 'PSNR_f', 'PSNR_z', 'MAE_f', 'MAE_z', 'SSIM_f', 'SSIM_z'.
        best_psnr_f_est: Best PSNR_z observed so far.
        best_f_est: z-step estimate corresponding to best_psnr_f_est.

    Returns:
        metrics: Updated metrics dict with one new entry per key.
        best_psnr_f_est: Updated best PSNR_z (unchanged if current is not higher).
        best_f_est: z-step estimate at the new best PSNR_z, or unchanged.
    """
    metrics['PSNR_f'].append(utils.compute_psnr(f, f_est))
    metrics['PSNR_z'].append(utils.compute_psnr(f, z_est))
    metrics['MAE_f'].append(utils.compute_mae(f, f_est))
    metrics['MAE_z'].append(utils.compute_mae(f, z_est))
    metrics['SSIM_f'].append(utils.compute_ssim(f, f_est))
    metrics['SSIM_z'].append(utils.compute_ssim(f, z_est))

    if best_psnr_f_est <= metrics['PSNR_z'][-1]:
        best_psnr_f_est = metrics['PSNR_z'][-1]
        best_f_est = z_est

    return metrics, best_psnr_f_est, best_f_est


def update_projection_metrics(
    g: np.ndarray,
    g_est: np.ndarray,
    g_nf_est: np.ndarray,
    metrics: dict[str, list[float]],
    best_psnr_g_est: float,
    best_g_est: Optional[np.ndarray],
) -> tuple[dict[str, list[float]], float, np.ndarray]:
    """Append sinogram-domain quality metrics for both the classical and NF estimates.

    Computes PSNR, MAE, SSIM, and HFEN between the reference sinogram g and
    both a classical estimate (g_est) and the neural field sinogram (g_nf_est).
    Updates the best-so-far NF sinogram estimate based on PSNR_nf.

    Args:
        g: Reference (noisy) sinogram, shape (2*spatial_dim, P).
        g_est: Classical sinogram estimate, shape (2*spatial_dim, P).
        g_nf_est: Neural field sinogram estimate, shape (2*spatial_dim, P).
        metrics: Running history dict with list-valued entries for each metric key.
            Expected keys: 'PSNR_g', 'PSNR_nf', 'MAE_g', 'MAE_nf',
            'SSIM_g', 'SSIM_nf', 'HFEN_g', 'HFEN_nf'.
        best_psnr_g_est: Best PSNR_nf observed so far.
        best_g_est: NF sinogram estimate corresponding to best_psnr_g_est.

    Returns:
        metrics: Updated metrics dict with one new entry per key.
        best_psnr_g_est: Updated best PSNR_nf (unchanged if current is not higher).
        best_g_est: NF sinogram estimate at the new best PSNR_nf, or unchanged.
    """
    metrics['PSNR_g'].append(utils.compute_psnr(g, g_est))
    metrics['PSNR_nf'].append(utils.compute_psnr(g, g_nf_est))
    metrics['MAE_g'].append(utils.compute_mae(g, g_est))
    metrics['MAE_nf'].append(utils.compute_mae(g, g_nf_est))
    metrics['SSIM_g'].append(utils.compute_ssim(g, g_est))
    metrics['SSIM_nf'].append(utils.compute_ssim(g, g_nf_est))
    metrics['HFEN_g'].append(utils.compute_hfen(g, g_est))
    metrics['HFEN_nf'].append(utils.compute_hfen(g, g_nf_est))

    if best_psnr_g_est <= metrics['PSNR_nf'][-1]:
        best_psnr_g_est = metrics['PSNR_nf'][-1]
        best_g_est = g_nf_est

    return metrics, best_psnr_g_est, best_g_est
