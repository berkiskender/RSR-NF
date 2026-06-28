from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import loss as loss_module
import metrics as metrics_module
import utils


def f_update(
    lmbda: float,
    beta: float,
    denoiser: nn.Module,
    f_est: torch.Tensor,
    f_nf_est: torch.Tensor,
    f: torch.Tensor,
    gamma: torch.Tensor,
    denoiser_type: str,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, list[float]]:
    """ADMM primal f-step via Regularization by Denoising (RED) framework applying static restoration priors (RSR).

    Computes the closed-form RED update:
        f_red = mask * relu(lmbda * D(f_est) + beta * (f_nf + gamma)) / (lmbda + beta)

    where D is the pretrained restoration operator. Also returns squared-norm errors against
    the ground-truth f for diagnostic tracking.

    Args:
        lmbda: RSR regularization weight.
        beta: ADMM augmented Lagrangian penalty weight.
        denoiser: Pretrained denoiser network (frozen).
        f_est: Current f estimate, shape (H, W, P).
        f_nf_est: Neural field estimate of f, shape (H, W, P).
        f: Ground-truth object, shape (H, W, P). Used only for error metrics.
        gamma: ADMM dual variable, shape (H, W, P).
        denoiser_type: Architecture variant; 'wavelet' skips the (P, 1, H, W) permutation.
        mask: Field-of-view binary mask, shape (H, W).

    Returns:
        f_red: Updated f estimate after the RED proximal step, shape (H, W, P).
        errors: List of four squared Frobenius norms:
            [||f - f_est||², ||f - D(f_est)||², ||f - mask*(f_nf+gamma)||², ||f - f_red||²].
    """
    f_est_permuted = f_est.permute(2, 0, 1)[:, None, :, :] if denoiser_type != 'wavelet' else f_est
    f_denoised = (
        mask[:, :, None] * F.relu(denoiser(f_est_permuted)).squeeze().permute(1, 2, 0)
        if denoiser_type != 'wavelet'
        else mask[:, :, None] * F.relu(denoiser(f_est))
    )
    f_red = (
        mask[:, :, None]
        * F.relu(lmbda * f_denoised + beta * (f_nf_est + gamma))
        / (lmbda + beta + 1e-8)
    )

    errors = [
        torch.norm(f - f_est).detach().cpu().numpy() ** 2,
        torch.norm(f - f_denoised).detach().cpu().numpy() ** 2,
        torch.norm(f - mask[:, :, None] * (f_nf_est + gamma)).detach().cpu().numpy() ** 2,
        torch.norm(f - f_red).detach().cpu().numpy() ** 2,
    ]

    return f_red, errors


def f_update_static(
    lmbda: float,
    beta: float,
    denoiser: nn.Module,
    f: torch.Tensor,
    f_nf_est: torch.Tensor,
    gamma: torch.Tensor,
    denoiser_type: str,
    P: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    """ADMM primal f-step for single-frame (static) reconstruction.

    Applies the RED proximal update without temporal indexing. Equivalent to
    f_update but operates on a single spatial volume rather than a temporal stack.

    Args:
        lmbda: RSR regularization weight.
        beta: ADMM augmented Lagrangian penalty weight.
        denoiser: Pretrained denoiser network (frozen).
        f: Current f estimate, shape (H, W, P).
        f_nf_est: Neural field estimate of f, shape (H, W, P).
        gamma: ADMM dual variable, shape (H, W, P).
        denoiser_type: Architecture variant; 'wavelet' skips the permutation step.
        P: Number of time frames (unused directly; retained for API consistency).
        mask: Field-of-view binary mask, shape (H, W).

    Returns:
        Updated f estimate after the RED proximal step, shape (H, W, P).
    """
    if denoiser_type == 'wavelet':
        f_denoised = F.relu(denoiser(f))
    else:
        f_denoised = (
            F.relu(denoiser(f.permute(2, 0, 1)[:, None, :, :]))
            .squeeze()[None, ...]
            .permute(1, 2, 0)
        )
    return (
        mask[:, :, None]
        * F.relu(lmbda * f_denoised + beta * (f_nf_est + gamma))
        / (lmbda + beta + 1e-8)
    )


def dual_variable_update(
    gamma: torch.Tensor,
    f: torch.Tensor,
    f_nf_est: torch.Tensor,
) -> torch.Tensor:
    """ADMM dual variable (scaled Lagrange multiplier) update.

    Implements the standard scaled-form ADMM dual ascent step:
        gamma <- gamma + f_nf - f

    Args:
        gamma: Current dual variable, shape (H, W, P).
        f: Current f-step primal estimate, shape (H, W, P).
        f_nf_est: Neural field primal estimate, shape (H, W, P).

    Returns:
        Updated dual variable, shape (H, W, P).
    """
    return gamma + f_nf_est - f


def learn_static_recon(
    R: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    g: np.ndarray,
    theta_exp: np.ndarray,
    beta: float,
    num_primal_iter: int,
) -> list[dict, torch.Tensor]:
    """Trains the model for static (single-frame) reconstruction.

    Runs num_primal_iter gradient steps minimising the data-fidelity term plus
    the ADMM variable-splitting penalty with the current dual variable absorbed
    into the model state.

    Args:
        R: Radon forward operator, shape (num_angles, spatial_dim, spatial_dim²).
        model: Neural field model carrying f_est, z_est, and gamma_est as attributes.
        optimizer: Parameter optimiser (e.g. AdamW).
        scheduler: Learning rate scheduler.
        criterion: Loss function applied to sinogram residuals (e.g. MSELoss).
        g: Measured sinogram, shape (num_angles, spatial_dim).
        theta_exp: View angles in degrees, shape (num_angles,).
        beta: ADMM augmented Lagrangian penalty weight.
        num_primal_iter: Number of gradient descent steps per call.

    Returns:
        loss_epoch: Dict of per-iteration loss histories keyed by component name.
        g_f_est: Final sinogram estimate, shape (spatial_dim, num_angles).
    """
    # Initialize loss vectors
    loss_epoch = {}
    [
        loss_epoch['total'],
        loss_epoch['SSIM_f'],
        loss_epoch['MAE_f'],
        loss_epoch['f'],
        loss_epoch['g'],
        loss_epoch['var_split'],
    ] = [[] for _ in range(6)]
    loss_epoch['PSNR_f'] = [-1e4]

    # Convert view angle and measurement inputs to torch
    g_gt = torch.Tensor(g).cuda()

    # Training
    for epoch in range(num_primal_iter):
        # Compute projections from the estimated object
        g_f_est = torch.einsum('pjs,s->pj', R, model.f_est.view(-1))

        # Compute the data fidelity
        loss_g = criterion(g_f_est, g_gt)

        # Compute the Lagrangian
        loss_var_split = (beta / 2) * torch.linalg.norm(
            model.f_est - model.z_est + model.gamma_est
        ) ** 2
        loss = loss_g + loss_var_split

        # Backprop
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()
        scheduler.step()

        # Log computed loss values
        loss_epoch['total'].append(loss.data.cpu().numpy())
        loss_epoch['g'].append(loss_g.data.cpu().numpy())
        loss_epoch['var_split'].append(loss_var_split.data.cpu().numpy())

    return [loss_epoch, g_f_est]


def dual_variable_update_static(
    gamma: torch.Tensor,
    f: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """ADMM dual variable update for the static variable-splitting formulation.

    Scaled-form dual ascent step for the (f, z) splitting:
        gamma <- gamma + f - z

    Args:
        gamma: Current dual variable, shape (H, W).
        f: Current neural field estimate, shape (H, W).
        z: Current denoiser output (z-step estimate), shape (H, W).

    Returns:
        Updated dual variable, shape (H, W).
    """
    return gamma + f - z


def z_update_static(
    lmbda: float,
    beta: float,
    denoiser: nn.Module,
    patchifier: nn.Module | None,
    z: torch.Tensor,
    f: torch.Tensor,
    gamma: torch.Tensor,
    denoiser_type: str,
    P: int,
) -> torch.Tensor:
    """ADMM z-step (denoiser proximal update) for static reconstruction.

    Computes the closed-form minimiser of the z-subproblem:
        z* = (lmbda * D(z) + beta * (f + gamma)) / (lmbda + beta)

    Currently only supports full-image denoiser types.

    Args:
        lmbda: RSR regularization weight.
        beta: ADMM augmented Lagrangian penalty weight.
        denoiser: Pretrained full-image denoiser network (frozen).
        patchifier: Patch extraction/aggregation module (unused for full_img types).
        z: Current z estimate, shape (H, W).
        f: Current neural field estimate, shape (H, W).
        gamma: Current dual variable, shape (H, W).
        denoiser_type: Must be 'full_img' or 'full_img_plinear'.
        P: Number of time frames (unused; retained for API consistency).

    Returns:
        Updated z estimate, shape (H, W).

    Raises:
        NotImplementedError: If denoiser_type is not a supported full-image variant.
    """
    if denoiser_type in ['full_img', 'full_img_plinear']:
        return (lmbda * denoiser(z[None, None, :, :]).squeeze() + beta * (f + gamma)) / (
            lmbda + beta
        )
    else:
        raise NotImplementedError('ADMM f update not implemented for the denoiser type.')


def learn_NF_proj(
    model: nn.Module,
    model_enc: nn.Module,
    xyt_grid: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    g: np.ndarray,
    num_primal_iter: int,
    P: int,
    spatial_dim: int,
    R: torch.Tensor | None = None,
    gamma_est: torch.Tensor | None = None,
    beta: float = 1,
    rep: int = 4,
    tv_weight: float = 0,
) -> list:
    """Neural field training step using full-batch projection loss.

    Runs num_primal_iter gradient steps on the combined objective:
        L = ||R * f_nf - g||² + (beta/2) * ||f_nf - f_est + gamma||² + TV(f_nf)

    Uses a positional encoding module (model_enc) applied on the fly to xyt_grid.

    Args:
        model: Neural field model with f_est and gamma_est attributes.
        model_enc: Positional encoding module mapping raw xyt_grid to encoded features.
        xyt_grid: Raw spatiotemporal coordinate grid, shape (1, H, W, P, 3).
        optimizer: Parameter optimiser.
        scheduler: Learning rate scheduler.
        criterion: Data-fidelity loss (e.g. MSELoss with reduction='sum').
        g: Measured sinogram (numpy), shape (2*num_angles, P).
        num_primal_iter: Number of gradient steps.
        P: Total number of time frames.
        spatial_dim: Spatial side length d (reconstruction is d x d).
        R: Radon forward operator, shape (num_angles*P, spatial_dim, spatial_dim²).
        gamma_est: Current ADMM dual variable, shape (H, W, P).
        beta: ADMM augmented Lagrangian penalty weight.
        rep: Number of simultaneous views per time frame.
        tv_weight: Weight for the temporal total variation regularizer.

    Returns:
        [loss_epoch, f_nf_est, g_f_est]:
            loss_epoch: Dict of per-iteration loss histories.
            f_nf_est: Final neural field estimate, shape (H, W, P).
            g_f_est: Final sinogram estimate, shape (spatial_dim, num_angles*P).
    """
    # Initialize loss vectors
    loss_epoch = {}
    [
        loss_epoch['total'],
        loss_epoch['SSIM_f'],
        loss_epoch['MAE_f'],
        loss_epoch['f'],
        loss_epoch['g'],
        loss_epoch['var_split'],
        loss_epoch['tv'],
    ] = [[] for _ in range(7)]
    loss_epoch['PSNR_f'] = [-1e4]

    # Convert view angle and measurement inputs to torch
    g_gt = torch.Tensor(g).cuda()

    # Training
    for _ in range(num_primal_iter):
        xyt_grid_enc = model_enc(xyt_grid)
        f_nf_est = model(xyt_grid_enc)

        # Compute projections from the estimated object
        g_f_est = torch.einsum(
            'pjs,ps->jp',
            R,
            torch.repeat_interleave(f_nf_est.squeeze().permute(2, 0, 1), repeats=rep, dim=0).view(
                P, spatial_dim**2
            ),
        )

        # Compute the data fidelity
        loss_g = criterion(g_f_est, g_gt)

        # Compute the Lagrangian
        loss_var_split = (beta / 2) * torch.linalg.norm(
            f_nf_est.squeeze() - model.f_est.squeeze() + gamma_est
        ) ** 2

        loss_tv = loss_module.total_variation_temp_loss(
            f_nf_est.squeeze().permute(2, 0, 1), tv_weight
        )

        loss = loss_g + loss_var_split + loss_tv  # + geo_weight * xyt_grad_sum

        # Backprop
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        # Log computed loss values
        loss_epoch['total'].append(loss.data.cpu().numpy())
        loss_epoch['g'].append(loss_g.data.cpu().numpy())
        loss_epoch['var_split'].append(loss_var_split.data.cpu().numpy())

    return [loss_epoch, f_nf_est, g_f_est]


def learn_NF_SGD_multiview(
    model: nn.Module,
    xyt_grid_enc: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    g_gt: torch.Tensor,
    num_primal_iter: int,
    P: int,
    spatial_dim: int,
    R: torch.Tensor,
    beta: float,
    rep: int,
    reg_t_weight: float,
    reg_t_loss_type: str,
    weight_model: float,
    type_model_reg: str,
    sgd_size: int,
) -> list:
    """Neural field SGD training step with stochastic time-index subsampling.

    The main ADMM primal NF-step. At each inner iteration a random minibatch of
    sgd_size time indices is drawn, the neural field is evaluated at those frames,
    and the total objective is minimised:
        L = ||R[t] * f_nf[t] - g[t]||²
            + (beta/2) * ||f_nf[t] - f_est[t] + gamma[t]||²
            + reg_t_weight * temporal_reg(f_nf)
            + weight_model * model_reg(theta)

    When epoch % (rep*P // sgd_size) == 0 and reg_t_weight != 0, the full
    temporal sequence is used so the temporal regularizer sees all frames.
    After all inner iterations the full f_nf_est is recomputed under no_grad.

    Args:
        model: Neural field model with f_est and gamma_est attributes.
        xyt_grid_enc: Pre-encoded spatiotemporal grid, shape (1, H, W, P, C).
        optimizer: Parameter optimiser.
        scheduler: Learning rate scheduler.
        criterion: Data-fidelity loss (e.g. MSELoss with reduction='sum').
        g_gt: Noisy sinogram on GPU, shape (2*spatial_dim, rep*P).
        num_primal_iter: Number of inner SGD steps.
        P: Total number of time frames.
        spatial_dim: Spatial side length d (reconstruction is d x d).
        R: Radon forward operator on GPU, shape (rep*P, spatial_dim, spatial_dim²).
        beta: ADMM augmented Lagrangian penalty weight.
        rep: Number of simultaneous views per time frame.
        reg_t_weight: Weight for the temporal regularization term.
        reg_t_loss_type: Temporal regularizer type (e.g. 'TV', 'l2_sec_ord_temp_loss').
        weight_model: Weight for the model parameter regularization term.
        type_model_reg: Model regularizer norm type ('l1' or 'l2').
        sgd_size: Number of time indices sampled per minibatch.

    Returns:
        [loss_epoch, f_nf_est, g_f_est]:
            loss_epoch: Dict of per-iteration loss histories keyed by component name.
            f_nf_est: Full neural field estimate (all P frames), shape (H, W, P).
            g_f_est: Sinogram estimate for the last minibatch, shape (spatial_dim, sgd_size).
    """
    # Initialize loss vectors
    loss_epoch = {key: [] for key in ['total', 'f', 'g', 'var_split', 'temp_reg', 'model_reg']}

    for epoch in range(num_primal_iter):
        time_idx = (
            np.arange(rep * P)
            if epoch % (rep * P // sgd_size) == 0 and reg_t_weight != 0
            else np.random.choice(np.arange(rep * P), size=sgd_size, replace=False)
        )

        f_nf_est = model(xyt_grid_enc[..., time_idx // rep])[0, 0]

        # Compute projections from the estimated object
        g_f_est = torch.einsum(
            'pjs,ps->jp', R[time_idx], f_nf_est.permute(2, 0, 1).view(len(time_idx), spatial_dim**2)
        )

        # Compute the data fidelity
        loss_g = criterion(g_f_est, g_gt[:, time_idx])

        # Compute the Lagrangian
        loss_var_split = (
            (beta / 2)
            * torch.linalg.norm(
                f_nf_est
                - model.f_est[:, :, time_idx // rep]
                + model.gamma_est[:, :, time_idx // rep]
            )
            ** 2
            if beta != 0
            else torch.zeros(1).cuda()
        )

        # Temporal regularization
        loss_temp_reg = (
            loss_module.reg_loss(f_nf_est[..., ::rep], reg_t_loss_type, reg_t_weight)
            if reg_t_weight != 0 and epoch % (rep * P // sgd_size) == 0
            else torch.zeros(1).cuda()
        )

        # l1 sparsity model
        loss_model_reg = (
            loss_module.model_reg(model, weight_model, type_model_reg)
            if weight_model != 0
            else torch.zeros(1).cuda()
        )

        loss = loss_g + loss_var_split + loss_temp_reg + loss_model_reg

        # Backprop
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        # Log computed loss values
        for key, value in zip(
            ['total', 'g', 'var_split', 'temp_reg', 'model_reg'],
            [loss, loss_g, loss_var_split, loss_temp_reg, loss_model_reg],
        ):
            loss_epoch[key].append(value.data.cpu().numpy())

    with torch.no_grad():
        f_nf_est = model(xyt_grid_enc).squeeze()

    return [loss_epoch, f_nf_est, g_f_est]


def static_embedding(
    model: nn.Module,
    xyt_grid_enc: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    f_static_embed: np.ndarray,
    f: np.ndarray,
    P: int,
    REP: int,
    num_epoch: int,
    sgd_size: int,
    rep: int,
) -> tuple[dict, float, np.ndarray, torch.Tensor, list]:
    """Initialises the neural field by fitting to a static FBP reconstruction.

    Pre-trains the model on a static (time-replicated) FBP estimate before the
    main ADMM loop, giving a warm start that avoids trivial local minima. Uses
    the same SGD minibatch strategy as learn_NF_SGD_multiview.

    After training, model.f_est is updated with the final neural field estimate
    and model.gamma_est is reset to zero.

    Args:
        model: Neural field model whose weights are updated in place.
        xyt_grid_enc: Pre-encoded spatiotemporal grid, shape (1, H, W, P, C).
        optimizer: Parameter optimiser (separate instance from the main loop).
        criterion: Loss function (e.g. MSELoss) comparing f_nf to the FBP target.
        f_static_embed: Static FBP reconstruction to fit, shape (H, W).
            Tiled internally to (H, W, REP*P).
        f: Ground-truth object for PSNR tracking, shape (H, W, P).
        P: Total number of time frames.
        REP: Number of simultaneous views per frame (used to tile f_static_embed).
        num_epoch: Number of outer training epochs.
        sgd_size: Number of time indices sampled per minibatch.
        rep: Subsampling stride when comparing against ground truth (f[..., ::rep]).

    Returns:
        metrics: Dict of per-checkpoint metric histories (PSNR, SSIM, MAE, HFEN, grad).
        best_psnr_f_static: Best PSNR_nf value observed during training.
        best_f_static_est: Neural field estimate at the epoch with best PSNR_nf.
        f_nf_est: Final full neural field estimate, shape (1, H, W, P, 1).
        loss_epoch: Per-iteration scalar loss values.
    """
    # Initialize recon metrics per epoch
    metrics = {
        key: []
        for key in [
            'PSNR_f',
            'PSNR_nf',
            'MAE_f',
            'MAE_nf',
            'SSIM_f',
            'SSIM_nf',
            'HFEN_f',
            'HFEN_nf',
            'grad_nf',
        ]
    }
    best_psnr_f_static = 1e0
    best_f_static_est = None

    upd_freq = num_epoch // 20
    loss_epoch = []
    f_static_embed = torch.Tensor(f_static_embed).cuda()[..., None].repeat(1, 1, REP * P)

    Nepoch = tqdm(range(num_epoch), desc='Initializing...', leave=True, ncols=160, colour='green')
    for epoch in Nepoch:
        time_idx = (
            np.arange(rep * P)
            if epoch % (rep * P // sgd_size) == 0
            else np.random.choice(np.arange(rep * P), size=sgd_size, replace=False)
        )
        f_nf_est = model(xyt_grid_enc[:, :, :, :, time_idx // rep]).squeeze()

        loss = criterion(f_nf_est.squeeze(), f_static_embed[..., time_idx].squeeze())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_epoch.append(loss.data.detach().cpu().numpy())

        # Compute PSNR, SSIM, MAE, HFEN
        if epoch % (num_epoch // upd_freq) == 0:
            with torch.no_grad():
                f_nf_est = model(xyt_grid_enc)
                metrics, best_psnr_f_static, best_f_static_est = metrics_module.update_metrics(
                    f[..., ::rep],
                    model.f_est.squeeze().detach().cpu().numpy(),
                    f_nf_est.squeeze().detach().cpu().numpy(),
                    metrics,
                    rep,
                    best_psnr_f_static,
                    best_f_static_est,
                    model,
                )

                Nepoch.set_description(
                    'PSNR st:%.3e/%.3e SSIM st:%.2e/%.2e MAE st:%.2e/%.2e HFEN st:%.2e/%.2e grad_nf st:%.3e'
                    % (
                        metrics['PSNR_f'][-1],
                        metrics['PSNR_nf'][-1],
                        metrics['SSIM_f'][-1],
                        metrics['SSIM_nf'][-1],
                        metrics['MAE_f'][-1],
                        metrics['MAE_nf'][-1],
                        metrics['HFEN_f'][-1],
                        metrics['HFEN_nf'][-1],
                        metrics['grad_nf'][-1],
                    )
                )

    with torch.no_grad():
        f_nf_est = model(xyt_grid_enc)
        model.f_est = f_nf_est.detach().clone().squeeze()
        model.gamma_est = torch.zeros(model.f_est.shape).cuda()

    return metrics, best_psnr_f_static, best_f_static_est, f_nf_est, loss_epoch
