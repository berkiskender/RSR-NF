"""Regularization and model-parameter loss functions for RSR-NF.

All spatial/temporal regularizers operate on tensors of shape (H, W, P) where
H and W are the spatial dimensions and P is the number of time frames.  They
return a scalar tensor suitable for direct use in backpropagation.

The public entry point for the training loop is reg_loss(), which dispatches to
the appropriate regularizer by name, and model_reg(), which penalizes the
magnitude of the network's own parameters.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Spatial regularizers
# ---------------------------------------------------------------------------

def total_variation_loss(x: torch.Tensor, weight: float) -> torch.Tensor:
    """Isotropic spatial total variation loss, averaged over all voxels.

    Penalises the sum of horizontal and vertical finite differences across the
    spatial dimensions, encouraging piecewise-constant reconstructions.

    Args:
        x: Input tensor, shape (H, W, P).
        weight: Scalar multiplier applied to the loss.

    Returns:
        Weighted, normalised TV scalar: weight * (TV_h + TV_w) / (H * W * P).
    """
    h_x, w_x, P = x.size()
    tv_h = torch.abs(x[1:, :, :] - x[:-1, :, :]).sum()
    tv_w = torch.abs(x[:, 1:, :] - x[:, :-1, :]).sum()
    return weight * (tv_h + tv_w) / (h_x * w_x * P)


def l2_loss(x: torch.Tensor, weight: float) -> torch.Tensor:
    """Squared spatial finite-difference loss, averaged over all voxels.

    L2 analogue of total_variation_loss: penalises squared horizontal and
    vertical differences, favouring smooth reconstructions.

    Args:
        x: Input tensor, shape (H, W, P).
        weight: Scalar multiplier applied to the loss.

    Returns:
        Weighted, normalised L2 spatial loss: weight * (||Dh x||² + ||Dv x||²) / (H * W * P).
    """
    h_x, w_x, P = x.size()
    tv_h = (torch.abs(x[1:, :, :] - x[:-1, :, :]) ** 2).sum()
    tv_w = (torch.abs(x[:, 1:, :] - x[:, :-1, :]) ** 2).sum()
    return weight * (tv_h + tv_w) / (h_x * w_x * P)


# ---------------------------------------------------------------------------
# Temporal regularizers
# ---------------------------------------------------------------------------

def total_variation_temp_loss(x: torch.Tensor, weight: float) -> torch.Tensor:
    """First-order temporal total variation loss, averaged over all voxels.

    Penalises the L1 norm of frame-to-frame differences along the time axis,
    encouraging temporal smoothness.

    Args:
        x: Input tensor, shape (H, W, P).
        weight: Scalar multiplier applied to the loss.

    Returns:
        Weighted, normalised temporal TV scalar: weight * TV_t / (H * W * P).
    """
    h_x, w_x, P = x.size()
    tv_t = torch.abs(x[:, :, 1:] - x[:, :, :-1]).sum()
    return weight * tv_t / (h_x * w_x * P)


def l2_temp_loss(x: torch.Tensor, weight: float) -> torch.Tensor:
    """First-order temporal L2 loss, averaged over all voxels.

    Penalises the squared difference between the first and all subsequent frames,
    biasing the reconstruction towards a static solution.

    Args:
        x: Input tensor, shape (H, W, P).
        weight: Scalar multiplier applied to the loss.

    Returns:
        Weighted, normalised scalar: weight * ||x[:,:,0] - x[:,:,1:]||² / (H * W * P).
    """
    h_x, w_x, P = x.size()
    loss_t = (torch.abs(x[:, :, :1] - x[:, :, :-1]) ** 2).sum()
    return weight * loss_t / (h_x * w_x * P)


def l2_temp_group_loss(x: torch.Tensor, weight: float) -> torch.Tensor:
    """First-order temporal group-Frobenius loss, averaged over all voxels.

    Computes the Frobenius norm of the frame-to-frame difference matrix,
    coupling spatial locations within each temporal step.

    Args:
        x: Input tensor, shape (H, W, P).
        weight: Scalar multiplier applied to the loss.

    Returns:
        Weighted, normalised Frobenius norm: weight * ||Dt x||_F / (H * W * P).
    """
    h_x, w_x, P = x.size()
    loss_t = torch.norm(x[:, :, :1] - x[:, :, :-1], p='fro')
    return weight * loss_t / (h_x * w_x * P)


def sec_ord_temp_loss_l1(x: torch.Tensor, weight: float) -> torch.Tensor:
    """Second-order temporal L1 loss (temporal acceleration penalty).

    Penalises the L1 norm of the discrete second derivative along the time axis,
    encouraging linear (constant-velocity) temporal trajectories.

    Args:
        x: Input tensor, shape (H, W, P).
        weight: Scalar multiplier applied to the loss.

    Returns:
        Weighted, normalised scalar: weight * ||Dt² x||_1 / (H * W * P).
    """
    h_x, w_x, P = x.size()
    loss_t = torch.abs(x[:, :, 2:] - 2 * x[:, :, 1:-1] + x[:, :, :-2]).sum()
    return weight * loss_t / (h_x * w_x * P)


def sec_ord_temp_loss_l2(x: torch.Tensor, weight: float) -> torch.Tensor:
    """Second-order temporal L2 loss (temporal acceleration penalty).

    Penalises the squared L2 norm of the discrete second derivative along the
    time axis. Unlike the other regularizers this version is NOT normalised by
    the volume size (H * W * P) — the weight must be tuned accordingly.

    Args:
        x: Input tensor, shape (H, W, P).
        weight: Scalar multiplier applied to the loss.

    Returns:
        Weighted (unnormalised) scalar: weight * ||Dt² x||².
    """
    loss_t = (torch.abs(x[:, :, 2:] - 2 * x[:, :, 1:-1] + x[:, :, :-2]) ** 2).sum()
    return weight * loss_t


def sec_ord_temp_group_loss_l2(x: torch.Tensor, weight: float) -> torch.Tensor:
    """Second-order temporal group-Frobenius loss.

    Computes the Frobenius norm of the discrete second temporal derivative
    matrix, coupling spatial locations within each second-order difference.

    Args:
        x: Input tensor, shape (H, W, P).
        weight: Scalar multiplier applied to the loss.

    Returns:
        Weighted, normalised Frobenius norm: weight * ||Dt² x||_F / (H * W * P).
    """
    h_x, w_x, P = x.size()
    loss_t = torch.norm(x[:, :, 2:] - 2 * x[:, :, 1:-1] + x[:, :, :-2], p='fro')
    return weight * loss_t / (h_x * w_x * P)


# ---------------------------------------------------------------------------
# Dispatcher and model regularizer
# ---------------------------------------------------------------------------

_LOSS_REGISTRY: dict[str, callable] = {
    'TV':                        total_variation_loss,
    'TV_temp_loss':              total_variation_temp_loss,
    'l2_temp_loss':              l2_temp_loss,
    'l2_temp_group_loss':        l2_temp_group_loss,
    'l2_loss':                   l2_loss,
    'l1_sec_ord_temp_loss':      sec_ord_temp_loss_l1,
    'l2_sec_ord_temp_loss':      sec_ord_temp_loss_l2,
    'l2_sec_ord_temp_group_loss': sec_ord_temp_group_loss_l2,
}


def reg_loss(x: torch.Tensor, loss_type: str, weight: float) -> torch.Tensor:
    """Dispatch to a named regularization loss function.

    Args:
        x: Input tensor, shape (H, W, P).
        loss_type: One of the supported regularizer names:
            'TV', 'TV_temp_loss', 'l2_temp_loss', 'l2_temp_group_loss',
            'l2_loss', 'l1_sec_ord_temp_loss', 'l2_sec_ord_temp_loss',
            'l2_sec_ord_temp_group_loss'.
        weight: Scalar multiplier forwarded to the chosen regularizer.

    Returns:
        Scalar regularization loss tensor.

    Raises:
        NotImplementedError: If loss_type is not in the supported set.
    """
    if loss_type not in _LOSS_REGISTRY:
        raise NotImplementedError(
            f"Unsupported loss_type '{loss_type}'. "
            f"Choose from: {sorted(_LOSS_REGISTRY)}"
        )
    return _LOSS_REGISTRY[loss_type](x, weight)


def model_reg(module: nn.Module, weight: float, norm_type: str) -> torch.Tensor:
    """L1 or L2 regularization penalty on all trainable model parameters.

    Concatenates all parameter tensors into a single vector and computes its
    weighted L1 or L2 norm, encouraging parameter magnitude to stay small.

    Args:
        module: Neural network module whose parameters are regularized.
        weight: Scalar multiplier applied to the norm.
        norm_type: Norm to apply — 'l1' for L1 (sparsity) or 'l2' for L2 (weight decay).

    Returns:
        Weighted parameter norm scalar: weight * ||params||_{norm_type}.

    Raises:
        NotImplementedError: If norm_type is not 'l1' or 'l2'.
    """
    params = torch.cat([p.view(-1) for p in module.parameters()])

    if norm_type == 'l1':
        return weight * torch.norm(params, 1)
    elif norm_type == 'l2':
        return weight * torch.norm(params, 2)
    else:
        raise NotImplementedError(f"Unsupported norm_type '{norm_type}'. Choose 'l1' or 'l2'.")
