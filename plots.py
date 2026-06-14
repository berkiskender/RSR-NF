"""Visualization helpers for RSR-NF.

Covers: per-iteration metric convergence plots and training diagnostics.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt


# ---------------------------------------------------------------------------
# Training visualisation helpers
# ---------------------------------------------------------------------------

def train_visualization(
    loss_epoch: dict,
    f_est: np.ndarray,
    f_psm_est: np.ndarray,
    gamma_bar_est: np.ndarray,
    f: np.ndarray,
    t: int,
    plot: bool,
) -> None:
    """Display intermediate training results for PSM-based RED.

    Shows a row of loss-component curves and a row of per-frame image panels
    (estimated object, NF estimate, dual variable, ground truth, residual).

    Args:
        loss_epoch: Dict of per-iteration loss scalars with keys
            'total', 'temp_reg', 'g', 'var_split'.
        f_est: Current ADMM f-step estimate, shape (H, W, P).
        f_psm_est: Current NF estimate, shape (H, W, P).
        gamma_bar_est: Current dual variable estimate, shape (H, W, P).
        f: Ground-truth object, shape (H, W, P).
        t: Time-frame index to display.
        plot: If False, this function is a no-op.
    """
    if not plot:
        return

    metrics = ['total', 'temp_reg', 'g', 'var_split']
    titles = [
        r'$\ln$(loss)', r'$\ln$(loss $t$ reg)',
        r'$\ln$(loss $g$)', r'$\ln$(loss aug lag)',
    ]

    plt.figure(figsize=(15, 2))
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(1, 9, i + 1)
        plt.title(title)
        plt.plot(np.log(loss_epoch[metric]))
        plt.grid()

    images = [
        f_est[..., t], f_psm_est[..., t],
        gamma_bar_est[..., t], f[..., t],
        f_est[..., t] - f[..., t],
    ]
    img_titles = [r'$f_{est}$', r'$f_{nf}$', r'$\gamma_{est}$', r'$f$', r'$f-f_{est}$']
    for i, (img, img_title) in enumerate(zip(images, img_titles)):
        plt.subplot(1, 9, i + 5)
        plt.title(img_title)
        plt.imshow(img)
        plt.colorbar()
    plt.show()


def red_visualization(metrics: dict, plot: bool) -> None:
    """Display RED training convergence curves.

    Plots log-scale norm-error curves for the four internal RED stages:
    red_in, denoised, red_out, and dual.

    Args:
        metrics: Dict with keys 'f_red_in_norm_err', 'f_denoised_norm_err',
            'f_red_out_norm_err', 'f_dual_norm_err', each a list of scalars.
        plot: If False, this function is a no-op.
    """
    if not plot:
        return

    plt.figure(figsize=(12, 6))
    plt.title(r'RED $\ln$(loss)')
    for label, linestyle in zip(
        ['red_in', 'denoised', 'red_out', 'dual'], ['--', '-.', '-', '--']
    ):
        plt.plot(
            np.log(metrics[f'f_{label}_norm_err']),
            label=label, linestyle=linestyle, linewidth=0.5)
    plt.grid()
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------------
# Neural-field RED result plots
# ---------------------------------------------------------------------------

def plot_and_save_red_nf_results(
    params: dict,
    obj_type: str,
    results_per_config: list[tuple],
    P: int,
    update_type: str,
    noise_std: float,
    ang_period: Optional[int],
    pos_enc_type: str,
    mapping_size: int,
    scale: float,
    save_fig: bool,
) -> None:
    """Plot per-iteration metric curves for a sweep of NF-RED configurations.

    Prints metrics in experimental order, plots PSNR/SSIM/MAE/HFEN convergence
    curves and gradient-norm curves, then summarises the best-PSNR result for
    each configuration. Optionally saves figures, metric arrays, and text logs.

    Args:
        params: Global experiment parameters dict. Required keys:
            'denoiser_type', optionally 'pSize', 'pStride', 'num_layers_denoiser',
            'num_channels_denoiser', 'pos_enc_type_sweep'.
        obj_type: Dataset identifier string (used in output directory name).
        results_per_config: List of per-configuration result tuples.
            Each tuple: (best_psnr, metrics_dict, _, hparams_dict, data_dict).
            hparams_dict keys: 'beta', 'lmbda', 'num_layers', 'num_channels',
            'mapping_size', 'scale', 'reg_t_weight', 'sgd_size', 'weight_model',
            'denoiser_model', 'num_epoch', 'num_primal_iter', 'lr_primal',
            'rep', 'reg_mode', 'init'.
            metrics_dict keys: 'PSNR_f', 'SSIM_f', 'MAE_f', 'HFEN_f', 'grad_nf'.
            data_dict keys: 'f_est'.
        P: Number of time frames.
        update_type: ADMM update mode string (used in directory name).
        noise_std: Measurement noise standard deviation (used in filenames).
        ang_period: Angle period for bit-reversal schedule; None for aperiodic.
        pos_enc_type: Positional encoding type (used in directory name).
        mapping_size: Fourier feature mapping size (used in directory name).
        scale: Frequency scale (used in directory name).
        save_fig: If True, save figures as .jpg/.pdf, arrays as .npy, and a
            text summary as .txt.
    """
    str_period = f'_{ang_period}' if ang_period else ''
    out_dir = (
        f'data/REBUTTAL_NF_2d_t_CT_results/{obj_type}_{P}{str_period}'
        f'_denoiser_{params["denoiser_type"]}_{update_type}'
        f'_enc_{pos_enc_type}_mapS_{mapping_size}_sc_{scale}/'
    )
    if params['denoiser_type'] == 'patch_based':
        out_dir += (
            f'_patch_sz_{params["pSize"]}_patch_str_{params["pStride"]}'
            f'_num_layers_{params["num_layers"]}_num_ch_{params["num_channels"]}'
            f'_num_layers_denoiser_{params["num_layers_denoiser"]}'
            f'_num_ch_denoiser_{params["num_channels_denoiser"]}'
        )
    out_dir += 'data/'
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print('Results with experimental order:\n')
    for res in results_per_config:
        print(
            f'{res[0]} PSNR: {res[1]["PSNR_f"][-1]:.2e} '
            f'SSIM: {res[1]["SSIM_f"][-1]:.2e} '
            f'MAE: {res[1]["MAE_f"][-1]:.2e} '
            f'HFEN: {res[1]["HFEN_f"][-1]:.2e} {res[3]}'
        )

    results_sorted = sorted(results_per_config, key=lambda res: res[0])

    # --- metric convergence curves ---
    metric_keys = ['PSNR_f', 'SSIM_f', 'MAE_f', 'HFEN_f']
    metric_titles = ['PSNR (dB)', 'SSIM', 'MAE', 'HFEN']

    plt.figure(figsize=(20, 3))
    for cnt, res in enumerate(results_sorted):
        label_str = (
            r'$\beta$:%.1e $\lambda$:%.1e' % (res[3]['beta'], res[3]['lmbda'])
            + ' l/ch/mapS/sc/tReg/sgdS/modR/D:\n%d/%d/%d/%d/%.1e/%d/%.1e/%s' % (
                res[3]['num_layers'], res[3]['num_channels'],
                res[3]['mapping_size'], res[3]['scale'],
                res[3]['reg_t_weight'], res[3]['sgd_size'],
                res[3]['weight_model'], res[3]['denoiser_model'])
        )
        linestyle = ['--', '-', '-.'][cnt % 3]
        for i, (metric, title) in enumerate(zip(metric_keys, metric_titles)):
            plt.subplot(1, 4, i + 1)
            plt.title(title)
            plt.plot(res[1][metric], label=label_str, linestyle=linestyle)
            plt.grid(linestyle='--', linewidth=0.5)

            save_np_path = (
                f'beta_{res[3]["beta"]:.2e}_lambda_{res[3]["lmbda"]:.2e}_'
                f'PSNR_{res[0]:.2e}_nlyr_{res[3]["num_layers"]}_nch_{res[3]["num_channels"]}_'
                f'enc_{pos_enc_type}_mapS_{res[3]["mapping_size"]}_sc_{res[3]["scale"]}_'
                f'init_{res[3]["init"]}_sgdS_{res[3]["sgd_size"]}_rep_{res[3]["rep"]}'
                f'_{res[3]["reg_mode"]}_{metric}.npy'
            )
            np.save(out_dir + save_np_path, res[1][metric])

        plt.legend(bbox_to_anchor=(0.75, -0.25), ncol=3)

    max_psnr = results_sorted[0][0]

    if save_fig:
        stem = (
            'max_PSNR_%.2e_acc_res_P_%d_num_epoch_%d'
            '_num_primal_iter_%d_noise_std_%.2e'
            '_lr_%.2e_num_exps_%d_rep_%d_%s' % (
                max_psnr, P, res[3]['num_epoch'], res[3]['num_primal_iter'],
                noise_std, res[3]['lr_primal'], len(results_sorted),
                res[3]['rep'], res[3]['reg_mode'])
        )
        plt.savefig(out_dir + stem + '.jpg', bbox_inches='tight')
        plt.savefig(out_dir + stem + '.pdf', bbox_inches='tight')
        with open(out_dir + stem + '.txt', 'w') as txt_file:
            for res in results_sorted:
                txt_file.write(
                    f"{res[0]} PSNR: {res[1]['PSNR_f'][-1]:.2e} "
                    f"SSIM: {res[1]['SSIM_f'][-1]:.2e} "
                    f"MAE: {res[1]['MAE_f'][-1]:.2e} "
                    f"HFEN: {res[1]['HFEN_f'][-1]:.2e} {res[3]}\n"
                )
    plt.show()

    # --- best-PSNR summary and array export ---
    print('Results with increasing PSNR:\n')
    for res in results_sorted:
        best_idx = np.argmax(np.array(res[1]['PSNR_f']))
        print(
            f"PSNR: {res[0]:.2e} SSIM: {res[1]['SSIM_f'][best_idx]:.2e} "
            f"MAE: {res[1]['MAE_f'][best_idx]:.2e} "
            f"HFEN: {res[1]['HFEN_f'][best_idx]:.2e}"
        )
        if save_fig:
            suffix = (
                f'beta_{res[3]["beta"]:.2e}_lambda_{res[3]["lmbda"]:.2e}_'
                f'PSNR_{res[0]:.2e}_nlyr_{res[3]["num_layers"]}_nch_{res[3]["num_channels"]}_'
                f'enc_{params["pos_enc_type_sweep"]}_mapS_{res[3]["mapping_size"]}_sc_{res[3]["scale"]}_'
                f'init_{res[3]["init"]}_sgdS_{res[3]["sgd_size"]}_rep_{res[3]["rep"]}'
                f'_{res[3]["reg_mode"]}.npy'
            )
            np.save(out_dir + 'f_est' + suffix, res[4]['f_est'])

    # --- gradient norm ---
    plt.figure(figsize=(6, 3))
    for cnt, res in enumerate(results_sorted):
        label_str = (
            r'$\beta$:%.1e $\lambda$:%.1e' % (res[3]['beta'], res[3]['lmbda'])
            + ' l/ch/mapS/sc/tReg/sgdS/modR: %d/%d/%d/%d/%.1e/%d/%.1e' % (
                res[3]['num_layers'], res[3]['num_channels'],
                res[3]['mapping_size'], res[3]['scale'],
                res[3]['reg_t_weight'], res[3]['sgd_size'], res[3]['weight_model'])
        )
        linestyle = ['--', '-', '-.'][cnt % 3]
        plt.subplot(1, 1, 1)
        plt.title(r'$\log_{10}\|\alpha\|_2$')
        plt.plot(
            np.log10(np.array(res[1]['grad_nf']) + 1e-10),
            label=label_str, linestyle=linestyle)
        plt.grid(linestyle='--', linewidth=0.5)
        plt.legend(bbox_to_anchor=(0.75, -0.25), ncol=3)
    plt.ylim(-3, 1)
    if save_fig:
        stem = (
            'grad_nf_P_%d_num_epoch_%d_num_primal_iter_%d'
            '_noise_std_%.2e_lr_%.2e_num_exps_%d_rep_%d' % (
                P, res[3]['num_epoch'], res[3]['num_primal_iter'],
                noise_std, res[3]['lr_primal'],
                len(results_sorted), res[3]['rep'])
        )
        plt.savefig(out_dir + stem + '.jpg', bbox_inches='tight')
        plt.savefig(out_dir + stem + '.pdf', bbox_inches='tight')
    plt.show()

    for res in results_sorted[-3:]:
        red_visualization(res[1], True)
