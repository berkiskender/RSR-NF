# RSR-NF: Neural Field Regularization by Static Restoration Priors for Dynamic Imaging
#
# Authors: Berk Iskender, Sushan Nakarmi, Nitin Daphalapurkar, Marc L. Klasky, Yoram Bresler
# Year:    2025
# Paper:   https://arxiv.org/abs/2503.10015
#
# Copyright (C) 2025 Berk Iskender. All rights reserved.
#
# This source code accompanies the paper above. If you use this code or build upon
# this work, please cite:
#
#   B. Iskender, S. Nakarmi, N. Daphalapurkar, M. L. Klasky, Y. Bresler,
#   "RSR-NF: Neural Field Regularization by Static Restoration Priors for Dynamic Imaging,"
#   arXiv:2503.10015, 2025.

import argparse
import yaml
import itertools
from dataclasses import dataclass

import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm

import utils_misc
import utils
import train
import metrics as metrics_module
import models
import plots


@dataclass
class HParams:
    num_layers: int
    num_channels: int
    pos_enc_type: str
    mapping_size: int
    scale: float
    lmbda: float
    beta: float
    reg_t_weight: float
    init: str
    t_freq_ratio: float
    act_type: str
    num_epoch: int
    num_primal_iter: int
    lr_primal: float
    reg_t_loss_type: str
    num_red_iter: int
    sch_thr: float
    sch_decay: float
    weight_model: float
    type_model_reg: str
    sgd_size: int
    noise_std: float
    rep: int
    reg_mode: str
    num_layers_denoiser: int
    num_channels_denoiser: int
    lambda_jr: float
    final_act: str
    denoiser_model: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the main script with a configuration file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--noise_std", type=float, required=True)
    parser.add_argument("--ang_period", type=int, default=None)
    parser.add_argument("--motion", type=str, default='piecewise_affine_transform')
    parser.add_argument("--view_ang_sch", type=str, default='bit_reversal')
    parser.add_argument("--pi_symm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gen_g_all", action='store_true', default=False)
    parser.add_argument("--save_noisy_meas", action='store_true', default=False)
    parser.add_argument("--fwd_model_path", type=str,
                        default='/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/obj_domain_psm/forward_model/')
    parser.add_argument("--data_path", type=str,
                        default='/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/obj_domain_psm/data/')
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main():
    np.set_printoptions(precision=2, suppress=True)

    args = parse_args()
    print('Args:', args)

    with open(args.config) as file:
        params = yaml.safe_load(file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_device(args.device)

    num_frames = params['P']
    ang_period = args.ang_period
    motion = args.motion
    obj_type = params['obj_type']
    spatial_dim = params['spatial_dim']
    view_ang_sch = args.view_ang_sch
    pi_symm = args.pi_symm
    ang_range = np.pi if pi_symm else 2 * np.pi
    noise_std_cli = args.noise_std
    gen_g_all = args.gen_g_all
    save_noisy_meas = args.save_noisy_meas
    fwd_model_path = args.fwd_model_path
    data_path = args.data_path
    num_rep = args.rep
    num_meas = num_rep * num_frames
    plot = False

    # Compute view angle scheme
    theta_exp = utils.generate_theta(
        num_meas, ang_range=ang_range, period=ang_period)[view_ang_sch]

    # Load phantom
    if obj_type in ['cardiac_rep_sq']:
        f = utils_misc.load_f(obj_type, motion, spatial_dim, 128)
        f = f[..., ::(128 // num_meas)]
    elif obj_type in ['polymer_subint'] and num_frames == 8:
        f = utils_misc.load_f(obj_type, motion, spatial_dim, 128)
        f = f[..., 64 - num_frames // 2:64 + num_frames // 2]
    elif num_meas >= 32:
        f = np.repeat(utils_misc.load_f(obj_type, motion, spatial_dim, num_frames), num_rep, axis=-1)
    else:
        f = utils_misc.load_f(obj_type, motion, spatial_dim, 32)[..., ::32 // num_frames]
    if obj_type == 'polymer_subint':
        f /= 16.0
    f_cuda = torch.tensor(f, dtype=torch.float32, device=device)

    # Compute subsampled measurements from the true object
    g = utils.obtain_projections(f, theta_exp, num_meas)
    g_pi_symm = utils.obtain_projections(f, theta_exp + np.pi, num_meas)
    g_symm_long = utils.construct_pi_symm_g(g, g_pi_symm, ang_range)

    # Compute/Load full set of measurements
    add_path = '_mean_corr' if obj_type == 'material' else ''
    if gen_g_all:
        f_pol_radon, _ = utils.generate_f_pol(
            f, num_frames, spatial_dim, num_meas, theta_exp, obj_type, ang_period,
            data_path, save=True, add_path=add_path, rep=num_rep)
    else:
        str_period = '' if ang_period is None else '_' + str(ang_period)
        str_rep = '' if num_rep == 1 else '_rep_%d' % num_rep
        f_pol_radon = np.load(
            data_path + 'f_pol_s_%s%s_%d%s%s.npy' % (obj_type, add_path, num_frames, str_period, str_rep))

    # Compute/Load measurements with AWGN
    if save_noisy_meas:
        _, g_symm_long_noisy = utils.add_meas_noise(
            g, g_symm_long, noise_std_cli, f_pol_radon.max(), ang_range, obj_type,
            num_frames, ang_period, data_path, save=save_noisy_meas, add_path=add_path, rep=num_rep)
    elif ang_range == np.pi:
        str_period = '' if ang_period is None else '_' + str(ang_period)
        str_rep = '' if num_rep == 1 else '_rep_%d' % num_rep
        g_symm_long_noisy = np.load(
            data_path + 'g_radon_symm_long_noisy_%s%s_%d%s_noise_std_%.2e%s.npy' % (
                obj_type, add_path, num_frames, str_period, noise_std_cli, str_rep))
    else:
        raise NotImplementedError

    # Load forward operator
    R, R_cuda = utils.load_radon_op(
        pi_symm, spatial_dim, num_meas, period=ang_period, path=fwd_model_path)

    # Static recon from time-sequential measurements
    f_static_rec = utils.static_recon(g_symm_long_noisy[:spatial_dim], theta_exp)
    print('\nStatic recon PSNR (dB):', utils_misc.compute_psnr(f, f_static_rec[..., None]))

    params['rep'] = [num_rep]
    params['sgd_size'] = [num_meas // 32, num_meas // 16]

    print('\nTraining params:', params)

    results_per_config = []

    for combo in itertools.product(
        params['num_layers_sweep'], params['num_channels_sweep'],
        params['pos_enc_type_sweep'], params['mapping_size_sweep'],
        params['scale_sweep'], params['lmbda_sweep'],
        params['beta_sweep'], params['reg_t_weight_sweep'],
        params['init'], params['t_freq_ratio_sweep'],
        params['act_type'], params['num_epoch_sweep'],
        params['num_primal_iter_sweep'], params['lr_primal_sweep'],
        params['reg_t_loss_type'], params['num_red_iter'],
        params['sch_thr_step_ratio'], params['sch_decay_ratio'],
        params['model_reg_weight'], params['type_model_reg'],
        params['sgd_size'], params['noise_std'], params['rep'],
        params['reg_mode'], params['num_layers_denoiser'],
        params['num_channels_denoiser'], params['lambda_jr'],
        params['final_act'], params['denoiser_model'],
    ):
        hp = HParams(*combo)
        print(hp)

        # FIX LATER
        hp.lmbda = hp.beta

        xyt_grid = torch.tensor(
            utils_misc._create_yxt_grid(
                (spatial_dim, spatial_dim, num_frames)).transpose(3, 0, 1, 2)[None, ...],
            dtype=torch.float32, device=device, requires_grad=False)

        xyt_grid_enc, input_ch, model_enc = utils.pos_enc(
            xyt_grid, hp.pos_enc_type, spatial_dim, hp.mapping_size, hp.scale, hp.t_freq_ratio)

        if hp.init in ['none', 'static']:
            model = models.NeuralFieldModel3D_fc_res(
                input_ch=input_ch, output_dim=1, num_layers=hp.num_layers,
                num_channels=hp.num_channels, spatial_dim=spatial_dim, mask=True, P=num_frames,
                act_type=hp.act_type, f_max=f.max(), final_act=hp.final_act).cuda()
        elif hp.init in ['oracle', 'red-psm', 'proj_nf']:
            model = utils.load_model(
                input_ch, hp.num_layers, hp.num_channels, num_frames, obj_type,
                hp.pos_enc_type, hp.mapping_size, hp.scale, hp.init, hp.t_freq_ratio)
            f_nf_est = model(xyt_grid_enc)
            model.f_est = f_nf_est.squeeze().detach().clone()

        model_params = utils.get_params(model)
        p_count = utils.count_parameters(model)
        optimizer = torch.optim.AdamW(model_params, lr=hp.lr_primal)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int((hp.num_primal_iter * hp.num_epoch) * hp.sch_thr),
            gamma=hp.sch_decay)
        criterion = nn.MSELoss(reduction='sum')

        if hp.init == 'static':
            lr_primal_static = 1.0e-5
            num_epoch_static = 2500
            sgd_size_static = num_meas
            optimizer_static = torch.optim.AdamW(model_params, lr=lr_primal_static)

            [metrics_static, best_psnr_f_static, best_f_static_est, f_nf_static_est,
             loss_epoch_static] = train.static_embedding(
                model, xyt_grid_enc, optimizer_static, criterion, f_static_rec, f,
                num_frames, hp.rep, num_epoch_static, sgd_size_static, num_rep)

        if 'polymer_binary_subint' in obj_type:
            denoiser_obj_type = 'polymer_binary'
        elif 'polymer_subint' in obj_type:
            denoiser_obj_type = 'polymer'
        else:
            denoiser_obj_type = obj_type

        model_denoiser, patchifier = utils.denoising_network_loader(
            hp.denoiser_model, denoiser_type=params['denoiser_type'],
            pSize=params['pSize'], pStride=params['pStride'],
            num_layers=hp.num_layers_denoiser, spatial_dim=spatial_dim,
            obj_type=denoiser_obj_type, num_channels=hp.num_channels_denoiser,
            noise_est_type=params['noise_est_type'], epochs=params['denoiser_epochs'],
            noise_std=hp.noise_std, lambda_jr=hp.lambda_jr)

        g_gt = torch.tensor(g_symm_long_noisy, dtype=torch.float32, device=device)

        metrics_dict = {
            key: [] for key in [
                'PSNR_f', 'PSNR_nf', 'MAE_f', 'MAE_nf', 'SSIM_f', 'SSIM_nf',
                'HFEN_f', 'HFEN_nf', 'grad_nf', 'f_red_in_norm_err',
                'f_denoised_norm_err', 'f_red_out_norm_err', 'f_dual_norm_err',
            ]
        }
        best_psnr_f = 1e0
        best_f_est = None
        loss_epochs = []

        init_psnr = utils.compute_psnr(
            f[..., ::hp.rep], model.f_est.squeeze().detach().cpu().numpy())
        print('Initial PSNR:%.2e dB' % init_psnr)

        progress = tqdm(
            range(hp.num_epoch),
            desc=('PSNR:%.3e/%.3e SSIM:%.2e/%.2e MAE:%.2e/%.2e HFEN:%.2e/%.2e grad_nf:%.3e' %
                  tuple([0 for _ in range(9)])),
            leave=True, ncols=160, colour='green')

        for epoch in progress:
            [loss_epoch, f_nf_est, g_f_est] = train.learn_NF_SGD_multiview(
                model=model, xyt_grid_enc=xyt_grid_enc, optimizer=optimizer,
                scheduler=scheduler, criterion=criterion, g_gt=g_gt,
                num_primal_iter=hp.num_primal_iter, P=num_frames,
                spatial_dim=spatial_dim, R=R_cuda, beta=hp.beta, rep=hp.rep,
                reg_t_weight=hp.reg_t_weight, reg_t_loss_type=hp.reg_t_loss_type,
                weight_model=hp.weight_model, type_model_reg=hp.type_model_reg,
                sgd_size=hp.sgd_size)

            with torch.no_grad():
                if hp.reg_mode == 'RED_reg' and (hp.beta != 0 and hp.lmbda != 0):
                    for _ in range(hp.num_red_iter):
                        model.f_est, red_metrics = train.f_update(
                            hp.lmbda, hp.beta, model_denoiser, model.f_est.squeeze(),
                            f_nf_est.detach().squeeze(), f_cuda[..., ::hp.rep],
                            model.gamma_est.squeeze(), params['denoiser_type'], model.mask)
                        for i, key in enumerate(['f_red_in_norm_err', 'f_denoised_norm_err',
                                                 'f_red_out_norm_err', 'f_dual_norm_err']):
                            metrics_dict[key].append([red_metrics[i]])
                    model.gamma_est = train.dual_variable_update(
                        model.gamma_est, model.f_est.squeeze(), f_nf_est.detach().squeeze())
                elif hp.reg_mode == 'no_reg' or (hp.beta == 0 and hp.lmbda == 0):
                    model.f_est = f_nf_est.detach().squeeze()
                else:
                    raise NotImplementedError

                if epoch % params['metric_upd_freq'] == 0:
                    if plot:
                        plots.train_visualization(
                            loss_epoch, model.f_est.detach().cpu(),
                            f_nf_est.squeeze().detach().cpu(),
                            model.gamma_est.detach().cpu(), f, num_frames // 8,
                            epoch % params['res_plot_freq'] == 0)

                    metrics_dict, best_psnr_f, best_f_est = metrics_module.update_metrics(
                        f[..., ::hp.rep],
                        model.f_est.squeeze().detach().cpu().numpy(),
                        f_nf_est.squeeze().detach().cpu().numpy(),
                        metrics_dict, hp.rep, best_psnr_f, best_f_est, model)

                    progress.set_description(
                        'PSNR:%.2e/%.2e SSIM:%.2e/%.2e MAE:%.2e/%.2e HFEN:%.2e/%.2e grad_nf:%3e' % (
                            metrics_dict['PSNR_f'][-1], metrics_dict['PSNR_nf'][-1],
                            metrics_dict['SSIM_f'][-1], metrics_dict['SSIM_nf'][-1],
                            metrics_dict['MAE_f'][-1], metrics_dict['MAE_nf'][-1],
                            metrics_dict['HFEN_f'][-1], metrics_dict['HFEN_nf'][-1],
                            metrics_dict['grad_nf'][-1]))

                    loss_epochs.append(loss_epoch)

        if hp.reg_mode == 'RED_reg' and plot:
            plots.red_visualization(metrics_dict, True)

        train_params = {
            'beta': hp.beta,
            'lmbda': hp.lmbda,
            'pSize': params['pSize'],
            'pStride': params['pStride'],
            'num_layers': hp.num_layers,
            'num_channels': hp.num_channels,
            'scale': hp.scale,
            'mapping_size': hp.mapping_size,
            'pos_enc_type': hp.pos_enc_type,
            'p_count': p_count,
            'reg_t_weight': hp.reg_t_weight,
            'init': hp.init,
            'num_epoch': hp.num_epoch,
            'num_primal_iter': hp.num_primal_iter,
            'lr_primal': hp.lr_primal,
            'num_red_iter': hp.num_red_iter,
            'sch_thr': hp.sch_thr,
            'sch_decay': hp.sch_decay,
            'sgd_size': hp.sgd_size,
            'act_type': hp.act_type,
            'noise_std': hp.noise_std,
            'rep': hp.rep,
            'weight_model': hp.weight_model,
            'type_model_reg': hp.type_model_reg,
            'denoiser_model': hp.denoiser_model,
            'reg_mode': hp.reg_mode,
            'denoiser_epochs': params['denoiser_epochs'],
            'num_layers_denoiser': hp.num_layers_denoiser,
            'num_channels_denoiser': hp.num_channels_denoiser,
            'lambda_jr': hp.lambda_jr,
            'final_act': hp.final_act,
        }
        train_data = {'f_est': best_f_est}

        results_per_config.append([
            max(metrics_dict['PSNR_f']), metrics_dict, loss_epoch,
            train_params, train_data, params['denoiser_type'],
        ])

        del model, model_denoiser, patchifier, optimizer, scheduler

    update_type = 'nf_admm_updates_sgd_resnet_%s_init_%s_scale_corr' % (hp.reg_mode, hp.init)
    plots.plot_and_save_red_nf_results(
        params, obj_type, results_per_config, num_frames, update_type,
        noise_std=noise_std_cli, ang_period=ang_period,
        pos_enc_type=hp.pos_enc_type, mapping_size=hp.mapping_size,
        scale=hp.scale, save_fig=True)


if __name__ == "__main__":
    main()
