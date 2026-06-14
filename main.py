import argparse
import yaml
import itertools
from tqdm import tqdm

import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable

import utils_misc
import utils
import train
import metrics
import models
import plots


def main():
    np.set_printoptions(precision=2, suppress=True)

    args = parser.parse_args()
    print('Args:', args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(args.device)

    # FLAGS & HYPERPARMETERS
    P = args.P  # Total number of time instances = P
    ANG_PERIOD = args.ang_period  # Number of distinct views. If no period == None
    MOTION = args.motion  # motion types: ['piecewise_affine_transform', 'cardiac', 'LLNL']
    OBJ_TYPE = args.obj_type  # available objects: ['walnut', 'material', 'cardiac_rep_sq', 'LLNL', 'polymer_binary', 'polymer_binary_subint', 'polymer_binary_subint_hardest']
    SPATIAL_DIM = args.spatial_dim  # Spatial dim of the recon: d x d, and proj: d, default: 128, LLNL data: 80
    VIEW_ANG_SCH = args.view_ang_sch  # ['linear', 'random', 'bit_reversal', 'golden_angle']
    PI_SYMM = args.pi_symm  # Exploit projection pi-symm
    ang_range = np.pi if PI_SYMM else 2 * np.pi  # the range of view angles
    NOISE_STD = args.noise_std  # Measurement noise std, default 1e-2/2
    GEN_G_ALL = args.gen_g_all  # If True: Generate noisy full set of measurements, False: load existing data
    SAVE_NOISY_MEAS = args.save_noisy_meas  # If True: save noisy projs, False: load saved ones
    FWD_MODEL_PATH = args.fwd_model_path
    REP = args.rep  # Number of simultaneous views
    NUM_MEAS = REP * P  # Total number of projections for the full projection data
    PLOT = False  # Plot intermediate results and RED update figures

    # LOAD VARIABLES
    # Compute view angle scheme
    theta_exp = utils.generate_theta(REP * P, ang_range=ang_range, period=ANG_PERIOD)[VIEW_ANG_SCH]

    # Load phantom
    if OBJ_TYPE in ['cardiac_rep_sq']:
        f = utils_misc.load_f(OBJ_TYPE, MOTION, SPATIAL_DIM, 128)
        f = f[..., ::(128//(REP*P))]
    elif OBJ_TYPE in ['polymer_subint'] and P == 8:
        f = utils_misc.load_f(OBJ_TYPE, MOTION, SPATIAL_DIM, 128)
        f = f[..., 64-P//2:64+P//2]
    elif REP * P >= 32:
        f = np.repeat(utils_misc.load_f(OBJ_TYPE, MOTION, SPATIAL_DIM, P), REP, axis=-1)
    else:
        f = utils_misc.load_f(OBJ_TYPE, MOTION, SPATIAL_DIM, 32)[..., ::32//P]
    if OBJ_TYPE == 'polymer_subint':
        f /= 16.0
    f_cuda = torch.Tensor(f).cuda()
        
    # Compute subsampled measurements from the true object
    g = utils.obtain_projections(f, theta_exp, NUM_MEAS)
    g_pi_symm = utils.obtain_projections(
        f, theta_exp + np.pi, NUM_MEAS)
    g_symm_long = utils.construct_pi_symm_g(
        g, g_pi_symm, ang_range)

    # Compute/Load full set of measurements
    PATH = '/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/obj_domain_psm/data/'
    ADD_PATH = '_mean_corr' if OBJ_TYPE == 'material' else ''
    if GEN_G_ALL:
        f_pol_radon, f_true_recon_radon = utils.generate_f_pol(
            f, P, SPATIAL_DIM, NUM_MEAS, theta_exp, OBJ_TYPE, ANG_PERIOD, PATH,
            save=True, add_path=ADD_PATH, rep=REP)
    else:
        STR_PERIOD = '' if ANG_PERIOD is None else '_' + str(ANG_PERIOD)
        STR_REP = '' if REP == 1 else '_rep_%d' %REP
        f_pol_radon = np.load(PATH + 'f_pol_s_%s%s_%d%s%s.npy' %(
            OBJ_TYPE, ADD_PATH, P, STR_PERIOD, STR_REP))

    # Compute/Load measurements with AWGN with std=noise_std
    if SAVE_NOISY_MEAS:
        g_noisy, g_symm_long_noisy = utils.add_meas_noise(
            g, g_symm_long, NOISE_STD, f_pol_radon.max(), ang_range, OBJ_TYPE, P, 
            ANG_PERIOD, PATH, save=SAVE_NOISY_MEAS, add_path=ADD_PATH, rep=REP)    
    elif ang_range == np.pi:
        STR_PERIOD = '' if ANG_PERIOD is None else '_' + str(ANG_PERIOD)
        STR_REP = '' if REP == 1 else '_rep_%d' %REP
        g_symm_long_noisy = np.load(
            PATH + 'g_radon_symm_long_noisy_%s%s_%d%s_noise_std_%.2e%s.npy' %(
                OBJ_TYPE, ADD_PATH, P, STR_PERIOD, NOISE_STD, STR_REP))
    else:
        raise NotImplementedError

    # Load forward operator
    R, R_cuda = utils.load_radon_op(
        PI_SYMM, SPATIAL_DIM, REP * P, period=ANG_PERIOD, path=FWD_MODEL_PATH)
    
    # STATIC RECON FROM TIME-SEQUENTIAL MEASUREMENTS
    f_static_rec = utils.static_recon(g_symm_long_noisy[:SPATIAL_DIM], theta_exp)
    print('\nStatic recon PSNR (dB):', utils_misc.compute_psnr(f, f_static_rec[..., None]))

    # CONFIG
    with open(args.config, "r") as file:  # Use the path provided via --config
        params = yaml.load(file, Loader=yaml.FullLoader)
        
    params['rep'] = [REP]
    params['sgd_size'] = [REP*P//32, REP*P//16]  # REP*P//8
    
    print('\nTraining params:', params)
    
    results_per_config = []

    # TRAIN FOR ALL PARAMETER CONFIGURATIONS
    for [num_layers, num_channels, pos_enc_type, mapping_size, scale, lmbda, beta,
        reg_t_weight, init, t_freq_ratio, act_type, num_epoch, num_primal_iter,
        lr_primal, reg_t_loss_type, num_red_iter, sch_thr, sch_decay,
        weight_model, type_model_reg, sgd_size, noise_std, rep, reg_mode, 
        num_layers_denoiser, num_channels_denoiser, lambda_jr, final_act, denoiser_model] in list(
        itertools.product(
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
            params['final_act'], params['denoiser_model'])):
        print(
            [num_layers, num_channels, pos_enc_type, mapping_size, scale, lmbda, 
            beta, reg_t_weight, init, t_freq_ratio, act_type, num_epoch, 
            num_primal_iter, lr_primal, reg_t_loss_type, num_red_iter, sch_thr, 
            sch_decay, weight_model, type_model_reg, sgd_size, noise_std, rep, 
            reg_mode, num_layers_denoiser, num_channels_denoiser, final_act]
            )
        
        # FIX LATER !
        lmbda = beta * 1
        
        xyt_grid = Variable(
            torch.Tensor(utils_misc._create_yxt_grid((
            SPATIAL_DIM, SPATIAL_DIM, P)).transpose(3, 0, 1, 2)[None, ...]).cuda(), 
            requires_grad=False)
        
        xyt_grid_enc, input_ch, model_enc = utils.pos_enc(
            xyt_grid, pos_enc_type, SPATIAL_DIM, mapping_size, scale, t_freq_ratio)
        
        if init in ['none', 'static']:
            model = models.NeuralFieldModel3D_fc_res(
                input_ch=input_ch, output_dim=1, num_layers=num_layers, 
                num_channels=num_channels, spatial_dim=SPATIAL_DIM, mask=True, P=P,
                act_type=act_type, f_max=f.max(), final_act=final_act).cuda()
        elif init in ['oracle', 'red-psm', 'proj_nf']:
            model = utils.load_model(
                input_ch, num_layers, num_channels, P, OBJ_TYPE, pos_enc_type,
                mapping_size, scale, init, t_freq_ratio)
            f_nf_est = model(xyt_grid_enc)
            model.f_est = f_nf_est.squeeze().detach().clone()
        
        p = utils.get_params(model)
        p_count = utils.count_parameters(model)
        optimizer = torch.optim.AdamW(p, lr=lr_primal)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int((num_primal_iter * num_epoch) * sch_thr),
            gamma=sch_decay)
        criterion = nn.MSELoss(reduction='sum')
        
        if init == 'static':
            lr_primal_static = 1.0e-5
            num_epoch_static = 2500
            sgd_size_static = REP * P // 1
            optimizer_static = torch.optim.AdamW(p, lr=lr_primal_static)
            
            [metrics_static, best_psnr_f_static, best_f_static_est, f_nf_static_est,
            loss_epoch_static] = train.static_embedding(
                model, xyt_grid_enc, optimizer_static, criterion, f_static_rec, f, 
                P, rep, num_epoch_static, sgd_size_static, REP)
        
        # Load RED denoiser & patchifier if denoiser_type == patch-based
        if 'polymer_binary_subint' in OBJ_TYPE:
            denoiser_obj_type = 'polymer_binary'
        elif 'polymer_subint' in OBJ_TYPE:
            denoiser_obj_type = 'polymer'
        else:
            denoiser_obj_type = OBJ_TYPE
        # denoiser_obj_type = 'walnut_normalized'
        # denoiser_obj_type = 'polymer_binary_normalized'
        # denoiser_obj_type = 'wavelet'
        
        model_denoiser, patchifier = utils.denoising_network_loader(
            denoiser_model, denoiser_type=params['denoiser_type'], 
            pSize=params['pSize'], pStride=params['pStride'], 
            num_layers=num_layers_denoiser, spatial_dim=SPATIAL_DIM, 
            obj_type=denoiser_obj_type, num_channels=num_channels_denoiser, 
            noise_est_type=params['noise_est_type'], epochs=params['denoiser_epochs'], 
            noise_std=noise_std, lambda_jr=lambda_jr)
        
        g_gt = torch.Tensor(g_symm_long_noisy).cuda()
        
        # Initialize recon metrics per epoch
        metrics = {}
        metrics = {
            key: [] for key in [
            'PSNR_f', 'PSNR_nf', 'MAE_f', 'MAE_nf', 'SSIM_f', 'SSIM_nf', 
            'HFEN_f', 'HFEN_nf', 'grad_nf', 'f_red_in_norm_err', 
            'f_denoised_norm_err', 'f_red_out_norm_err', 'f_dual_norm_err']
        }
        best_psnr_f = 1e0
        best_f_est = None
        
        loss_epochs = []
        
        init_psnr = utils.compute_psnr(
            f[..., ::rep], model.f_est.squeeze().detach().cpu().numpy())
        print('Initial PSNR:%.2e dB' %init_psnr)
        
        Nepoch = tqdm(range(num_epoch), 
                    desc=('PSNR:%.3e/%.3e SSIM:%.2e/%.2e MAE:%.2e/%.2e '
                          'HFEN:%.2e/%.2e grad_nf:%.3e' %tuple(
                                [0 for _ in range(9)])),
                    leave=True, ncols=160, colour='green')
        for epoch in Nepoch:
            # ADMM primal PSM step
            [loss_epoch, f_nf_est, g_f_est] = train.learn_NF_SGD_multiview(
                model=model, xyt_grid_enc=xyt_grid_enc, optimizer=optimizer, scheduler=scheduler, 
                criterion=criterion, g_gt=g_gt, num_primal_iter=num_primal_iter, 
                P=P, spatial_dim=SPATIAL_DIM, R=R_cuda, beta=beta, rep=rep, 
                reg_t_weight=reg_t_weight, reg_t_loss_type=reg_t_loss_type, 
                weight_model=weight_model, type_model_reg=type_model_reg, 
                sgd_size=sgd_size)

            with torch.no_grad():
                # ADMM primal f-step
                if reg_mode == 'RED_reg' and (beta != 0 and lmbda != 0):
                    for _ in range(num_red_iter):
                        model.f_est, red_metrics = train.f_update(
                            lmbda, beta, model_denoiser, model.f_est.squeeze(), 
                            f_nf_est.detach().squeeze(), f_cuda[..., ::rep], 
                            model.gamma_est.squeeze(), params['denoiser_type'], model.mask)
                        for i, key in enumerate(['f_red_in_norm_err', 'f_denoised_norm_err', 'f_red_out_norm_err', 'f_dual_norm_err']):
                            metrics[key].append([red_metrics[i]])
                    # ADMM dual step
                    model.gamma_est = train.dual_variable_update(
                        model.gamma_est, model.f_est.squeeze(), f_nf_est.detach().squeeze())
                elif reg_mode == 'no_reg' or (beta == 0 and lmbda == 0):
                    model.f_est = f_nf_est.detach().squeeze()
                else:
                    raise NotImplementedError

                # Compute metrics and plot results
                if epoch % params['metric_upd_freq'] == 0:
                    if PLOT:
                        plots.train_visualization(
                            loss_epoch, model.f_est.detach().cpu(),
                            f_nf_est.squeeze().detach().cpu(),
                            model.gamma_est.detach().cpu(), f, P//8, 
                            epoch % params['res_plot_freq'] == 0)

                    # Update accuracy metrics
                    metrics, best_psnr_f, best_f_est = metrics.update_metrics(
                        f[..., ::rep], 
                        model.f_est.squeeze().detach().cpu().numpy(), 
                        f_nf_est.squeeze().detach().cpu().numpy(), 
                        metrics, rep, best_psnr_f, best_f_est, model)

                    Nepoch.set_description(
                        'PSNR:%.2e/%.2e SSIM:%.2e/%.2e MAE:%.2e/%.2e HFEN:%.2e/%.2e grad_nf:%3e' %(
                            metrics['PSNR_f'][-1], metrics['PSNR_nf'][-1], 
                            metrics['SSIM_f'][-1], metrics['SSIM_nf'][-1], 
                            metrics['MAE_f'][-1], metrics['MAE_nf'][-1], 
                            metrics['HFEN_f'][-1], metrics['HFEN_nf'][-1],
                            metrics['grad_nf'][-1]))

                    loss_epochs.append(loss_epoch)

        plots.red_visualization(metrics, True) if (reg_mode == 'RED_reg' and PLOT) else None

        # Save training parameters        
        train_params, train_data = {}, {}
        [train_params['beta'], train_params['lmbda'], train_params['pSize'], 
        train_params['pStride'], train_params['num_layers'], 
        train_params['num_channels'], train_params['scale'], 
        train_params['mapping_size'], train_params['pos_enc_type'], 
        train_params['p_count'], train_params['reg_t_weight'], train_params['init'], 
        train_params['num_epoch'], train_params['num_primal_iter'], 
        train_params['lr_primal'], train_params['num_red_iter'], 
        train_params['sch_thr'], train_params['sch_decay'], train_params['sgd_size'], 
        train_params['act_type'], train_params['noise_std'], train_params['rep'], 
        train_params['weight_model'], train_params['type_model_reg'], 
        train_params['denoiser_model'], train_params['reg_mode'],
        train_params['denoiser_epochs'], train_params['num_layers_denoiser'], 
        train_params['num_channels_denoiser'], train_params['lambda_jr'], 
        train_params['final_act']] = [
            beta, lmbda, params['pSize'], params['pStride'], num_layers, 
            num_channels, scale, mapping_size, pos_enc_type, p_count, reg_t_weight, 
            init, num_epoch, num_primal_iter, lr_primal, num_red_iter, sch_thr, 
            sch_decay, sgd_size, act_type, noise_std, rep, weight_model, 
            type_model_reg, denoiser_model, reg_mode, params['denoiser_epochs'], 
            num_layers_denoiser, num_channels_denoiser, lambda_jr, final_act]
        [train_data['f_est']] = [best_f_est]

        results_per_config.append(
            [max(metrics['PSNR_f']), metrics, loss_epoch,
            train_params, train_data, params['denoiser_type']])
        
        del model, model_denoiser, patchifier, optimizer, scheduler

    update_type = 'nf_admm_updates_sgd_resnet_%s_init_%s_scale_corr' %(reg_mode, init)
    plots.plot_and_save_red_nf_results(
        params, OBJ_TYPE, results_per_config, P, update_type, noise_std=NOISE_STD, 
        ang_period=ANG_PERIOD, pos_enc_type=pos_enc_type, mapping_size=mapping_size, 
        scale=scale, save_fig=True)


parser = argparse.ArgumentParser(description="Run the main script with a configuration file.")
parser.add_argument('--config', type=str, required=True, help="Path to the configuration file (YAML).")
parser.add_argument("--P", type=int, required=True)
parser.add_argument("--obj_type", type=str, required=True)
parser.add_argument("--noise_std", type=float, required=True)
parser.add_argument("--spatial_dim", type=int, required=False, default=128)
parser.add_argument("--ang_period", type=int, required=False, default=None)
parser.add_argument("--motion", type=str, required=False, default='piecewise_affine_transform')
parser.add_argument("--view_ang_sch", type=str, required=False, default='bit_reversal')
parser.add_argument("--pi_symm", type=bool, required=False, default=True)
parser.add_argument("--gen_g_all", type=bool, required=False, default=False)
parser.add_argument("--save_noisy_meas", type=bool, required=False, default=False)
parser.add_argument("--fwd_model_path", type=str, required=False, default='/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/obj_domain_psm/forward_model/')
parser.add_argument("--rep", type=int, required=False, default=1)
parser.add_argument("--device", type=int, required=False, default=0)


if __name__ == "__main__":
    main()