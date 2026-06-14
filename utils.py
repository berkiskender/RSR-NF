from pathlib import Path
import os
import itertools

import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import radon, iradon
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_wavelet, estimate_sigma
from scipy import ndimage

import torch
from torch import nn as nn

import torch_dct as dct
from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)

import astra
import tomosipo as ts
from ts_algorithms import fbp, sirt, tv_min2d, fdk, nag_ls

import red_psm_models
import models

import models_recon_domain

np.set_printoptions(precision=2)


def mask_fov_object(f, spatial_dim):
    """
    Applies the field-of-view mask for tomographic imaging to each frame of f.
    """
    for i,j in list(
        itertools.product(np.arange(spatial_dim), np.arange(spatial_dim))):
        if (i-spatial_dim//2)**2 + (j-spatial_dim//2)**2 >= (spatial_dim//2)**2:
            f[i, j, :] = 0
    return f


def DecimalToBinary(n):
    return bin(n).replace("0b", "")


def bit_reversal(x, N):
    num_digit = 0
    while N // 2:
        N = N // 2
        num_digit += 1
    x = list(DecimalToBinary(x))
    while len(x) < num_digit:
        x = [0] + x
    x.reverse()
    return int("".join(str(n) for n in x), 2)


def obtain_projections(f, theta, P):
    """
    Computes the dynamic/undersampled projections of the time-varying object.
    """
    g = np.zeros([f.shape[0], P])
    for p in range(P):
        g[:, p] = radon(f[:, :, p], theta=[360 * theta[p] / (2 * np.pi)])[:, 0]
    return g


def obtain_all_projections(f, theta, P):
    """
    Computes the full set of projections of the time-varying object.
    """
    g = np.zeros([f.shape[0], len(theta), P])
    for p in range(P):
        g[:, :, p] = radon(f[:, :, p], theta=360 * theta / (2 * np.pi))
    return g


def static_recon(g, theta):
    return iradon(g, theta=360*theta/(2*np.pi), filter_name='ramp')


def compute_psnr(x_gt, x_rec):
    mse = np.mean((x_rec - x_gt) ** 2)
    return 20 * np.log10((np.max(x_gt) - np.min(x_gt)) / np.sqrt(mse))


def compute_mae(x_gt, x_rec):
    return np.mean(np.abs(x_rec - x_gt))


def compute_ssim(x_gt, x_rec):
    return ssim(x_gt, x_rec, data_range=x_rec.max() - x_rec.min())


def compute_hfen(f, f_est):
    log_filter = lambda x, y: ndimage.gaussian_laplace(x - y, sigma=1.5)
    
    if len(f.shape) == 2:
        N = 1
        hf_f = np.zeros([f.shape[0], f.shape[1], 1])
        f = f[..., None]
        f_est = f_est[..., None]
    else:
        N = f.shape[-1]
        hf_f = np.zeros(f.shape)
        
    for t in range(N):
        hf_f[..., t] = log_filter(f[..., t], f_est[..., t])
    return np.linalg.norm(hf_f)


def generate_theta(P, ang_range=2 * np.pi, period=None):
    """
    Computes different view angle sampling schemes.
    """
    repeat_ang_sch = lambda x: np.tile(x, P//period)
    period = P if period is None else period
    theta_linear = repeat_ang_sch(np.linspace(0, ang_range, period, endpoint=False))
    theta_random = repeat_ang_sch(np.random.uniform(0, ang_range, size=[period]))
    theta_bit_reversal = repeat_ang_sch(np.array([(
        ang_range / period) * bit_reversal(p, period) for p in range(period)]))
    theta_golden_angle = repeat_ang_sch(np.array([((
        p * (111.25 / 360) * 2 * np.pi) % (2 * np.pi)) for p in range(period)]))
    return theta_linear, theta_random, theta_bit_reversal, theta_golden_angle


def construct_pi_symm_g(g_radon, g_radon_pi_symm, ang_range):
    """
    Returns measurements with pi-symmetric versions concatenated.
    """
    if ang_range == np.pi:
        # Concatenate pi-symm measurements as new rows
        g_radon_symm_long = np.zeros([2 * g_radon.shape[0], g_radon.shape[1]])
        g_radon_symm_long[:g_radon.shape[0], :] = g_radon
        g_radon_symm_long[g_radon.shape[0]:, :] = g_radon_pi_symm
    else:
        g_radon_symm_long = g_radon, g_radon
    return g_radon_symm_long


def generate_f_pol(f, P, spatial_dim, num_instances, theta_exp, obj_type,
                   period=None, path=None, save=True, add_path=None, rep=1):
    """
    Computes full set of projections and FBP reconstruction for each time frame 
    of the object.
    """
    f_pol_s = np.zeros([spatial_dim, rep * P, num_instances])
    f_true_recon = np.zeros(f.shape)
    for t in range(num_instances):
        f_pol_s[..., t] = radon(
            f[..., t * rep * P // num_instances], theta=360 * theta_exp / (2 * np.pi))
        f_true_recon[..., t] = iradon(f_pol_s[..., t * rep * P // num_instances],
                                       theta=360 * theta_exp / (2 * np.pi))
    if save:
        str_period = '' if period is None else '_' + str(period)
        str_rep = '' if rep == 1 else '_rep_%d' %rep
        np.save(path+'f_pol_s_%s%s_%d%s%s.npy' %(
            obj_type, add_path, P, str_period, str_rep), f_pol_s)
        np.save(path+'f_true_recon_%s%s_%d%s%s.npy' %(
            obj_type, add_path, P, str_period, str_rep), f_true_recon)
    return f_pol_s, f_true_recon


def add_meas_noise(g_radon, g_radon_pi_symm_long, noise_std, g_max, ang_range,
                   obj_type, P, period, path, save=False, add_path=None, rep=1):
    """ 
    Adds gaussian noise with fixed std to the time-sequential projections.
    """
    g_radon_noisy = g_radon + np.random.normal(
        loc=0.0, scale=noise_std * g_max, size=g_radon.shape)
    g_radon_symm_long_noisy = np.zeros(g_radon_pi_symm_long.shape)
    g_radon_symm_long_noisy[:g_radon.shape[0]] = g_radon_noisy
    g_radon_symm_long_noisy[g_radon.shape[0]:] = g_radon_pi_symm_long[
        g_radon.shape[0]:] + np.flipud(g_radon_noisy - g_radon)
    if save:
        if ang_range == np.pi:
            str_period = '' if period is None else '_' + str(period)
            str_rep = '' if rep == 1 else '_rep_%d' %rep
            np.save(
                path + 'g_radon_symm_long_noisy_%s%s_%d%s_noise_std_%.2e%s.npy' %(
                obj_type, add_path, P, str_period, noise_std, str_rep),
                g_radon_symm_long_noisy)
    return g_radon_noisy, g_radon_symm_long_noisy


class patchifier(torch.nn.Module):
    """
    Returns the spatially patchified version of the input object at each frame.
    """
    def __init__(self, patchSize, patchStride, spatial_dim, bs):
        super(patchifier, self).__init__()
        
        self.patchSize, self.patchStride, self.bs = patchSize, patchStride, bs
        self.fold_params = dict(
            kernel_size=[patchSize, patchSize], stride=patchStride)
        self.fold = nn.Fold(
            output_size=[spatial_dim, spatial_dim], **self.fold_params)
        self.unfold = nn.Unfold(**self.fold_params)
        
    def merge(self, patches):
        patches = patches.contiguous().view(
            self.bs, patches.shape[0]//self.bs,
            self.patchSize*self.patchSize).permute(0,2,1)
        x = self.fold(patches)
        return x/(self.patchSize**2/self.patchStride**2)
    
    def forward(self, x):
        patches = self.unfold(x).permute(0,2,1)
        patches = patches.contiguous().view(
            self.bs*patches.shape[1], 1, self.patchSize, self.patchSize)
        return patches 


def denoising_network_loader(
        train_type, denoiser_type, pSize, pStride, num_layers, spatial_dim, obj_type, 
        num_channels, noise_est_type='direct', epochs=500, noise_std=5e-2, lambda_jr=None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filterSize = 3
    print('Denoiser info: ', train_type, obj_type, noise_est_type, epochs, 
          num_layers, num_channels, '%.1e' %noise_std)
    
    if lambda_jr is not None:
        print('lambda_jr: %.1e' %lambda_jr)
    
    if train_type in ['unet', 'unet_deblur', 'unet_limited', 'dncnn', 'dncnn_oracle', 'dncnn_jreg', 'dncnn_deblur', 'dncnn_limited', 'dncnn_limited_deblur']:
        if denoiser_type == 'full_img':
            model_name = train_type
            print(model_name)
            if train_type == 'dncnn_jreg' and lambda_jr is not None:
                lambda_jr = '_lambdajr_%.1e' %lambda_jr
            else:
                lambda_jr = ''
            if ('polymer' in obj_type) or ('cardiac' in obj_type) or ('deblur' in train_type) or ('limited' in train_type) or ('limited_deblur' in train_type):
                model_path = os.path.join(
                    '/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/obj_domain_psm/data/denoiser',
                    model_name + '_model_%s_%s_epochs_%d_num_layers_%d_num_ch_%d_noise_std_%.1e%s.pt' %(
                    obj_type, noise_est_type, epochs, num_layers, num_channels, noise_std, lambda_jr))
            else:
                model_path = os.path.join(
                    '/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/obj_domain_psm/data/denoiser',
                    model_name + '_model_%s_%s_epochs_%d_num_layers_%d_num_ch_%d%s.pt' %(
                    obj_type, noise_est_type, epochs, num_layers, num_channels, lambda_jr))
            if train_type == 'unet':
                model_denoiser = red_psm_models.UNet(1, 1, 32).cuda()
            elif train_type in ['dncnn', 'dncnn_jreg', 'dncnn_oracle', 'dncnn_deblur', 'dncnn_limited', 'dncnn_limited_deblur']:
                model_denoiser = red_psm_models.dncnn(
                    num_layers, num_channels, filterSize, noise_est_type).cuda()
            else:
                raise NotImplementedError
            patchifier_red = None
        elif denoiser_type == 'full_img_plinear':
            model_name = 'plinear_dncnn'
            model_path = os.path.join(
                '/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/obj_domain_psm/data/denoiser',
                model_name + '_model_%s_%s_epochs_%d_num_layers_%d_num_ch_%d.pt' %(
                obj_type, noise_est_type, epochs, num_layers, num_channels))
            model_denoiser = red_psm_models.dncnn_plinear(
                num_layers, num_channels, filterSize, noise_est_type).cuda()
            patchifier_red = None
        elif denoiser_type == 'patch_based_patchloss':
            model_name = 'dncnn_patchbased_patchloss'
            model_path = os.path.join(
                '/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/obj_domain_psm/data/denoiser',
                model_name + '_model_%s_%s_num_layers_%d_patch_size_%d_patch_stride_%d_num_ch_%d_epochs_%d.pt' %(
                    obj_type, noise_est_type, num_layers, pSize, pStride, 
                    num_channels, epochs))
            model_denoiser = red_psm_models.dncnnPatchBased_patchLoss(
                num_layers, num_channels, filterSize, noise_est_type).cuda()
            patchifier_red = patchifier(pSize, pStride, spatial_dim, 1)
        else:
            raise NotImplementedError
        
        model_denoiser = torch.load(model_path)
        model_denoiser.eval()
        for k, v in model_denoiser.named_parameters():
            v.requires_grad = False
        model_denoiser = model_denoiser.to(device)
        number_parameters = sum(map(lambda x: x.numel(), model_denoiser.parameters()))
        print('%s %s Model path: {%s} Params number: %d'%(
            train_type, denoiser_type, model_path, number_parameters))
    
    elif train_type == 'wavelet':
        model_denoiser = lambda x: torch.Tensor(
            np.nan_to_num(denoise_wavelet(
                x.squeeze().detach().cpu().numpy(),
                channel_axis=-1, 
                method='BayesShrink', 
                mode='soft', 
                rescale_sigma=True)
        )).cuda()
        patchifier_red = None
        
    else:
        raise NotImplementedError
    
    return model_denoiser, patchifier_red


def load_radon_op(pi_symm, spatial_dim, P, period=None, path=None):
    '''
    Loads the differentiable forward operator (Radon transform).

    Parameters:
    -----------
    pi_symm (bool): Indicates if the Radon operator incorporates pi-symmetry.
    spatial_dim (int): The spatial dimension of f.
    P (int): The number of measurements.
    period (int, optional): The number of repeated unique view angles. 
        If not provided, it defaults to P.
    path (str, optional): Forward operator path.

    Output:
    -------
    R (numpy.ndarray): The Radon operator matrix.
    R_cuda (torch.Tensor): The Radon operator matrix as a CUDA tensor for GPU.
    '''
    if period == None:
        period = P
    if pi_symm:
        R = np.load(
            path + 'A_radon_spatial_dim_%d_P_%d_bit_reversal_pi_symm.npy' %(
                spatial_dim, period))
    else:
        R = np.load(
            path + 'A_radon_spatial_dim_%d_P_%d_bit_reversal.npy' %(
                spatial_dim, period))
    R = np.tile(np.array(R), (P//period, 1, 1))
    R_cuda = torch.cuda.FloatTensor(R)
    return R, R_cuda


def load_f(obj_type, motion, spatial_dim, P):
    if obj_type in ['walnut', 'hydro'] or 'cardiac' in obj_type:
        f = np.load(
            '/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/obj_domain_psm/data/true_objects/%s/f_%s_%s_spatial_dim_%d_P_%d.npy' %(
                obj_type, obj_type, motion, spatial_dim, P))
    elif obj_type in ['material']:
        f = np.load(
            '/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/obj_domain_psm/data/true_objects/%s/%d/f_materials.npy' %(
                obj_type, P)).transpose(1,2,0)[:, :, :P] / 255
        f[f < 0.3] = 0
    elif obj_type in ['LLNL_S03', 'LLNL']:
        f = np.load(
            '/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/obj_domain_psm/data/true_objects/LLNL/%d/f_S03_008.npy' %P)[:,:,:P]
    elif obj_type in ['LLNL_S12']:
        f = np.load(
            '/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/obj_domain_psm/data/true_objects/LLNL/%d/f_S12_001.npy' %P)[:,:,:P]
    elif 'pde' in obj_type:
        if P == 256:
            f = np.load(
                '/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/obj_domain_psm/data/true_objects/%s/f_%s_%s_spatial_dim_%d_P_%d.npy' %(
                    obj_type, obj_type, motion, spatial_dim, P))
        elif P == 128:
            f = np.load(
                '/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/obj_domain_psm/data/true_objects/%s/f_%s_%s_spatial_dim_%d_P_%d_2nd_half.npy' %(obj_type, obj_type, motion, spatial_dim, P))
    return f


def save_model(model, optimizer, scheduler, loss, obj_type, num_layers, num_channels,
               pos_enc_type, mapping_size, scale, P, epoch, embed_model,
               t_freq_ratio):
    PATH = 'data/oracle_init/model/'
    Path(PATH).mkdir(parents=True, exist_ok=True)
    PATH += '%s_nlyr_%d_nch_%d_enc_%s_mapS_%d_sc_%d_P_%d_%s_tfr_%.3e.pt' %(
        obj_type, num_layers, num_channels, pos_enc_type, mapping_size, scale, P,
        embed_model, t_freq_ratio)
    
    torch.save({'epoch': epoch, 'model': model, 'optimizer': optimizer, 
                'scheduler': scheduler, 'loss': loss}, PATH)
    pass


def save_gauss_enc_model(model_enc, obj_type, num_layers, num_channels,
                         pos_enc_type, mapping_size, scale, P, epoch, embed_model,
                         t_freq_ratio):
    PATH = 'data/oracle_init/model_enc/'
    Path(PATH).mkdir(parents=True, exist_ok=True)
    PATH += '%s_nlyr_%d_nch_%d_enc_%s_mapS_%d_sc_%d_P_%d_%s_tfr_%.3e.pt' %(
        obj_type, num_layers, num_channels, pos_enc_type, mapping_size, scale, P,
        embed_model, t_freq_ratio)
    
    torch.save({'model_state_dict': model_enc.state_dict()}, PATH)
    pass


# def load_model(input_ch, num_layers, num_channels, P, obj_type,
#                pos_enc_type, mapping_size, scale, embed_model, t_freq_ratio):
#     model = NF_2D_models.NeuralFieldModel3D_fc(input_ch=input_ch, output_dim=1,
#                                             num_layers=num_layers, 
#                                             num_channels=num_channels, P=P)
        
#     PATH = 'data/oracle_init/model/'
#     Path(PATH).mkdir(parents=True, exist_ok=True)
#     PATH += '%s_nlyr_%d_nch_%d_enc_%s_mapS_%d_sc_%d_P_%d_%s_tfr_%.3e.pt' %(
#         obj_type, num_layers, num_channels, pos_enc_type, mapping_size, scale,
#         P, embed_model, t_freq_ratio)
#     checkpoint = torch.load(PATH)
#     model = checkpoint['model']
#     optimizer = checkpoint['optimizer']

#     return model, optimizer



def load_model(input_ch, num_layers, num_channels, P, obj_type,
               pos_enc_type, mapping_size, scale, embed_model, t_freq_ratio):
    model = models.NeuralFieldModel3D_fc(input_ch=input_ch, output_dim=1,
                                            num_layers=num_layers, 
                                            num_channels=num_channels, P=P).cuda()
    PATH = 'data/oracle_init/model/'
    Path(PATH).mkdir(parents=True, exist_ok=True)
    PATH += '%s_nlyr_%d_nch_%d_enc_%s_mapS_%d_sc_%d_P_%d_%s_tfr_%.3e.pt' %(
        obj_type, num_layers, num_channels, pos_enc_type, mapping_size, scale,
        P, embed_model, t_freq_ratio)
    print('NF Model path:', PATH)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def load_model_enc(image_size, mapping_size, scale, obj_type, num_layers, 
                   num_channels, pos_enc_type, P, embed_model, t_freq_ratio):
    model_enc = models.GaussianFourierEncoding3D(
        input_dim=3, image_size=image_size, mapping_size=mapping_size, 
        scale=scale)
    
    PATH = 'data/oracle_init/model_enc/'
    Path(PATH).mkdir(parents=True, exist_ok=True)
    PATH += '%s_nlyr_%d_nch_%d_enc_%s_mapS_%d_sc_%d_P_%d_%s_tfr_%.3e.pt' %(
        obj_type, num_layers, num_channels, pos_enc_type, mapping_size, scale, 
        P, embed_model, t_freq_ratio)

    checkpoint = torch.load(PATH)
    model_enc.load_state_dict(checkpoint['model_state_dict'])
    return model_enc


def save_projection_model(model, optimizer, loss, obj_type, num_layers, num_channels, 
                          pos_enc_type, mapping_size, scale, P, epoch, embed_model, 
                          t_freq_ratio):
    PATH = 'data/oracle_init/model_projection/'
    Path(PATH).mkdir(parents=True, exist_ok=True)
    PATH += '%s_nlyr_%d_nch_%d_enc_%s_mapS_%d_sc_%d_P_%d_%s_tfr_%.3e.pt' %(
        obj_type, num_layers, num_channels, pos_enc_type, mapping_size, scale,
        P, embed_model, t_freq_ratio)
    torch.save({'epoch': epoch, 'model': model,
                'optimizer': optimizer, 'loss': loss}
               , PATH)
    pass


def load_projection_model(input_ch, num_layers, num_channels, P, obj_type, 
                          pos_enc_type, mapping_size, scale, embed_model,
                          t_freq_ratio, P_static):
    model = models.NF3DProjDomain(input_ch=input_ch, output_dim=1,
                                        num_layers=num_layers,
                                        num_channels=num_channels,
                                        spatial_dim=128, P=P, act_type='relu',
                                        P_static=P_static, init=embed_model)
    PATH = 'data/oracle_init/model_projection/'
    Path(PATH).mkdir(parents=True, exist_ok=True)
    PATH += '%s_nlyr_%d_nch_%d_enc_%s_mapS_%d_sc_%d_P_%d_%s_tfr_%.3e.pt' %(
        obj_type, num_layers, num_channels, pos_enc_type, mapping_size, scale,
        P, embed_model, t_freq_ratio)
    checkpoint = torch.load(PATH)
    model = checkpoint['model']
    return model


def denoising_network_loader_proj(train_type, denoiser_type, pSize, pStride, 
                                  num_layers, spatial_dim, obj_type, 
                                  num_channels, noise_est_type='direct', 
                                  epochs=500, P_static=128, noise_std_max=5e-2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filterSize = 3
    if train_type == 'dncnn':
        if denoiser_type == 'full_img':
            model_name = 'dncnn'
            model_path = os.path.join(
                '/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/'
                '2D_neural_fields/own_imp/data/denoiser',
                model_name + '_g_model_%s_%s_epochs_%d_P_static_%d_num_layers_%d_num_ch_%d_noise_std_max_%.2e.pt' %(
                    obj_type, noise_est_type, epochs, P_static, num_layers,
                    num_channels, noise_std_max))
            model_dncnn = red_psm_models.dncnn(
                num_layers, num_channels, filterSize, noise_est_type).cuda()
            patchifier_red = None
        else:
            raise NotImplementedError
    model_dncnn = torch.load(model_path)
    model_dncnn.eval()
    for k, v in model_dncnn.named_parameters():
        v.requires_grad = False
    model_dncnn = model_dncnn.to(device)
    number_parameters = sum(map(lambda x: x.numel(), model_dncnn.parameters()))
    print('DnCNN %s Model path: {%s} Params number: %d'%(
        denoiser_type, model_path, number_parameters))
    return model_dncnn, patchifier_red


def obtain_full_recon_cpu(g, theta, spatial_dim, P, filt='ramp', interp='cubic'):
    g = g.detach().cpu().numpy()
    
    f_rec = np.zeros([spatial_dim, spatial_dim, P])
    for t in range(P):
        f_rec[..., t] = iradon(g[..., t], theta=360*theta/(2*np.pi),
                               filter_name=filt, interpolation=interp)
    return f_rec


def tomo_op(spatial_dim, P, P_static, angles, pixel_size=1.0,
            rot_axis_pos = (0.0, 0.0, 0.0)):
    # Detector parameters:
    detector_shape=(spatial_dim, P)
    detector_position = (0, 0, 0)

    # Volume parameters:
    volume_shape = np.array([spatial_dim, spatial_dim, P])
    voxel_size = np.array([1.0, 1.0, 1.0])

    # Rotation parameters
    rot_axis_pos = rot_axis_pos

    # Geometries
    pg = ts.parallel_vec(
        shape=detector_shape,
        ray_dir=(0, 1, 0),
        det_pos=detector_position,
        det_v=(pixel_size, 0, 0),
        det_u=(0, 0, pixel_size),
    )

    vg0 = ts.volume(
        shape=volume_shape,
        pos=(0, 0, 0),
        size=volume_shape * voxel_size,
    )
    R = ts.rotate(pos=rot_axis_pos, axis=(0, 0, 1), angles=angles)
    vg = R * vg0.to_vec()

    A = ts.operator(vg, pg)
    return A, pg, vg


def tomo_recon(R, g, num_iter, l2_reg, mode):
    if mode == 'SIRT':
        f_rec = sirt(R, g, num_iterations=num_iter, min_constraint=0)
    elif mode == 'TV_min':
        f_rec = tv_min2d(R, g, 0.0001, num_iterations=num_iter, min_constraint=0)
    elif mode == 'nag_ls':
        f_rec = nag_ls(R, g, num_iterations=num_iter, min_constraint=0, l2_regularization=l2_reg)
    else:
        raise NotImplementedError
    return f_rec


def pos_enc(xyt_grid, pos_enc_type, spatial_dim, mapping_size, scale, t_freq_ratio):
    if pos_enc_type == 'sine':
        model_enc = models.SineEncoding(input_dim=3, image_size=SPATIAL_DIM).cuda()
        xyt_grid_enc = model_enc(xyt_grid)
        input_ch = 6
    elif pos_enc_type == 'gaussian':
        if init == 'none':
            model_enc = models.GaussianFourierEncoding3D(
                input_dim=3, image_size=SPATIAL_DIM, mapping_size=mapping_size,
                scale=scale).cuda()
        xyt_grid_enc = model_enc(xyt_grid)
        input_ch = 2 * mapping_size
    elif pos_enc_type == 'fourier':
        model_enc = models.FourierEncoding2D_t(
            input_dim=3, mapping_size=mapping_size, t_freq_ratio=t_freq_ratio).cuda()
        xyt_grid_enc = model_enc(xyt_grid)
        input_ch = 6 * mapping_size    
    elif pos_enc_type == 'none':
        xyt_grid_enc = xyt_grid
        input_ch = 3
    return xyt_grid_enc, input_ch, model_enc


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def compute_grad_norm(model):
    total_sq_norm = 0
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    for p in model_parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_sq_norm += param_norm.item() ** 2
    return total_sq_norm ** 0.5


def get_params(net, downsampler=None):
    '''Returns parameters that we want to optimize over.
        Args:
            net: network
    '''
    
    params = []
    params += [x for x in net.parameters()]
    return params
