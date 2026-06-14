import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import imageio
from PIL import Image
import itertools

# Shepp-Logan
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale


def load_shepp_logan(size=256):
    return rescale(shepp_logan_phantom(), scale=size/400, mode='reflect',
                   channel_axis=None)[..., None]


def _create_yx_grid(grid_size):
    """
    Creates mesh grid of normalised pixel coordinates based on matrix indexing convention
    """
    h, w = grid_size
    coords_i = np.linspace(0, 1, h, endpoint=False)
    coords_j = np.linspace(0, 1, w, endpoint=False)
    grid = np.stack(np.meshgrid(coords_i, coords_j, indexing='ij'), axis=-1).astype('float32')
    return grid


def _create_yxt_grid(grid_size):
    """
    Creates mesh grid of normalised pixel coordinates based on matrix indexing convention
    """
    h, w, P = grid_size
    coords_i = np.linspace(0, 1, h, endpoint=False)
    coords_j = np.linspace(0, 1, w, endpoint=False)
    coords_t = np.linspace(0, 1, P, endpoint=False)
    grid = np.stack(np.meshgrid(coords_i, coords_j, coords_t, indexing='ij'), axis=-1).astype('float32')
    return grid


def get_params(net, downsampler=None):
    '''Returns parameters that we want to optimize over.
        Args:
            net: network
    '''
    
    params = []
    params += [x for x in net.parameters()]
    return params


def compute_psnr(x_gt, x_rec):
    mse = np.mean((x_rec - x_gt) ** 2)
    return 20 * np.log10((np.max(x_gt) - np.min(x_gt)) / np.sqrt(mse))


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def compute_grad_norm(model):
    total_sq_norm = 0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_sq_norm += param_norm.item() ** 2
    return total_sq_norm ** 0.5


def load_f(obj_type, motion, spatial_dim, P):
    if obj_type in ['walnut', 'hydro'] or 'cardiac' in obj_type:
        if P >= 32:
            f = np.load(
                '/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/obj_domain_psm/data/true_objects/%s/f_%s_%s_spatial_dim_%d_P_%d.npy' %(obj_type, obj_type, motion, spatial_dim, P))
        else:
            f = np.load(
                '/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/obj_domain_psm/data/true_objects/%s/f_%s_%s_spatial_dim_%d_P_%d.npy' %(obj_type, obj_type, motion, spatial_dim, 32))[..., ::32//P]
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
                '/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/obj_domain_psm/data/true_objects/%s/f_%s_%s_spatial_dim_%d_P_%d_2nd_half.npy' %(
                    obj_type, obj_type, motion, spatial_dim, P))
    elif obj_type == 'polymer_binary':
        f = np.load('/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/'
                    'nitin_files/phantoms/48_binary_/obj_%d_full.npy' %P)
    elif obj_type == 'polymer_binary_subint':
        if P >= 32:
            f = np.load('/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/'
                        'nitin_files/phantoms/48_binary_/obj_int_len_%d_int_no_1.npy' %P)
        else:
            f = np.load('/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/'
                        'nitin_files/phantoms/48_binary_/obj_int_len_%d_int_no_1.npy' %32)[..., ::32//P]
    elif obj_type == 'polymer_binary_subint_hardest':
        if P >= 32:
            f = np.load('/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/'
                        'nitin_files/phantoms/48_binary_/obj_int_len_%d_int_no_5.npy' %P)
        else:
            f = np.load('/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/'
                        'nitin_files/phantoms/48_binary_/obj_int_len_%d_int_no_5.npy' %128)[..., ::128//P]
    elif obj_type == 'polymer_subint':
        if P >= 32:
            f = np.load('/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/'
                        'nitin_files/phantoms/48_/obj_int_len_%d_int_no_1.npy' %P)
        else:
            f = np.load('/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/'
                        'nitin_files/phantoms/48_/obj_int_len_%d_int_no_1.npy' %32)[..., ::32//P]
    elif obj_type == 'polymer_subint_hardest':
        if P >= 32:
            f = np.load('/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/'
                        'nitin_files/phantoms/48_/obj_int_len_%d_int_no_5.npy' %P)
        else:
            f = np.load('/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/'
                        'nitin_files/phantoms/48_/obj_int_len_%d_int_no_5.npy' %128)[..., ::128//P]        
    return f


def total_variation_loss(img, weight):
    P, h_img, w_img = img.size()
    tv_h = torch.abs(img[:,1:,:]-img[:,:-1,:]).sum()
    tv_w = torch.abs(img[:,:,1:]-img[:,:,:-1]).sum()
    return weight*(tv_h+tv_w)/(h_img*w_img*P)


def total_variation_spatiotemp_loss(img, weight_spatial, weight_temp):
    P, h_img, w_img = img.size()
    tv_h = torch.abs(img[:, 1:, :]-img[:, :-1, :]).sum()
    tv_w = torch.abs(img[:, :, 1:]-img[:, :, :-1]).sum()
    tv_t = torch.abs(img[1:, :, :]-img[:-1, :, :]).sum()
    return (weight_spatial*(tv_h + tv_w) + weight_temp*tv_t)/(h_img * w_img * P)


def total_variation_temp_loss(img, weight):
    P, h_img, w_img = img.size()
    tv_t = torch.abs(img[1:, :, :]-img[:-1, :, :]).sum()
    return weight*(tv_t)/(h_img*w_img*P)