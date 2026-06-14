import itertools

import numpy as np
from skimage.metrics import structural_similarity as ssim

import torch
from torch import nn as nn

import torch_dct as dct
from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)

from ts_algorithms import fbp, sirt, tv_min2d, fdk, nag_ls

import red_psm_models
import models


def total_variation_loss(x, weight):
    h_x, w_x, P = x.size()
    tv_h = torch.abs(x[1:,:,:]-x[:-1,:,:]).sum()
    tv_w = torch.abs(x[:,1:,:]-x[:,:-1,:]).sum()
    return weight*(tv_h+tv_w)/(h_x*w_x*P)


def total_variation_temp_loss(x, weight):
    h_x, w_x, P = x.size()
    tv_t = torch.abs(x[:, :, 1:]-x[:, :, :-1]).sum()
    return weight*(tv_t)/(h_x*w_x*P)


def l2_temp_loss(x, weight):
    h_x, w_x, P = x.size()
    loss_t = (torch.abs(x[:, :, :1]-x[:, :, :-1])**2).sum()
    return weight*(loss_t)/(h_x*w_x*P)


def l2_temp_group_loss(x, weight):
    h_x, w_x, P = x.size()
    loss_t = torch.norm(x[:, :, :1]-x[:, :, :-1], p='fro')
    return weight*(loss_t)/(h_x*w_x*P)


def l2_loss(x, weight):
    h_x, w_x, P = x.size()
    tv_h = (torch.abs(x[1:,:,:]-x[:-1,:,:])**2).sum()
    tv_w = (torch.abs(x[:,1:,:]-x[:,:-1,:])**2).sum()
    return weight*(tv_h+tv_w)/(h_x*w_x*P)


def sec_ord_temp_loss_l1(x, weight):
    h_x, w_x, P = x.size()
    loss_t = torch.abs(x[:, :, 2:]-2*x[:, :, 1:-1]+x[:, :, :-2]).sum()
    return weight*(loss_t)/(h_x*w_x*P)


# def sec_ord_temp_loss_l2(x, weight):
#     h_x, w_x, P = x.size()
#     loss_t = (torch.abs(x[:, :, 2:]-2*x[:, :, 1:-1]+x[:, :, :-2])**2).sum()
#     return weight * loss_t/(h_x * w_x * P)


def sec_ord_temp_loss_l2(x, weight):
    h_x, w_x, P = x.size()
    loss_t = (torch.abs(x[:, :, 2:]-2*x[:, :, 1:-1]+x[:, :, :-2])**2).sum()
    return weight * loss_t


def sec_ord_temp_group_loss_l2(x, weight):
    h_x, w_x, P = x.size()
    loss_t = torch.norm(x[:, :, 2:]-2*x[:, :, 1:-1]+x[:, :, :-2], p='fro')
    return weight*(loss_t)/(h_x*w_x*P)


def dct_l1_loss(x, P_DCT_U, weight):
    h_x, w_x, P = x.size()
    # coeff = P_DCT_U @ x.view(h_x * w_x, P).T
    # return weight * torch.abs(coeff).sum() / (h_x*w_x*P)
    return weight * torch.abs(dct.dct(x)).sum() / (h_x*w_x*P)


def wt_l1_loss(x, xfm, weight):
    h_x, w_x, P = x.size()
    X = x.view(h_x * w_x, P)[:, None, None, :]
    Yl, Yh = xfm(X)
    return weight * (torch.abs(Yl).sum() + torch.abs(Yh[0]).sum() + torch.abs(Yh[1]).sum() + torch.abs(Yh[2]).sum()) / (h_x*w_x*P)


def reg_loss(x, loss_type, weight):
    if loss_type == 'TV':
        return total_variation_loss(x, weight)
    elif loss_type == 'TV_temp_loss':
        return total_variation_temp_loss(x, weight)
    elif loss_type == 'l2_temp_loss':
        return l2_temp_loss(x, weight)
    elif loss_type == 'l2_temp_group_loss':
        return l2_temp_group_loss(x, weight)
    elif loss_type == 'l2_loss':
        return l2_loss(x, weight)
    elif loss_type == 'l1_sec_ord_temp_loss':
        return sec_ord_temp_loss_l1(x, weight)
    elif loss_type == 'l2_sec_ord_temp_loss':
        return sec_ord_temp_loss_l2(x, weight)
    elif loss_type == 'l2_sec_ord_temp_group_loss':
        return sec_ord_temp_group_loss_l2(x, weight)
    # elif loss_type == 'l1_dct_temp_loss':
    #     return dct_l1_loss(x, P_DCT_U, weight)
    # elif loss_type == 'l1_wt_temp_loss':
    #     return wt_l1_loss(x, xfm, weight)
    else:
        return NotImplementedError()
    

def model_reg(module, weight, type):
    params = torch.cat([x.view(-1) for x in module.parameters()])
    
    if type == 'l1':
        reg = weight * torch.norm(params, 1)
    elif type == 'l2':
        reg = weight * torch.norm(params, 2)
    else:
        raise NotImplementedError
    
    return reg