import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

import utils
import loss
import metrics


def f_update(lmbda, beta, denoiser, f_est, f_nf_est, f, gamma, denoiser_type, mask):
    f_est_permuted = f_est.permute(2, 0, 1)[:, None, :, :] if denoiser_type != 'wavelet' else f_est
    f_denoised = mask[:, :, None] * F.relu(denoiser(f_est_permuted)).squeeze().permute(1, 2, 0) if denoiser_type != 'wavelet' else mask[:, :, None] * F.relu(denoiser(f_est))
    f_red = mask[:, :, None] * F.relu(lmbda * f_denoised + beta * (f_nf_est + gamma)) / (lmbda + beta + 1e-8)

    errors = [
        torch.norm(f - f_est).detach().cpu().numpy()**2,  # f_noisy_err
        torch.norm(f - f_denoised).detach().cpu().numpy()**2,  # f_denoised_err
        torch.norm(f - mask[:, :, None] * (f_nf_est + gamma)).detach().cpu().numpy()**2,  # f_dual_err
        torch.norm(f - f_red).detach().cpu().numpy()**2  # f_red_err
    ]

    return f_red, errors


def f_update_static(
    lmbda, beta, denoiser, f, f_nf_est, gamma, denoiser_type, P, mask):    
    if denoiser_type == 'wavelet':
        f_denoised = F.relu(denoiser(f))
    else:
        f_denoised = F.relu(denoiser(
            f.permute(2,0,1)[:, None, :, :])).squeeze()[None, ...].permute(1,2,0)
    return mask[:, :, None] * F.relu(
        lmbda * f_denoised + beta * (f_nf_est + gamma)) / (lmbda + beta + 1e-8)
    

def dual_variable_update(gamma, f, f_nf_est):
    ''' Performs ADMM dual variable update for gamma. '''
    return gamma + f_nf_est - f


def learn_static_recon(R, model, optimizer, scheduler, criterion, g,
                       theta_exp, beta, num_primal_iter):
    # Initialize loss vectors
    loss_epoch = {}
    [loss_epoch['total'], loss_epoch['SSIM_f'], loss_epoch['MAE_f'], 
     loss_epoch['f'], loss_epoch['g'], loss_epoch['var_split']
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
            model.f_est - model.z_est + model.gamma_est)**2
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


def dual_variable_update_static(gamma, f, z):
    return gamma + f - z


def z_update_static(lmbda, beta, denoiser, patchifier, z, f, gamma, denoiser_type, P):
    if denoiser_type in ['full_img', 'full_img_plinear']:
        return (lmbda * denoiser(z[None, None, :, :]).squeeze() + beta * (
                f + gamma)) / (lmbda + beta)
    else:
        raise NotImplementedError('ADMM f update not implemented for the denoiser type.')
        
        
def learn_NF_proj(model, model_enc, xyt_grid, optimizer, scheduler, criterion, 
                  g, num_primal_iter, P, spatial_dim, R=None, gamma_est=None, 
                  beta=1, rep=4, tv_weight=0):
    # Initialize loss vectors
    loss_epoch = {}
    [loss_epoch['total'], loss_epoch['SSIM_f'], loss_epoch['MAE_f'], 
     loss_epoch['f'], loss_epoch['g'], loss_epoch['var_split'],
     loss_epoch['tv']] = [[] for _ in range(7)]
    loss_epoch['PSNR_f'] = [-1e4]
    
    # Convert view angle and measurement inputs to torch
    g_gt = torch.Tensor(g).cuda()
        
    # Training
    for _ in range(num_primal_iter):
        
        xyt_grid_enc = model_enc(xyt_grid)
        f_nf_est = model(xyt_grid_enc)
        
        # Compute projections from the estimated object
        g_f_est = torch.einsum('pjs,ps->jp', R, torch.repeat_interleave(
            f_nf_est.squeeze().permute(2,0,1), repeats=rep, dim=0).view(
            P, spatial_dim**2))
        
        # Compute the data fidelity
        loss_g = criterion(g_f_est, g_gt)
        
        # Compute the Lagrangian
        loss_var_split = (beta / 2) * torch.linalg.norm(
            f_nf_est.squeeze() - model.f_est.squeeze() + gamma_est)**2
        
        loss_tv = utils.total_variation_temp_loss(
            f_nf_est.squeeze().permute(2,0,1), tv_weight)
        
        loss = loss_g + loss_var_split + loss_tv # + geo_weight * xyt_grad_sum 
        
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


def learn_NF_SGD_multiview(model, xyt_grid_enc, optimizer, scheduler, criterion, g_gt, num_primal_iter, P, spatial_dim, R, beta, rep, reg_t_weight, reg_t_loss_type, weight_model, type_model_reg, sgd_size):
    # Initialize loss vectors
    loss_epoch = {key: [] for key in ['total', 'f', 'g', 'var_split', 'temp_reg', 'model_reg']}
    
    for epoch in range(num_primal_iter):
        time_idx = np.arange(rep * P) if epoch % (rep * P // sgd_size) == 0 and reg_t_weight != 0 else np.random.choice(np.arange(rep * P), size=sgd_size, replace=False)
        
        f_nf_est = model(xyt_grid_enc[..., time_idx // rep]).squeeze()
        
        # Compute projections from the estimated object
        g_f_est = torch.einsum('pjs,ps->jp', R[time_idx], f_nf_est.permute(2, 0, 1).view(len(time_idx), spatial_dim**2))
        
        # Compute the data fidelity
        loss_g = criterion(g_f_est, g_gt[:, time_idx])
        
        # Compute the Lagrangian
        loss_var_split = (beta / 2) * torch.linalg.norm(f_nf_est - model.f_est[:, :, time_idx // rep] + model.gamma_est[:, :, time_idx // rep])**2 if beta != 0 else torch.zeros(1).cuda()
        
        # Temporal regularization
        loss_temp_reg = loss.reg_loss(f_nf_est[..., ::rep], reg_t_loss_type, reg_t_weight) if reg_t_weight != 0 and epoch % (rep * P // sgd_size) == 0 else torch.zeros(1).cuda()
        
        # l1 sparsity model
        loss_model_reg = loss.model_reg(model, weight_model, type_model_reg) if weight_model != 0 else torch.zeros(1).cuda()
        
        loss = loss_g + loss_var_split + loss_temp_reg + loss_model_reg
        
        # Backprop 
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        
        # Log computed loss values
        for key, value in zip(['total', 'g', 'var_split', 'temp_reg', 'model_reg'], [loss, loss_g, loss_var_split, loss_temp_reg, loss_model_reg]):
            loss_epoch[key].append(value.data.cpu().numpy())
    
    with torch.no_grad():
        f_nf_est = model(xyt_grid_enc).squeeze()
        
    return [loss_epoch, f_nf_est, g_f_est]


def static_embedding(model, xyt_grid_enc, optimizer, criterion, f_static_embed, 
                     f, P, REP, num_epoch, sgd_size, rep):
    # Initialize recon metrics per epoch
    metrics = {key: [] for key in ['PSNR_f', 'PSNR_nf', 'MAE_f', 'MAE_nf', 
                                   'SSIM_f', 'SSIM_nf', 'HFEN_f', 'HFEN_nf', 
                                   'grad_nf']}
    best_psnr_f_static = 1e0
    best_f_static_est = None
    
    upd_freq = num_epoch // 20
    loss_epoch = []
    f_static_embed = torch.Tensor(f_static_embed).cuda()[..., None].repeat(1, 1, REP * P)
    
    Nepoch = tqdm(range(num_epoch), desc='Initializing...', leave=True, ncols=160, colour='green')
    for epoch in Nepoch:
        time_idx = np.arange(rep * P) if epoch % (rep * P // sgd_size) == 0 else np.random.choice(np.arange(rep * P), size=sgd_size, replace=False)
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
                metrics, best_psnr_f_static, best_f_static_est = metrics.update_metrics(
                    f[..., ::rep], model.f_est.squeeze().detach().cpu().numpy(), 
                    f_nf_est.squeeze().detach().cpu().numpy(), metrics, rep, 
                    best_psnr_f_static, best_f_static_est, model)
                
                Nepoch.set_description(
                    'PSNR st:%.3e/%.3e SSIM st:%.2e/%.2e MAE st:%.2e/%.2e HFEN st:%.2e/%.2e grad_nf st:%.3e' %(
                        metrics['PSNR_f'][-1], metrics['PSNR_nf'][-1], 
                        metrics['SSIM_f'][-1], metrics['SSIM_nf'][-1], 
                        metrics['MAE_f'][-1], metrics['MAE_nf'][-1], 
                        metrics['HFEN_f'][-1], metrics['HFEN_nf'][-1], 
                        metrics['grad_nf'][-1]))

    with torch.no_grad():
        f_nf_est = model(xyt_grid_enc)
        model.f_est = f_nf_est.detach().clone().squeeze()
        model.gamma_est = torch.zeros(model.f_est.shape).cuda()
    
    return metrics, best_psnr_f_static, best_f_static_est, f_nf_est, loss_epoch

