import utils

def update_metrics(f, f_est, f_nf_est, metrics, rep, best_psnr_f_est, best_f_est, model):
    ''' Updates accuracy metrics. '''
    
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


def update_static_metrics(f, f_est, z_est, metrics, best_psnr_f_est, best_f_est):
    ''' Updates accuracy metrics. '''
    
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


def update_projection_metrics(g, g_est, g_nf_est, metrics, best_psnr_g_est, best_g_est):
    ''' Updates accuracy metrics. '''
    
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