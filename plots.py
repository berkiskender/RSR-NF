from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
plt.rcParams['animation.html'] = 'html5'
import matplotlib.animation as animation


def plot_and_save_red_psm_results(params, obj_type, results_per_config, P,
                                  update_type, noise_std, ang_period,
                                  save_fig):
    STR_PERIOD = '' if ang_period is None else '_' + str(ang_period)
    PATH = 'data/red_psm_results/%s_%d%s_denoiser_%s_%s/' %(
        obj_type, P, STR_PERIOD, params['denoiser_type'], 
        update_type)
    if params['denoiser_type'] == 'patch_based':
        PATH = PATH + '_patch_sz_' + str(
            params['pSize']) + '_patch_str_' + str(
            params['pStride']) + '_num_layers_' + str(
            params['num_layers']) + '_num_ch_' + str(
            params['num_channels'])
    PATH += 'data/'
    Path(PATH).mkdir(parents=True, exist_ok=True)

    print('Results with experimental order:\n')
    for res in results_per_config:
        print(res[0], 'PSNR:', res[1]['PSNR_f'][-1],
              'SSIM:', res[1]['SSIM_f'][-1],
              'MAE:', res[1]['MAE_f'][-1],
              'HFEN:', res[1]['HFEN_f'][-1], res[3])

    results_per_config_sorted = sorted(
        results_per_config, key=lambda res: res[0])

    cnt = 0
    plt.figure(figsize=(20,3))
    for res in results_per_config_sorted:
        label_str = (
            r'$\beta$:%.1e $\lambda$:%.1e' %(res[3]['beta'], res[3]['lmbda'])
            + '\npsz/pstr/layers:%d/%d/%d' %(
                res[3]['pSize'], res[3]['pStride'], res[3]['num_layers']))
        if cnt % 3 == 0:
            LINESTYLE='--'
        elif cnt % 3 == 1:
            LINESTYLE='-'
        else:
            LINESTYLE='-.'
        plt.subplot(1,4,1); plt.title('PSNR (dB)')
        plt.plot(res[1]['PSNR_f'], label=label_str, linestyle=LINESTYLE)
        plt.grid(linestyle='--', linewidth=0.5)
        plt.subplot(1,4,2); plt.title('SSIM')
        plt.plot(res[1]['SSIM_f'], label=label_str, linestyle=LINESTYLE)
        plt.grid(linestyle='--', linewidth=0.5)
        plt.subplot(1,4,3); plt.title('MAE')
        plt.plot(res[1]['MAE_f'], label=label_str, linestyle=LINESTYLE)
        plt.grid(linestyle='--', linewidth=0.5)
        plt.subplot(1,4,4); plt.title('HFEN')
        plt.plot(res[1]['HFEN_f'], label=label_str, linestyle=LINESTYLE)
        plt.grid(linestyle='--', linewidth=0.5)
        plt.legend(bbox_to_anchor=(0.75, -0.25), ncol=3)
        cnt += 1
    if save_fig:
        save_fig_path = (
            'acc_res_P_%d_%s_num_epoch_%d'
            '_num_primal_iter_%d_noise_std_%.2e'
            '_lr_%.2e_num_exps_%d_init_%s' %(
                P, res[3]['temporal_basis'], params['num_epoch'], 
                params['num_primal_iter'], noise_std, params['lr_primal'], 
                len(results_per_config_sorted), params['temporal_basis']))
        plt.savefig(PATH + save_fig_path + '.jpg', bbox_inches='tight')
        plt.savefig(PATH + save_fig_path + '.pdf', bbox_inches='tight')
    plt.show()
    
    cnt = 0
    plt.figure(figsize=(5,3))
    for res in results_per_config_sorted:
        label_str = (
            r'$\beta$:%.1e $\lambda$:%.1e' %(res[3]['beta'], res[3]['lmbda'])
            + '\npsz/pstr/layers:%d/%d/%d' %(
                res[3]['pSize'], res[3]['pStride'], res[3]['num_layers']))
        if cnt % 3 == 0:
            LINESTYLE='--'
        elif cnt % 3 == 1:
            LINESTYLE='-'
        else:
            LINESTYLE='-.'
        plt.subplot(1,1,1); plt.title(r'$\|\alpha\|_2$')
        plt.plot(res[1]['grad_nf'], label=label_str, linestyle=LINESTYLE)
        plt.grid(linestyle='--', linewidth=0.5)
        plt.legend(bbox_to_anchor=(0.75, -0.25), ncol=3)
        cnt += 1
    plt.ylim(-3, 1)
    if save_fig:
        save_fig_path = (
            'grad_nf_P_%d_%s_num_epoch_%d'
            '_num_primal_iter_%d_noise_std_%.2e'
            '_lr_%.2e_num_exps_%d_init_%s' %(
                P, res[3]['temporal_basis'], params['num_epoch'], 
                params['num_primal_iter'], noise_std, params['lr_primal'], 
                len(results_per_config_sorted), params['temporal_basis']))
        plt.savefig(PATH + save_fig_path + '.jpg', bbox_inches='tight')
        plt.savefig(PATH + save_fig_path + '.pdf', bbox_inches='tight')
    plt.show()
    
    print('Results with increasing PSNR:\n')
    for res in results_per_config_sorted:
        print(
            'PSNR:', res[0],
            'SSIM:', res[1]['SSIM_f'][np.argmax(np.array(res[1]['PSNR_f']))], 
            'MAE:', res[1]['MAE_f'][np.argmax(np.array(res[1]['PSNR_f']))],
            'HFEN', res[1]['HFEN_f'][np.argmax(np.array(res[1]['PSNR_f']))]
        )
        if save_fig:
            save_data_path = (
                '_K_%d_d_%d_beta_%.2e_lambda_%.2e_xi_%.2e_chi_%.2e'
                '_PSNR_%.2e_pSize_%d_pStride_%d_numlayers_%d_init_%s.npy' %(
                    res[3]['K'], res[3]['z_dim'], res[3]['beta'],
                    res[3]['lmbda'], res[3]['xi'], res[3]['chi'], res[0], 
                    res[3]['pSize'], res[3]['pStride'], res[3]['num_layers'], 
                    params['temporal_basis']))
            np.save(PATH + 'f_est' + save_data_path, res[4]['f_est'])
            np.save(PATH + 'spatial_basis_est' + save_data_path,
                    res[4]['spatial_basis_fcts'])
            np.save(PATH + 'temporal_basis_est' + save_data_path,
                    res[4]['temporal_latent_fcts'])
            np.save(PATH + 'psi_est' + save_data_path,
                    res[4]['psi_mtx'].detach().cpu().numpy())
    pass


def train_visualization(loss_epoch, f_est, f_psm_est, gamma_bar_est, f, t, plot):
    ''' Visualize intermediate training results for PSM-based RED '''
    if plot:
        metrics = ['total', 'temp_reg', 'g', 'var_split']
        titles = [r'$\ln$(loss)', r'$\ln$(loss $t$ reg)', r'$\ln$(loss $g$)', r'$\ln$(loss aug lag)']
        
        plt.figure(figsize=(15, 2))
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            plt.subplot(1, 9, i + 1)
            plt.title(title)
            plt.plot(np.log(loss_epoch[metric]))
            plt.grid()
        
        images = [f_est[..., t], f_psm_est[..., t], gamma_bar_est[..., t], f[..., t], f_est[..., t] - f[..., t]]
        img_titles = [r'$f_{est}$', r'$f_{nf}$', r'$\gamma_{est}$', r'$f$', r'$f-f_{est}$']
        
        for i, (img, img_title) in enumerate(zip(images, img_titles)):
            plt.subplot(1, 9, i + 5)
            plt.title(img_title)
            plt.imshow(img)
            plt.colorbar()
        
        plt.show()
     
    
def red_visualization(metrics, plot):
    ''' Visualize RED training results '''
    if plot:
        plt.figure(figsize=(12, 6))
        plt.title(r'RED $\ln$(loss)')
        for label, linestyle in zip(['red_in', 'denoised', 'red_out', 'dual'], ['--', '-.', '-', '--']):
            plt.plot(np.log(metrics[f'f_{label}_norm_err']), label=label, linestyle=linestyle, linewidth=0.5)
        plt.grid()
        plt.legend()
        plt.show()


def train_visualization_proj(loss_epoch, g_est, g_nf_est, gamma_bar_est, g, t):
    ''' Visualize intermediate training results for projection-based RED '''
    metrics = ['total', 'tv_t', 'tv_s', 'var_split', 'g']
    titles = [r'$\ln$(loss)', r'$\ln$(tv_t)', r'$\ln$(tv_s)', r'$\ln$(var_split)', r'$\ln$(loss $g$)']
    
    plt.figure(figsize=(10, 4))
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(2, 5, i + 1)
        plt.title(title)
        plt.plot(np.log(loss_epoch[metric] + 1e-8))
    
    images = [g_est[..., t], g_nf_est[..., t], gamma_bar_est[..., t], g[..., t], g_est[..., t].numpy() - g[..., t]]
    img_titles = [r'$g_{est}$', r'$g_{nf,est}$', r'$\gamma_{est}$', r'$g$', r'$g-g_{est}$']
    
    for i, (img, img_title) in enumerate(zip(images, img_titles)):
        plt.subplot(2, 5, i + 6)
        plt.title(img_title)
        plt.imshow(img)
        plt.colorbar()
    
    plt.show()


def plot_psm_basis_fcts(psi_mtx, spatial_basis_fcts, K):
    fig, ax = plt.subplots(1, K+1, figsize=(3*(K+1), 3))
    ax[0].set_ylabel(r'$\Psi$')
    for k in range(K+1):
        ax[k].plot(psi_mtx[k].detach().cpu())
        ax[k].tick_params(left=False, right=False , labelleft=False,
                labelbottom=False, bottom=False)
    plt.show()
    
    fig, ax = plt.subplots(1, K+1, figsize=(3*(K+1), 3))
    ax[0].set_ylabel(r'$\Lambda$')
    for k in range(K+1):
        ax[k].imshow(spatial_basis_fcts[k].detach().cpu())
        ax[k].tick_params(left=False, right=False , labelleft=False,
                labelbottom=False, bottom=False)
    plt.show()
    pass


def display_f(f, image, rate, P):
    ''' Display frames of the 2D field f '''
    Psqrt = np.int(np.ceil(np.sqrt(P / rate)))
    plt.figure(figsize=(5, 5))
    for p in range(P):
        if p % rate == 0:
            plt.subplot(Psqrt, Psqrt, p // rate + 1)
            plt.imshow(f[..., p], cmap='gray')
            plt.clim(0, np.max(image))
    plt.show()
    

def display_inputs(f, theta_exp, P, num_frames):
    ''' Display ground-truth frames and acquisition scheme '''
    num_frames = 8
    fig, ax = plt.subplots(1, num_frames, figsize=(2.5*num_frames, 3))
    plt.suptitle(r'Ground-truth frames $f_t$ for $t \in [0, 1]$')
    for k in range(num_frames):
        ax[k].imshow(f[..., k*P//num_frames], cmap='gray', clim=(0,f.max()))
        ax[k].set_title(r'$t = %d/%d$' %(k*P//num_frames, P))
        ax[k].axis('off')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(2.5*num_frames, 3))
    plt.suptitle(r'Acquisition scheme $\{\theta(t)\}$ for $t \in [0, 1]$')
    plt.scatter(np.arange(len(theta_exp))/len(theta_exp), theta_exp, s=10)
    plt.ylabel(r'$\theta(t)$, $rad$')
    plt.xlabel(r'$t$, $sec$')
    plt.grid(linestyle='--', linewidth=0.25, which='major', alpha=0.75)
    plt.show()
    
    
def plot_and_save_red_nf_results(params, obj_type, results_per_config, P, 
                                 update_type, noise_std, ang_period, 
                                 pos_enc_type, mapping_size, scale, save_fig):
    ''' Plot and save RED NF results '''
    STR_PERIOD = f'_{ang_period}' if ang_period else ''
    PATH = (f'data/REBUTTAL_NF_2d_t_CT_results/{obj_type}_{P}{STR_PERIOD}_denoiser_'
            f'{params["denoiser_type"]}_{update_type}_enc_{pos_enc_type}_mapS_'
            f'{mapping_size}_sc_{scale}/')
    if params['denoiser_type'] == 'patch_based':
        PATH += (f'_patch_sz_{params["pSize"]}_patch_str_{params["pStride"]}_'
                 f'num_layers_{params["num_layers"]}_num_ch_{params["num_channels"]}_'
                 f'num_layers_denoiser_{params["num_layers_denoiser"]}_'
                 f'num_ch_denoiser_{params["num_channels_denoiser"]}')
    PATH += 'data/'
    Path(PATH).mkdir(parents=True, exist_ok=True)

    print('Results with experimental order:\n')
    for res in results_per_config:
        print(f'{res[0]} PSNR: {res[1]["PSNR_f"][-1]:.2e} SSIM: {res[1]["SSIM_f"][-1]:.2e} '
              f'MAE: {res[1]["MAE_f"][-1]:.2e} HFEN: {res[1]["HFEN_f"][-1]:.2e} {res[3]}')

    results_per_config_sorted = sorted(results_per_config, key=lambda res: res[0])

    cnt = 0
    plt.figure(figsize=(20, 3))
    metrics = ['PSNR_f', 'SSIM_f', 'MAE_f', 'HFEN_f']
    titles = ['PSNR (dB)', 'SSIM', 'MAE', 'HFEN']
    for res in results_per_config_sorted:
        label_str = (
            r'$\beta$:%.1e $\lambda$:%.1e' % (res[3]['beta'], res[3]['lmbda'])
            + ' l/ch/mapS/sc/tReg/sgdS/modR/D:\n%d/%d/%d/%d/%.1e/%d/%.1e/%s' % (
                res[3]['num_layers'], res[3]['num_channels'], res[3]['mapping_size'],
                res[3]['scale'], res[3]['reg_t_weight'], res[3]['sgd_size'],
                res[3]['weight_model'], res[3]['denoiser_model']
            )
        )
        LINESTYLE = ['--', '-', '-.'][cnt % 3]
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            plt.subplot(1, 4, i + 1)
            plt.title(title)
            plt.plot(res[1][metric], label=label_str, linestyle=LINESTYLE)
            plt.grid(linestyle='--', linewidth=0.5)

            save_np_path = (f'beta_{res[3]["beta"]:.2e}_lambda_{res[3]["lmbda"]:.2e}_' 
                            f'PSNR_{res[0]:.2e}_nlyr_{res[3]["num_layers"]}_nch_{res[3]["num_channels"]}_'
                            f'enc_{pos_enc_type}_mapS_{res[3]["mapping_size"]}_sc_{res[3]["scale"]}_'
                            f'init_{res[3]["init"]}_sgdS_{res[3]["sgd_size"]}_rep_{res[3]["rep"]}_{res[3]["reg_mode"]}_{metric}.npy')
            np.save(PATH + save_np_path, res[1][metric])

        plt.legend(bbox_to_anchor=(0.75, -0.25), ncol=3)
        cnt += 1

    max_PSNR = results_per_config_sorted[0][0]

    if save_fig:
        save_fig_path = (
            'max_PSNR_%.2e_acc_res_P_%d_num_epoch_%d'
            '_num_primal_iter_%d_noise_std_%.2e'
            '_lr_%.2e_num_exps_%d_rep_%d_%s' %(
                max_PSNR, P, res[3]['num_epoch'], res[3]['num_primal_iter'], noise_std,
                res[3]['lr_primal'], len(results_per_config_sorted), res[3]['rep'], 
                res[3]['reg_mode']))
        plt.savefig(PATH + save_fig_path + '.jpg', bbox_inches='tight')
        plt.savefig(PATH + save_fig_path + '.pdf', bbox_inches='tight')

        with open(PATH + save_fig_path + '.txt', "w") as text_file:
            for res in results_per_config_sorted:
                out = (f"{res[0]} PSNR: {res[1]['PSNR_f'][-1]:.2e} "
                       f"SSIM: {res[1]['SSIM_f'][-1]:.2e} "
                       f"MAE: {res[1]['MAE_f'][-1]:.2e} "
                       f"HFEN: {res[1]['HFEN_f'][-1]:.2e} {res[3]}")
                text_file.write(out + "\n")

    plt.show()
    
    print('Results with increasing PSNR:\n')
    for res in results_per_config_sorted:
        psnr_idx = np.argmax(np.array(res[1]['PSNR_f']))
        print(f"PSNR: {res[0]:.2e} SSIM: {res[1]['SSIM_f'][psnr_idx]:.2e} "
              f"MAE: {res[1]['MAE_f'][psnr_idx]:.2e} HFEN: {res[1]['HFEN_f'][psnr_idx]:.2e}")
        if save_fig:
            save_data_path = (f'beta_{res[3]["beta"]:.2e}_lambda_{res[3]["lmbda"]:.2e}_'
                              f'PSNR_{res[0]:.2e}_nlyr_{res[3]["num_layers"]}_nch_{res[3]["num_channels"]}_'
                              f'enc_{params["pos_enc_type_sweep"]}_mapS_{res[3]["mapping_size"]}_sc_{res[3]["scale"]}_'
                              f'init_{res[3]["init"]}_sgdS_{res[3]["sgd_size"]}_rep_{res[3]["rep"]}_{res[3]["reg_mode"]}.npy')
            np.save(PATH + 'f_est' + save_data_path, res[4]['f_est'])
            

    cnt = 0
    plt.figure(figsize=(6, 3))
    for res in results_per_config_sorted:
        label_str = (
            r'$\beta$:%.1e $\lambda$:%.1e' % (res[3]['beta'], res[3]['lmbda'])
            + ' l/ch/mapS/sc/tReg/sgdS/modR: %d/%d/%d/%d/%.1e/%d/%.1e' % (
                res[3]['num_layers'], res[3]['num_channels'], res[3]['mapping_size'],
                res[3]['scale'], res[3]['reg_t_weight'], res[3]['sgd_size'], res[3]['weight_model']
            )
        )
        LINESTYLE = ['--', '-', '-.'][cnt % 3]
        plt.subplot(1, 1, 1)
        plt.title(r'$\log_{10}\|\alpha\|_2$')
        plt.plot(np.log10(np.array(res[1]['grad_nf']) + 1e-10), label=label_str, linestyle=LINESTYLE)
        plt.grid(linestyle='--', linewidth=0.5)
        plt.legend(bbox_to_anchor=(0.75, -0.25), ncol=3)
        cnt += 1
    plt.ylim(-3, 1)
    if save_fig:
        save_fig_path = (
            'grad_nf_P_%d_num_epoch_%d_num_primal_iter_%d_noise_std_%.2e_lr_%.2e_num_exps_%d_rep_%d' % (
                P, res[3]['num_epoch'], res[3]['num_primal_iter'], noise_std,
                res[3]['lr_primal'], len(results_per_config_sorted), res[3]['rep']
            )
        )
        plt.savefig(PATH + save_fig_path + '.jpg', bbox_inches='tight')
        plt.savefig(PATH + save_fig_path + '.pdf', bbox_inches='tight')
    plt.show()
    
    
    for res in results_per_config_sorted[-3:]:
        red_visualization(res[1], True)
    
    return None


def save_animation(f, P, fps_true, downsampling_rate, cbar=False, vmin=0, vmax=1, cmap='viridis', save_path=None, name=None):
    ''' Save animation of the 2D field f '''
    fps = fps_true // downsampling_rate
    nSeconds = P // (downsampling_rate * fps)
    snapshots = f[..., ::downsampling_rate].transpose(2, 0, 1)

    fig, ax = plt.subplots()
    im = ax.imshow(snapshots[0], interpolation='none', aspect='auto', cmap=cmap, vmin=(vmin if cbar else None), vmax=(vmax if cbar else None))
    ax.axis('off')
    plt.tight_layout()

    def animate_func(i):
        if i % fps == 0:
            print('.', end='')
        im.set_array(snapshots[i])
        return [im]

    anim = animation.FuncAnimation(fig, animate_func, frames=nSeconds * fps, interval=1000 / fps)
    anim.save(f'{save_path}/f_sample_res_{name}.gif', fps=fps)
    anim.save(f'{save_path}/f_sample_res_{name}.mp4', fps=fps)
    print('Done!')
    plt.close()


def save_animation_comp(f, f_rec, P, fps_true, downsampling_rate, cbar, vmin, vmax, cmap, cmap_err, save_path, name):
    ''' Save animation of the 2D field f and its reconstruction f_rec '''
    fps = fps_true // downsampling_rate
    nSeconds = P // (downsampling_rate * fps)
    snapshots1 = f[..., ::downsampling_rate].transpose(2, 0, 1)
    snapshots2 = f_rec[..., ::downsampling_rate].transpose(2, 0, 1)
    snapshots3 = np.abs(snapshots1 - snapshots2)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ims = [ax[i].imshow(snapshots1[0], interpolation='none', aspect='auto', cmap=cmap if i < 2 else cmap_err,
                        vmin=(vmin if cbar else None), vmax=(vmax if cbar else None)) for i in range(3)]
    for a in ax:
        a.axis('off')
    plt.tight_layout()

    def animate_func(i):
        if i % fps == 0:
            print('.', end='')
        for im, snapshot in zip(ims, [snapshots1[i], snapshots2[i], snapshots3[i]]):
            im.set_array(snapshot)
        return ims

    anim = animation.FuncAnimation(fig, animate_func, frames=nSeconds * fps, interval=1000 / fps)
    anim.save(f'{save_path}/f_comp_sample_res_{name}.gif', fps=fps)
    anim.save(f'{save_path}/f_comp_sample_res_{name}.mp4', fps=fps)
    print('Done!')
    plt.close()


def save_animation_comp_red(f, f_rec, f_rec_red, P, fps_true, downsampling_rate, cbar, vmin, vmax, cmap, cmap_err, save_path, name):
    ''' Save animation of the 2D field f, its reconstruction f_rec, and its reconstruction with red f_rec_red '''
    fps = fps_true // downsampling_rate
    nSeconds = P // (downsampling_rate * fps)
    snapshots1 = f[..., ::downsampling_rate].transpose(2, 0, 1)
    snapshots2 = f_rec[..., ::downsampling_rate].transpose(2, 0, 1)
    snapshots3 = f_rec_red[..., ::downsampling_rate].transpose(2, 0, 1)
    snapshots4 = np.abs(snapshots1 - snapshots2)
    snapshots5 = np.abs(snapshots1 - snapshots3)

    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    ims = [ax[i].imshow(snapshots1[0], interpolation='none', aspect='auto', cmap=cmap if i < 3 else cmap_err,
                        vmin=(vmin if cbar else None), vmax=(vmax if cbar else None)) for i in range(5)]
    for a in ax:
        a.axis('off')
    plt.tight_layout()

    def animate_func(i):
        if i % fps == 0:
            print('.', end='')
        for im, snapshot in zip(ims, [snapshots1[i], snapshots2[i], snapshots3[i], snapshots4[i], snapshots5[i]]):
            im.set_array(snapshot)
        return ims
    
    anim = animation.FuncAnimation(fig, animate_func, frames=nSeconds * fps, interval=1000 / fps)
    anim.save(f'{save_path}/f_comp_w_red_sample_res_{name}.gif', fps=fps)
    anim.save(f'{save_path}/f_comp_w_red_sample_res_{name}.mp4', fps=fps)
    print('Done!')
    plt.close()


