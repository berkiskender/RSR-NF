import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

plt.style.use(['dark_background'])

import torch
import torchvision.transforms as T
from jacobian import JacobianReg_l2
from skimage.transform import iradon, radon
from torch import nn as nn

import denoiser_data_load
import models_denoiser
import utils


def main():
    np.set_printoptions(precision=2, suppress=True)

    args = parser.parse_args()
    print(args)

    obj_type = args.obj_type
    spatial_dim = args.spatial_dim
    std_high = args.std_high
    std_low = args.std_low
    noise_std = args.noise_std
    path = args.path
    denoiser_type = args.denoiser_type
    numLayers = args.numLayers
    numChannels = args.numChannels
    filterSize = args.filterSize
    noise_est_type = args.noise_est_type
    lr = args.lr
    Nepoch = args.Nepoch

    blur = args.blur
    rotate = args.rotate
    limited_view = args.limited_view
    lambda_jr = args.lambda_jr

    save_fig_path = args.save_fig_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(args.device)

    [train_data_clean, test_data_clean, f_dynamic_clean] = load_data(
        obj_type, std_high, std_low, noise_std, path
    )

    print(
        '\nTrain data shape and max dens:',
        train_data_clean.shape,
        train_data_clean.max(),
        '\nTest data shape and max dens:',
        test_data_clean.shape,
        test_data_clean.max(),
        '\nDynamic data shape and max dens:',
        f_dynamic_clean.shape,
        f_dynamic_clean.max(),
    )

    model = load_denoiser(args)
    print('\nModel:\n', model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    rot_p, rot_angles = 0.5, [0, 90, 180, 270]

    if blur:
        tx = T.GaussianBlur(kernel_size=(3, 5), sigma=(0.0001, 2.0))
        blur_tx = blur_transform
    else:
        tx = None
        blur_tx = lambda x: x

    if limited_view:
        limited_view_tx = limited_view_transform
        P_set = [8, 16, 32, 64, 128]
    else:
        limited_view_tx = lambda x: x
        P_set = None

    if rotate:
        rot_p, rot_angles = 0.5, [0, 90, 180, 270]
    else:
        rot_p, rot_angles = 0.0, [0]

    mask = generate_mask(args)

    train_dataset = denoiser_data_load.Dataset(
        train_data_clean, train_data_clean, rot_p=rot_p, rot_angles=rot_angles
    )
    dataloader = denoiser_data_load.DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=0
    )

    val_dataset = denoiser_data_load.Dataset(
        test_data_clean, test_data_clean, rot_p=rot_p, rot_angles=rot_angles
    )
    val_dataloader = denoiser_data_load.DataLoader(
        val_dataset, batch_size=16, shuffle=True, num_workers=0
    )

    if obj_type in ['polymer', 'polymer_binary', 'polymer_binary_normalized']:
        dynamic_dataset = denoiser_data_load.Dataset(
            f_dynamic_clean, f_dynamic_clean, rot_p=rot_p, rot_angles=rot_angles
        )
        dynamic_dataloader = denoiser_data_load.DataLoader(
            dynamic_dataset, batch_size=16, shuffle=True, num_workers=0
        )

    reg_fun, reg_fun_val = JacobianReg_l2(), JacobianReg_l2(eval_mode=True)
    epsilon = 0.0

    loss_train_epoch, loss_val_epoch, loss_dyn_epoch, loss_reg_epoch = [], [], [], []

    t = trange(Nepoch, desc='Loss - Train: %.2e Reg: %.2e Val: %.2e' % (0, 0, 0), leave=True)

    for epoch in t:
        loss_train, loss_reg = 0, 0
        for i_batch, sample_batch in enumerate(dataloader):
            noise_std_rand_train = float(np.random.uniform(low=std_low, high=std_high))

            transformed = limited_view_tx(sample_batch['noisy'], P_set)
            transformed = blur_tx(transformed, tx)
            transformed = noise_transform(
                transformed, max_dens=sample_batch['noisy'].max(), noise_std=noise_std_rand_train
            )

            denoised = model(mask * transformed)
            loss = criterion(denoised, mask * (sample_batch['gt']))

            if lambda_jr != 0:
                reg_loss_max = compute_reg(
                    denoised, sample_batch['noisy'], model, reg_fun, epsilon=epsilon
                )
                reg_loss_val = lambda_jr * reg_loss_max
            else:
                reg_loss_val = torch.linalg.norm(torch.zeros(1).cuda())

            loss += reg_loss_val

            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()

            loss_train += loss.data.cpu().numpy()
            loss_reg += reg_loss_val.data.cpu().numpy()

        loss_train_epoch.append(loss_train / (i_batch + 1))
        loss_reg_epoch.append(loss_reg / (i_batch + 1))

        with torch.no_grad():
            loss_val = 0
            for i_batch, sample_batch in enumerate(val_dataloader):
                noise_std_rand_val = float(np.random.uniform(low=std_low, high=std_high))
                transformed = limited_view_tx(sample_batch['noisy'], P_set)
                transformed = blur_tx(transformed, tx)
                transformed = noise_transform(
                    transformed, max_dens=sample_batch['noisy'].max(), noise_std=noise_std_rand_val
                )

                denoised = model.eval()(mask * transformed)
                loss_val += criterion(denoised, mask * (sample_batch['gt'])).data.cpu().numpy()
        loss_val_epoch.append(loss_val / (i_batch + 1))

        if obj_type in ['polymer_binary', 'polymer', 'polymer_binary_normalized']:
            with torch.no_grad():
                loss_dyn = 0
                for i_batch, sample_batch in enumerate(dynamic_dataloader):
                    noise_std_rand_dyn = float(np.random.uniform(low=std_low, high=std_high))
                    transformed = limited_view_tx(sample_batch['noisy'], P_set)
                    transformed = blur_tx(transformed, tx)
                    transformed = noise_transform(
                        transformed,
                        max_dens=sample_batch['noisy'].max(),
                        noise_std=noise_std_rand_dyn,
                    )

                    denoised = model.eval()(mask * transformed)
                    loss_dyn += criterion(denoised, mask * (sample_batch['gt'])).data.cpu().numpy()
            loss_dyn_epoch.append(loss_dyn / (i_batch + 1))

        if obj_type in ['walnut', 'walnut_normalized']:
            t.set_description(
                'Loss - Train: %.2e Reg: %.2e Val: %.2e'
                % (loss_train_epoch[-1], loss_reg_epoch[-1], loss_val_epoch[-1])
            )
        elif obj_type in ['polymer', 'polymer_binary', 'polymer_binary_normalized']:
            t.set_description(
                'Loss - Train: %.2e Reg: %.2e Val: %.2e Dyn: %.2e'
                % (loss_train_epoch[-1], loss_reg_epoch[-1], loss_val_epoch[-1], loss_dyn_epoch[-1])
            )

        if epoch > Nepoch // 8 and loss_val_epoch[-1] == min(loss_val_epoch):
            torch.save(
                model,
                path
                + '%s_model_%s_%s_epochs_%d_num_layers_%d_num_ch_%d_noise_std_%.1e.pt'
                % (
                    denoiser_type,
                    obj_type,
                    noise_est_type,
                    Nepoch,
                    numLayers,
                    numChannels,
                    std_high,
                ),
            )

    save_results(loss_train_epoch, loss_val_epoch, loss_dyn_epoch, train_data_clean, args)


def save_results(loss_train_epoch, loss_val_epoch, loss_dyn_epoch, train_data_clean, args):
    Path(args.save_fig_path).mkdir(parents=True, exist_ok=True)
    filename = (
        f'{args.obj_type}_{args.denoiser_type}_noise_{args.std_low}_{args.std_high}_est_type_{args.noise_est_type}'
        f'_lr_{args.lr}_lyr_{args.numLayers}_ch_{args.numChannels}_filtsize_{args.filterSize}_blur_{args.blur}_rotate_{args.rotate}'
    )

    plt.figure()
    plt.title(r'$\log_{10}$(MSE) vs epoch')
    plt.plot(np.log10(loss_train_epoch), label='train')
    plt.plot(np.log10(loss_val_epoch), label='val')
    plt.plot(np.log10(loss_dyn_epoch), label='dyn')
    plt.legend()
    plt.xlabel('epoch')
    plt.grid(linewidth=0.2)
    plt.savefig(args.save_fig_path + filename + '_MSE.jpg', bbox_inches='tight')
    plt.savefig(args.save_fig_path + filename + '_MSE.pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.title('PSNR (dB) vs. epoch')
    plt.plot(10 * np.log10(train_data_clean.max() / np.array(loss_train_epoch)), label='train')
    plt.plot(10 * np.log10(train_data_clean.max() / np.array(loss_val_epoch)), label='val')
    plt.plot(10 * np.log10(train_data_clean.max() / np.array(loss_dyn_epoch)), label='dyn')
    plt.legend()
    plt.xlabel('epoch')
    plt.grid(linewidth=0.2)
    plt.savefig(args.save_fig_path + filename + '_PSNR.jpg', bbox_inches='tight')
    plt.savefig(args.save_fig_path + filename + '_PSNR.pdf', bbox_inches='tight')
    plt.close()


def generate_mask(args):
    mask = torch.ones([1, 1, args.spatial_dim, args.spatial_dim]).cuda()
    for i in range(args.spatial_dim):
        for j in range(args.spatial_dim):
            if (i - args.spatial_dim // 2) ** 2 + (j - args.spatial_dim // 2) ** 2 >= (
                args.spatial_dim // 2
            ) ** 2:
                mask[0, 0, i, j] = 0
    return mask


def load_denoiser(args):
    if 'plinear' in args.denoiser_type:
        # model = models_denoiser.plinear_denoiser(
        #     args.numLayers, args.latent_dim, args.spatial_dim, args.est_type=noise_est_type).cuda()
        model = models_denoiser.dncnn_plinear(
            args.numLayers, args.numChannels, args.filterSize, est_type=args.noise_est_type
        ).cuda()
    elif 'dncnn' in args.denoiser_type:
        model = models_denoiser.dncnn(
            args.numLayers, args.numChannels, args.filterSize, est_type=args.noise_est_type
        ).cuda()
    elif 'unet' in args.denoiser_type:
        model = models_denoiser.UNet(in_channels=1, out_channels=1, init_features=32).cuda()
    return model


def noise_transform(x, max_dens, noise_std):
    return x + (max_dens * noise_std) * torch.randn(x.shape).cuda()


def limited_view_transform(f, P_set):
    P = np.random.choice(P_set)
    alpha = np.random.uniform(0.0, 1.0)

    for i in range(f.shape[0]):
        theta_dict = {}
        theta = utils.generate_theta(P, ang_range=np.pi, period=None)['bit_reversal']
        g = radon(f[i].squeeze().detach().cpu().numpy(), theta=360 * theta / (2 * np.pi))
        f[i] = (
            alpha
            * torch.Tensor(iradon(g, theta=360 * theta / (2 * np.pi), filter_name='ramp'))
            .view(1, f.shape[2], f.shape[3])
            .cuda()
            + (1 - alpha) * f[i]
        )
    return f


def blur_transform(f, tx):
    alpha = np.random.uniform(0.0, 1.0)
    f = (1 - alpha) * f + alpha * tx(f)
    return f


def compute_reg(out, data_true, model, reg_fun, epsilon):
    """
    Computes the regularization reg_fun applied to the correct point
    """
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    torch.backends.cudnn.benchmark = True

    out_detached = out.detach().type(Tensor)
    true_detached = data_true.detach().type(Tensor)

    tau = torch.rand(true_detached.shape[0], 1, 1, 1).type(Tensor)
    out_detached = tau * out_detached + (1 - tau) * true_detached
    out_detached.requires_grad_()

    out_reg = model(out_detached)

    out_net_reg = 2.0 * out_reg - out_detached
    reg_loss = reg_fun(out_detached, out_net_reg)
    reg_loss_max = torch.max(reg_loss, torch.ones_like(reg_loss) - epsilon)
    return reg_loss_max.max()


def load_data(obj_type, std_high, std_low, noise_std, path):
    if 'walnut' in obj_type:
        train_data_clean = np.load(path + 'walnut_train_warped_gt.npy')
        test_data_clean = np.load(path + 'walnut_test_warped_gt.npy')

        # train_data_noisy = np.load(path+'walnut_train_warped_noise_std_%.2e.npy' %(noise_std))
        # test_data_noisy = np.load(path+'walnut_test_warped_noise_std_%.2e.npy' %(noise_std))

        # train_data_noisy_rand = np.load(path+'walnut_train_warped_noise_std_rand_low_%.2e_high_%.2e.npy' %(std_low, std_high))
        # test_data_noisy_rand = np.load(path+'walnut_test_warped_noise_std_rand_low_%.2e_high_%.2e.npy' %(std_low, std_high))

        f_dynamic_clean = np.load(path + 'walnut_test_warped_gt.npy')
        # f_dynamic_noisy_rand = np.load(path+'walnut_test_warped_noise_std_rand_low_%.2e_high_%.2e.npy' %(std_low, std_high))

        if obj_type == 'walnut_normalized':
            norm_sc = train_data_clean.max()

            train_data_clean /= norm_sc
            test_data_clean /= norm_sc
            # train_data_noisy /= norm_sc
            # test_data_noisy /= norm_sc
            # train_data_noisy_rand /= norm_sc
            # test_data_noisy_rand /= norm_sc

    elif obj_type == 'polymer_binary':
        train_data_clean = np.load(path + 'polymer_binary_train_gt_unmasked.npy')
        test_data_clean = np.load(path + 'polymer_binary_test_gt_unmasked.npy')
        f_dynamic_clean = np.load(path + 'polymer_binary_dynamic_gt_unmasked.npy')

        # train_data_noisy_rand = np.load(path+'polymer_binary_train_noise_std_rand_low_%.2e_high_%.2e_unmasked.npy' %(std_low, std_high))
        # test_data_noisy_rand = np.load(path+'polymer_binary_test_noise_std_rand_low_%.2e_high_%.2e_unmasked.npy' %(std_low, std_high))
        # f_dynamic_noisy_rand = np.load(path+'polymer_binary_dynamic_noise_std_rand_low_%.2e_high_%.2e_unmasked.npy' %(std_low, std_high))

    elif obj_type == 'polymer_binary_normalized':
        max_dens_walnut = 0.07184881716966625

        train_data_clean = np.load(path + 'polymer_binary_train_gt_unmasked.npy') * max_dens_walnut
        test_data_clean = np.load(path + 'polymer_binary_test_gt_unmasked.npy') * max_dens_walnut
        f_dynamic_clean = np.load(path + 'polymer_binary_dynamic_gt_unmasked.npy') * max_dens_walnut

        # train_data_noisy_rand = np.load(path+'polymer_binary_train_noise_std_rand_low_%.2e_high_%.2e_unmasked.npy' %(std_low, std_high)) * max_dens_walnut
        # test_data_noisy_rand = np.load(path+'polymer_binary_test_noise_std_rand_low_%.2e_high_%.2e_unmasked.npy' %(std_low, std_high)) * max_dens_walnut
        # f_dynamic_noisy_rand = np.load(path+'polymer_binary_dynamic_noise_std_rand_low_%.2e_high_%.2e_unmasked.npy' %(std_low, std_high)) * max_dens_walnut

    elif obj_type == 'polymer':
        train_data_clean = np.load(path + 'polymer_train_gt_unmasked.npy')
        test_data_clean = np.load(path + 'polymer_test_gt_unmasked.npy')
        f_dynamic_clean = np.load(path + 'polymer_test_gt_unmasked.npy')

        # train_data_noisy_rand = np.load(path+'polymer_train_noise_std_rand_low_%.2e_high_%.2e_unmasked.npy' %(std_low, std_high))
        # test_data_noisy_rand = np.load(path+'polymer_test_noise_std_rand_low_%.2e_high_%.2e_unmasked.npy' %(std_low, std_high))
        # f_dynamic_noisy_rand = np.load(path+'polymer_test_noise_std_rand_low_%.2e_high_%.2e_unmasked.npy' %(std_low, std_high))

    return (
        train_data_clean,
        test_data_clean,
        f_dynamic_clean,
    )  # train_data_noisy_rand, test_data_noisy_rand, f_dynamic_noisy_rand


parser = argparse.ArgumentParser()
parser.add_argument(
    '--obj_type', type=str, required=True
)  # ['walnut', 'walnut_normalized', 'polymer_binary', 'polymer']
parser.add_argument('--std_high', type=float, required=True)
parser.add_argument('--std_low', type=float, required=False, default=0.0)
parser.add_argument('--noise_std', type=float, required=False, default=2e-2)
parser.add_argument('--spatial_dim', type=int, required=False, default=128)
parser.add_argument(
    '--denoiser_type', type=str, required=True
)  # ['dncnn', 'dncnn_oracle', 'dncnn_deblur']
parser.add_argument('--blur', type=bool, required=False, default=False)
parser.add_argument('--rotate', type=bool, required=False, default=False)
parser.add_argument('--limited_view', type=bool, required=False, default=False)
parser.add_argument('--lambda_jr', type=float, required=False, default=0.0)
parser.add_argument('--numLayers', type=int, required=False, default=6)
parser.add_argument('--numChannels', type=int, required=False, default=64)
parser.add_argument('--filterSize', type=int, required=False, default=3)
parser.add_argument('--noise_est_type', type=str, required=False, default='direct')
parser.add_argument(
    '--path',
    type=str,
    required=False,
    default='/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/obj_domain_psm/data/denoiser/',
)
parser.add_argument(
    '--save_fig_path',
    type=str,
    required=False,
    default='/home/berk/Desktop/spatio_temporal/2D_time_variant_tomography/2D_neural_fields/own_imp/paper_figures/denoiser/',
)
parser.add_argument('--lr', type=float, required=False, default=5e-4)
parser.add_argument('--Nepoch', type=int, required=False, default=500)
parser.add_argument('--device', type=int, required=False, default=0)


if __name__ == '__main__':
    main()
