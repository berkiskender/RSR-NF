import itertools

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct

import utils
import plots

from collections import OrderedDict


class dncnn(torch.nn.Module):
    """
    DnCNN denoiser for RED-PSM framework. 
    """
    
    def __init__(self, numLayers, numChannels, filterSize, est_type='direct'):
        """
        Initializes DnCNN denoiser model.

        Parameters:
        -----------
        numLayers (int): Number of layers.
        numChannels (int): Number of convolutional channels per layer.
        filterSize (int): Filter size per convolutional filter.
        est_type (int): Denoising type. 'direct' or 'residual'.
        """
        super(dncnn, self).__init__()
        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=numChannels,
                      kernel_size=filterSize, padding=1), nn.ReLU())
        layers = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels=numChannels, out_channels=numChannels, 
                    kernel_size=filterSize, padding=1),
                nn.ReLU()) for i in range(numLayers)]
        self.main = nn.Sequential(*layers)
        self.final_layer = nn.Conv2d(
            in_channels=numChannels, out_channels=1, kernel_size=filterSize,
            padding=1)
        self.est_type = est_type
        
    def forward(self, x):
        """
        Performs denoising of the noisy input x.

        Parameters:
        ----------
        x (torch.Tensor): Noisy input frame. 
            Shape: [bs, num_ch, spatial_dim, spatial_dim]

        Returns:
        ----------
        output (torch.Tensor): Denoised image or the estimated noise. 
            Shape: [spatial_dim, spatial_dim]
        """
        if self.est_type == 'direct':
            return self.final_layer(self.main(self.init_layer(x)))
        elif self.est_type == 'residual':
            return x - self.final_layer(self.main(self.init_layer(x)))


class dncnnPatchBased(torch.nn.Module):
    """
    Patch-based DnCNN denoiser for RED-PSM framework. 
    """
    
    def __init__(self, numLayers, numChannels, filterSize, pSize,
                 pStride, spatial_dim, est_type='direct'):
        super(dncnnPatchBased, self).__init__()
        
        self.pSize = pSize
        self.filterSize = filterSize
        self.pStride = pStride
        self.est_type = est_type
        
        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=numChannels, 
                      kernel_size=filterSize, padding=1), nn.ReLU())
        layers = [nn.Sequential(
            nn.Conv2d(
                in_channels=numChannels, out_channels=numChannels, 
                kernel_size=filterSize, padding=1), 
            nn.ReLU()) for i in range(numLayers)]
        self.main = nn.Sequential(*layers)
        self.final_layer = nn.Conv2d(in_channels=numChannels, out_channels=1, 
                                     kernel_size=filterSize, padding=1)
        
        self.fold_params = dict(kernel_size=[pSize, pSize], stride=pStride)
        self.fold = nn.Fold(
            output_size=[spatial_dim, spatial_dim], **self.fold_params)
        self.unfold = nn.Unfold(**self.fold_params)
        
    def forward(self, x):
        patches = self.unfold(x).permute(0,2,1)
        patches = patches.contiguous().view(
            x.shape[0]*patches.shape[1], 1, self.pSize, self.pSize)
        patches = self.final_layer(self.main(self.init_layer(patches)))
        patches = patches.contiguous().view(
            x.shape[0], patches.shape[0]//x.shape[0],
            self.pSize*self.pSize).permute(0,2,1)
        if self.est_type == 'direct':
            return self.fold(patches)
        elif self.est_type == 'residual':
            return x - self.fold(patches)
    

class dncnnPatchBased_patchLoss(torch.nn.Module):
    """
    Patch-based DnCNN denoiser for RED-PSM with patch-wise denoised output.
    """
    
    def __init__(self, numLayers, numChannels, filterSize, est_type='direct'):
        super(dncnnPatchBased_patchLoss, self).__init__()
        self.filterSize = filterSize
        self.est_type = est_type
        self.init_layer = nn.Sequential(nn.Conv2d(1, numChannels,
                                                  filterSize, 1), nn.ReLU())
        layers = [nn.Sequential(nn.Conv2d(
            numChannels, numChannels, filterSize, 1),
                                nn.ReLU()) for i in range(numLayers)]
        self.main = nn.Sequential(*layers)
        self.final_layer = nn.Conv2d(in_channels=numChannels, out_channels=1,
                                     kernel_size=filterSize, padding=1)
        
    def forward(self, patches):
        if self.est_type == 'direct':
            return self.final_layer(self.main(self.init_layer(patches)))
        elif self.est_type == 'residual':
            return patches - self.final_layer(
                self.main(self.init_layer(patches)))


class RedPsm(nn.Module):
    """
    RED-PSM model class initializing temporal latent representations,
        spatial and temporal basis functions, and the full-rank object f. 
        Supports multiple measurements at a given time instant.
    """
    
    def __init__(self, P, K, spatial_dim, z_dim, temporal_basis, obj_type='walnut',
                 temp_init_type='random', f_init_type='random', 
                 spatial_init_type='random', temporal_mode='z', 
                 noise_std=0, mask=False, rep=4):
        """
        Initializes the RED-PSM model.

        Parameters:
        ----------
        P (int): Number of measurements.
        K (int): PSM order.
        spatial_dim (int): Spatial dimension.
        z_dim (int): Temporal latent representation dimension.
        temporal_basis (str): Basis to map latent representations to temporal 
            basis functions.
        obj_type (str): Object type for initialization.
        temp_init_type (str): Temporal basis initialization type.
        f_init_type (str): Initialization type for object f.
        spatial_init_type (str): Spatial basis initialization type.
        temporal_mode (str): Learn P-dim temporal basis fcts (Psi) or d-dim
            latent z.
        noise_std (float): Measurement noise std for correct initialization.
        mask (bool): If True, apply FOV mask for tomographic objects.
        rep (int): Number of simultaneous measurements/projections.
        """
        super(RedPsm, self).__init__()

        self.P, self.K, self.z_dim = P, K, z_dim
        self.temporal_basis = temporal_basis
        self.f_init_type = f_init_type
        self.spatial_init_type = spatial_init_type
        self.temporal_mode = temporal_mode
        self.rep = rep
                        
        # Cubic spline interpolation
        self.D_t = torch.linspace(0, 1, self.z_dim).cuda()
        self.P_t = torch.linspace(0, 1, self.P//self.rep).cuda()
        
        # self.DCT_U = torch.fft.fft(torch.eye(P)).cuda()
        self.DCT_U = dct.dct(torch.eye(P)).cuda()  # DCT-II done through the last dimension
        self.DCT_U /= (self.DCT_U[0, :] @ self.DCT_U[0, :])**0.5
        
        # plt.imshow(torch.real(self.DCT_U).detach().cpu())
        
        if temporal_basis in ['linear', 'spline', 'dct']:
            if temporal_mode == 'z':
                self.psi_mtx = torch.zeros([self.K + 1, self.P//self.rep],
                                           dtype=torch.float).cuda()
                if temp_init_type == 'random':
                    self.temporal_fcts = torch.autograd.Variable(
                        torch.randn(1, K + 1, z_dim).cuda(), requires_grad=True)
                elif temp_init_type == 'learned':
                    self.temp_fcts = torch.randn(1, K + 1, z_dim)
                    # self.temp_fcts = torch.load(
                    #     'data/temporal_latent_fcts_est'
                    #     '_%s_%s_P_%d_K_%d_L_out_%d_noise_std_%.2e.pt' %(
                    #     temporal_basis, obj_type, P, K, z_dim, noise_std))
                    self.temp_fcts = torch.load(
                        'data/temporal_latent_fcts_est'
                        '_%s_%s_P_%d_K_%d_L_out_%d_noise_std_%.2e.pt' %(
                        temporal_basis, obj_type, P, K, z_dim, noise_std))
                    self.temporal_fcts = torch.randn(1, K + 1, z_dim).cuda()
                    for k in range(self.K + 1):
                        self.temporal_fcts[:, k, :] = F.interpolate(
                            self.temp_fcts[:, k, :].view(
                                1, 1, self.temp_fcts.shape[2]), 
                            size=self.z_dim, mode='linear')
                    self.temporal_fcts = torch.autograd.Variable(
                        self.temporal_fcts, requires_grad=True)
                else:
                    raise NotImplementedError(
                        'temp_init_type not implemented.')
            elif temporal_mode == 'psi':
                self.temporal_fcts = torch.autograd.Variable(
                    torch.randn(1, K + 1, z_dim).cuda(), requires_grad=True)
                self.psi_mtx = torch.zeros([self.K + 1, self.P//self.rep],
                                           dtype=torch.float).cuda()

        # FOV mask
        self.mask = torch.autograd.Variable(
            torch.ones(spatial_dim, spatial_dim).cuda(), requires_grad=False)
        if mask:
            for i,j in list(itertools.product(
                np.arange(spatial_dim), np.arange(spatial_dim))):
                if (i-spatial_dim//2)**2 + (
                    j-spatial_dim//2)**2 >= (spatial_dim//2)**2:
                    self.mask[i,j] = 0
        
        # Spatial basis functions
        if spatial_init_type == 'random':
            self.spatial_basis_fcts = torch.autograd.Variable(
                torch.zeros(
                    K + 1, spatial_dim, spatial_dim).cuda(), requires_grad=True)
        elif spatial_init_type == 'learned':
            self.spatial_basis_fcts = torch.zeros(
                K + 1, spatial_dim, spatial_dim).cuda()
            self.spatial_basis_fcts[:K+1] = torch.load(
                'data/spatial_basis_fcts_est_%s_%s_P_%d_noise_std_%.2e.pt' %(
                temporal_basis, obj_type, P, noise_std)).cuda()[:, :, :K+1].permute(
                2, 0, 1)
            self.spatial_basis_fcts = torch.autograd.Variable(
                self.spatial_basis_fcts, requires_grad=True)
        else:
            raise NotImplementedError('Spatial basis type not implemented.')
        
        f_est = torch.einsum(
            'kp,kjs->pjs', self.psi_mtx, self.mask * self.spatial_basis_fcts)
        self.f_est = torch.autograd.Variable(
            (self.mask * f_est), requires_grad=True)


    def forward(self):
        """
        Computes the PSM estimate of the dynamic object.

        Parameters:
        ----------
        None.

        Returns:
        ----------
        psi_mtx (torch.Tensor): Temporal basis functions. Shape: [K, P]
        f_psm_est (torch.Tensor): PSM estimate of the object f. 
            Shape: [spatial_dim, spatial_dim, P]
        """
        if self.temporal_mode == 'z':
            psi_mtx = generate_psi_from_z(
                self.K, self.P//self.rep, self.temporal_fcts, self.z_dim, 
                self.temporal_basis, self.D_t, self.P_t, self.DCT_U,
                gen_temp=None)
        elif self.temporal_mode == 'psi':
            psi_mtx = self.psi_mtx
        else:
            raise NotImplementedError('Temporal mode not implemented.')
        f_psm_est = torch.einsum(
            'kp,kjs->pjs', psi_mtx, self.mask * self.spatial_basis_fcts)
        f_psm_est = (self.mask * f_psm_est).permute(1,2,0)
        return psi_mtx, f_psm_est

    
class plinear_denoiser(torch.nn.Module):
    """
    Pseudo-linear denoiser for RED-PSM framework. 
    """
    
    def __init__(self, numLayers, latent_dim, spatial_dim, est_type='direct'):
        """
        Initializes Pseudo-linear denoiser model.

        Parameters:
        ----------
        numLayers (int): Number of layers.
        latent_dim (int): Number of convolutional channels per layer.
        est_type (int): Denoising type. 'direct' or 'residual'.
        """
        super(plinear_denoiser, self).__init__()
        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=numChannels,
                      kernel_size=filterSize, padding=1), nn.ReLU())
        
        self.encoder = nn.Sequential()
        self.encoder.add_module(nn.Linear(spatial_dim**2, latent_dim, bias=True))
        self.encoder.add_module(nn.ReLU())
        for i in range(numLayers-1):
            self.encoder.add_module(nn.Linear(latent_dim, latent_dim, bias=True))
            self.encoder.add_module(nn.ReLU())
        self.encoder.add_module(nn.Linear(latent_dim, latent_dim*spatial_dim**2,
                                          bias=True))
        self.encoder.add_module(nn.ReLU())
            
        self.decoder = nn.Sequential()
        self.decoder.add_module(nn.Linear(spatial_dim**2, latent_dim, bias=True))
        self.decoder.add_module(nn.ReLU())
        for i in range(numLayers-1):
            self.decoder.add_module(nn.Linear(latent_dim, latent_dim, bias=True))
            self.decoder.add_module(nn.ReLU())
        self.decoder.add_module(nn.Linear(latent_dim, latent_dim*spatial_dim**2,
                                          bias=True))
        self.decoder.add_module(nn.ReLU())
        
        self.spatial_dim = spatial_dim
        self.latent_dim = latent_dim
        
        
    def forward(self, x):
        """
        Performs denoising of the noisy input x.

        Parameters:
        ----------
        x (torch.Tensor): Noisy input frame. 
            Shape: [bs, num_ch, spatial_dim, spatial_dim]

        Returns:
        ----------
        output (torch.Tensor): Denoised image or the estimated noise. 
            Shape: [spatial_dim, spatial_dim]
        """
        
        A1 = self.encoder(x).view(self.latent_dim, self.spatial_dim**2)
        A2 = self.decoder(x).view(self.spatial_dim**2, self.latent_dim)
        
        return A2 @ A1 @ x


class dncnn_plinear(torch.nn.Module):
    """
    plinear DnCNN denoiser for RED-PSM framework. 
    """
    def __init__(self, numLayers, numChannels, filterSize, est_type='direct'):
        """
        Initializes DnCNN denoiser model.

        Parameters:
        ----------
        numLayers (int): Number of layers.
        numChannels (int): Number of convolutional channels per layer.
        filterSize (int): Filter size per convolutional filter.
        est_type (int): Denoising type. 'direct' or 'residual'.
        """
        super(dncnn_plinear, self).__init__()
        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=numChannels,
                      kernel_size=filterSize, padding=1), nn.ReLU())
        layers = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels=numChannels, out_channels=numChannels,
                    kernel_size=filterSize, padding=1),
                nn.ReLU()) for i in range(numLayers)]
        self.main = nn.Sequential(*layers)
        self.final_layer = nn.Conv2d(
            in_channels=numChannels, out_channels=1, kernel_size=filterSize,
            padding=1)
        # self.final_nl_layer = nn.Sigmoid()
        self.est_type = est_type
        
    def forward(self, x):
        """
        Performs denoising of the noisy input x.

        Parameters:
        ----------
        x (torch.Tensor): Noisy input frame. 
            Shape: [bs, num_ch, spatial_dim, spatial_dim]

        Returns:
        ----------
        output (torch.Tensor): Denoised image or the estimated noise. 
            Shape: [spatial_dim, spatial_dim]
        """
        if self.est_type == 'direct':
            Wx = self.final_layer(self.main(self.init_layer(x)))
            return Wx * x
        elif self.est_type == 'residual':
            Wx = self.final_layer(self.main(self.init_layer(x)))
            return x - Wx * x
        else:
            raise NotImplementedError


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
