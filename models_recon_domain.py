import itertools

import numpy as np
from matplotlib import pyplot as plt
from torch.nn.functional import relu, interpolate
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_cubic_spline_interp
import torch_dct as dct 

import utils_misc
import plots


class TempGeneratorConvTrans(torch.nn.Module):
    def __init__(self, latent_dim, K, P):
        super(TempGeneratorConvTrans, self).__init__()
        
        self.main = torch.nn.Sequential(
            nn.ConvTranspose1d(K+1, (K+1), kernel_size=4, stride=2, dilation=1, 
                               padding=1, output_padding=0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose1d((K+1), (K+1), kernel_size=4, stride=2, 
                               dilation=1, padding=1, output_padding=0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose1d((K+1), (K+1), kernel_size=4, stride=2, 
                               dilation=1, padding=1, output_padding=0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose1d((K+1), (K+1), kernel_size=4, stride=2, 
                               dilation=1, padding=1, output_padding=0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose1d((K+1), K+1, kernel_size=4, stride=2, dilation=1, 
                               padding=1, output_padding=0, bias=False),
            # nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)   


class SpatialBasisRecon(nn.Module):
    def __init__(self, P, K, K_extra, spatial_dim, num_instances, z_dim,
                 learning_mode, Psi=None, obj_type='walnut', 
                 temp_fcts_type='random', f_est_type='random', 
                 spatial_mode='full_basis', 
                 # z_dim=16
                ):
        super(SpatialBasisRecon, self).__init__()

        self.relu = nn.ReLU()
        self.P = P
        self.K = K
        self.z_dim = z_dim
        self.num_instances = num_instances
        self.learning_mode = learning_mode

        self.spatial_mode = spatial_mode
        self.K_extra = K_extra

        # Ground truth Casorati temporal functions for Psi
        self.U_casorati = torch.Tensor(Psi[:,:z_dim]).cuda()
        if K_extra > 0:
            self.U_casorati_extra = torch.Tensor(Psi[:,z_dim:z_dim+K_extra+1]).cuda()

        if learning_mode in ['linear', 'spline', 'casorati']:
            if temp_fcts_type == 'random':
                self.temporal_fcts = torch.autograd.Variable(torch.randn(1, K + 1, z_dim).cuda(), requires_grad=True)
            if temp_fcts_type == 'learned':
                self.temp_fcts = torch.load('data/temp_fcts_%s_%s_P_%d.pt' %(learning_mode, obj_type, P))
                self.temporal_fcts = torch.randn(1, K + 1, z_dim).cuda()
                for k in range(self.K + 1):
                    self.temporal_fcts[:, k, :] = interpolate(
                        self.temp_fcts[:, k, :].view(1, 1, self.temp_fcts.shape[2]), size=self.z_dim, mode='linear')
                self.temporal_fcts = torch.autograd.Variable(self.temporal_fcts, requires_grad=True)

            if K_extra > 0:
                self.temp_fcts_extra = torch.autograd.Variable(torch.randn(1, K_extra + 1, z_dim).cuda(), requires_grad=True)
                self.temporal_fcts_extra = torch.randn(1, K_extra + 1, z_dim).cuda()
                for k in range(self.K_extra + 1):
                    self.temporal_fcts_extra[:, k, :] = interpolate(
                        self.temp_fcts_extra[:, k, :].view(
                            1, 1, self.temp_fcts_extra.shape[2]), size=self.z_dim, mode='linear')
                self.temporal_fcts_extra = torch.autograd.Variable(self.temporal_fcts_extra, requires_grad=True)

            self.mask = torch.autograd.Variable(torch.ones(spatial_dim, spatial_dim).cuda(), requires_grad=False)
            for i in range(spatial_dim):
                for j in range(spatial_dim):
                    if (i-spatial_dim//2)**2 + (j-spatial_dim//2)**2 >= (spatial_dim//2)**2:
                        self.mask[i,j] = 0

        # Cubic spline interpolation
        self.D_t = torch.linspace(0, 1, self.z_dim).cuda()
        self.P_t = torch.linspace(0, 1, self.P).cuda()
        
        if spatial_mode == 'dip':
            ngf = spatial_dim
            nc = 1
            self.generator = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(z_dim, ngf * 4 * (nc+1)//1, 4, 1, 0, bias=False),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf * 4 * (nc+1)//1, ngf * 2 * (nc+1)//1, 4, 2,1, bias=False),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf * 2 * (nc+1)//1, ngf * 1 * (nc+1)//1, 4, 2, 1, bias=False),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf * 1 * (nc+1)//1, ngf * (nc+1)//2, 4, 2, 1, bias=False),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf*(nc+1)//2, ngf*(nc+1)//4, 4, 2, 1, bias=False),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf*(nc+1)//4, nc, 4, 2, 1, bias=False),
            torch.nn.Tanh()
            )
            self.spatial_basis_fcts = torch.autograd.Variable(torch.randn(K + 1, z_dim, 1, 1).cuda(), requires_grad=True)
            if K_extra > 0:
                self.spatial_basis_fcts_extra = torch.autograd.Variable(
                    torch.randn(K_extra + 1, z_dim, 1, 1).cuda(), requires_grad=True)
            gen_total_param = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
            print(self.generator, gen_total_param)
        elif spatial_mode == 'full_basis':
            # f_est
            if f_est_type == 'random':
                self.spatial_basis_fcts = torch.autograd.Variable(
                    torch.randn(K + 1, spatial_dim, spatial_dim).cuda(), requires_grad=True)
            elif f_est_type == 'learned':
                self.spatial_basis_fcts = torch.autograd.Variable(
                    torch.load('data/spatial_basis_est_%s_P_%d.pt' %(obj_type, P)).cuda(), requires_grad=True)
            if K_extra > 0:
                self.spatial_basis_fcts_extra = torch.autograd.Variable(
                    torch.randn(K_extra + 1, spatial_dim, spatial_dim).cuda(), requires_grad=True)

    def forward(self):
        psi_mtx = torch.zeros([self.K + 1, self.P], dtype=torch.float).cuda()
        if self.learning_mode == 'linear':
            for k in range(self.K + 1):
                psi_mtx[k, :] = interpolate(self.temporal_fcts[:, k, :].view(1, 1, self.z_dim), size=self.P, mode=self.learning_mode)
        elif self.learning_mode == 'spline':
            for k in range(self.K + 1):
                psi_mtx[k, :] = torch_cubic_spline_interp.interp(self.D_t, self.temporal_fcts[:, k, :].view(self.z_dim),
                                                     self.P_t).view(1, 1, self.P)
        elif self.learning_mode == 'casorati':
            for k in range(self.K+1):
                psi_mtx[k, :] = (self.U_casorati @ self.temporal_fcts[:, k, :].view(self.z_dim)).view(1, 1, self.P)
                
        if self.K_extra > 0:
            psi_mtx_extra = torch.zeros([self.K_extra + 1, self.P], dtype=torch.float).cuda()
            if self.learning_mode == 'linear':
                for k in range(self.K_extra + 1):
                    psi_mtx_extra[k, :] = interpolate(
                        self.temporal_fcts_extra[:, k, :].view(1, 1, self.z_dim), size=self.P, mode=self.learning_mode)
            elif self.learning_mode == 'spline':
                for k in range(self.K_extra + 1):
                    psi_mtx_extra[k, :] = torch_cubic_spline_interp.interp(
                        self.D_t, self.temporal_fcts_extra[:, k, :].view(self.z_dim), self.P_t).view(1, 1, self.P)
            elif self.learning_mode == 'casorati':
                for k in range(self.K_extra + 1):
                    psi_mtx_extra[k, :] = (
                        self.U_casorati_extra @ self.temporal_fcts_extra[:, k, :].view(self.z_dim)).view(1, 1, self.P)
                        
        if self.spatial_mode == 'dip':
            spatial_basis_fcts = self.generator(self.spatial_basis_fcts).squeeze()
            f_est = torch.einsum('kp,kjs->pjs', psi_mtx, spatial_basis_fcts)
            if self.K_extra > 0:
                spatial_basis_fcts_extra = self.generator(self.spatial_basis_fcts_extra).squeeze()
                f_est += torch.einsum('kp,kjs->pjs', psi_mtx_extra, spatial_basis_fcts_extra)
            f_est = (self.mask * f_est).permute(1,2,0)

        elif self.spatial_mode == 'full_basis':
            f_est = torch.einsum('kp,kjs->pjs', psi_mtx, self.spatial_basis_fcts)
            if self.K_extra > 0:
                f_est += torch.einsum('kp,kjs->pjs', psi_mtx_extra, self.spatial_basis_fcts_extra)
            f_est = (self.mask * f_est).permute(1,2,0)

        return psi_mtx, f_est

    
def spatial_generator(z_dim, ngf, nc):
    return torch.nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 4 * (nc+1)//1, 4, 1, 0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4 * (nc+1)//1, ngf * 2 * (nc+1)//1, 4, 2,1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2 * (nc+1)//1, ngf * 1 * (nc+1)//1, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 1 * (nc+1)//1, ngf * (nc+1)//2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*(nc+1)//2, ngf*(nc+1)//4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*(nc+1)//4, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            )


class SpatialBasisReconTempReg(nn.Module):
    """ 
    RED-PSM model class initializing temporal latent representations, 
    spatial and temporal basis functions, and the full-rank object f. 
    """
    
    def __init__(self, P, K, K_extra, spatial_dim, num_instances, z_dim,
                 learning_mode, Psi=None, obj_type='walnut',
                 temp_fcts_type='random', f_est_type='random',
                 spatial_basis_type='random', spatial_mode='full_basis',
                 # z_dim=16, 
                 temporal_mode='z', noise_std=0, temp_reflect=None,
                 mask=False):
        super(SpatialBasisReconTempReg, self).__init__()

        self.P, self.K, self.K_extra, self.z_dim, self.num_instances = P, K, K_extra, z_dim, num_instances
        self.learning_mode, self.spatial_mode = learning_mode, spatial_mode
        self.f_est_type, self.spatial_basis_type, self.temporal_mode = f_est_type, spatial_basis_type, temporal_mode
        
        # Ground truth Casorati temporal functions for Psi
        self.U_casorati = torch.Tensor(Psi[:, :z_dim]).cuda()
        
        # Cubic spline interpolation
        self.D_t = torch.linspace(0, 1, self.z_dim).cuda()
        self.P_t = torch.linspace(0, 1, self.P).cuda()
        
        # self.DCT_U = torch.fft.fft(torch.eye(P)).cuda()
        self.DCT_U = dct.dct(torch.eye(P)).cuda()  # DCT-II done through the last dimension
        self.DCT_U /= (self.DCT_U[0, :] @ self.DCT_U[0, :])**0.5
            
        if learning_mode in ['linear', 'spline', 'casorati', 'dct']:
            
            if temporal_mode == 'z':
                self.psi_mtx = torch.zeros([self.K + self.K_extra + 1, self.P], dtype=torch.float).cuda()
                if temp_fcts_type == 'random':
                    self.temporal_fcts = torch.autograd.Variable(torch.randn(1, K + K_extra + 1, z_dim).cuda(), requires_grad=True)
                elif temp_fcts_type == 'learned':
                    self.temp_fcts = torch.randn(1, K + K_extra + 1, z_dim)  # Prosep learned temporal fcts of dim d = z_dim
                    if temp_reflect == None:
                        self.temp_fcts[:, :K+1, :] = torch.load('data/temporal_latent_fcts_est_%s_%s_P_%d_K_%d_L_out_%d_noise_std_%.2e.pt' %(
                            learning_mode, obj_type, P, K, z_dim, noise_std))
                    elif temp_reflect == 'end':
                        temp = torch.load('data/temporal_latent_fcts_est_%s_%s_P_%d_K_%d_L_out_%d_noise_std_%.2e.pt' %(
                            learning_mode, obj_type, P//2, K, z_dim//2, noise_std))
                        print(temp.shape, torch.flip(temp, dims=[2]).shape, torch.cat((temp, torch.flip(temp, dims=[2])), dim=2).shape)
                        self.temp_fcts[:, :K+1, :] = torch.cat((temp, torch.flip(temp, dims=[2])), dim=2)
                    self.temporal_fcts = torch.randn(1, K + K_extra + 1, z_dim).cuda()
                    for k in range(self.K + self.K_extra + 1):
                        self.temporal_fcts[:, k, :] = interpolate(self.temp_fcts[:, k, :].view(1, 1, self.temp_fcts.shape[2]), size=self.z_dim, mode='linear')
                    self.temporal_fcts = torch.autograd.Variable(self.temporal_fcts, requires_grad=True)
                elif temp_fcts_type == 'casorati':
                    self.temp_fcts = torch.randn(1, K + K_extra + 1, z_dim)  # Prosep learned temporal fcts of dim d = K+1
                    self.temp_fcts[:, :K+K_extra+1, :] = torch.load('data/casorati_temporal_latent_fcts_est_%s_%s_P_%d_L_out_%d_noise_std_%.2e.pt' %(
                        learning_mode, obj_type, P, z_dim, noise_std))[:, :K+K_extra+1, :]
                    self.temporal_fcts = torch.randn(1, K + K_extra + 1, z_dim).cuda()
                    for k in range(self.K + self.K_extra + 1):
                        self.temporal_fcts[:, k, :] = interpolate(self.temp_fcts[:, k, :].view(1, 1, self.temp_fcts.shape[2]), size=self.z_dim, mode=self.learning_mode)
                    self.temporal_fcts = torch.autograd.Variable(self.temporal_fcts, requires_grad=True)
                else:
                    raise NotImplementedError('temp_fcts_type not implemented.')

            elif temporal_mode == 'psi':
                self.temporal_fcts = None
                self.psi_mtx = torch.randn([K + K_extra + 1, self.P], dtype=torch.float).cuda()
                if temp_fcts_type == 'learned':
                    if self.learning_mode == 'linear':
                        for k in range(self.K + self.K_extra + 1):
                            self.psi_mtx[k, :] = interpolate(
                                self.temporal_fcts[:, k, :].view(1, 1, self.z_dim), size=self.P, mode=self.learning_mode)
                    elif self.learning_mode == 'spline':
                        for k in range(self.K + self.K_extra + 1):
                            self.psi_mtx[k, :] = torch_cubic_spline_interp.interp(self.D_t, self.temporal_fcts[:, k, :].view(self.z_dim),
                                                                 self.P_t).view(1, 1, self.P)
                    elif self.learning_mode == 'dct':
                        for k in range(self.K + self.K_extra + 1):
                            self.psi_mtx[k, :] = (self.DCT_U[:, :self.z_dim] @ self.temporal_fcts[:, k, :].view(self.z_dim)).view(1, 1, self.P)
                elif temp_fcts_type == 'casorati':
                    self.psi_mtx[:K+K_extra+1,:] = torch.load('data/temporal_basis_fcts_est_%s_%s_P_%d_noise_std_%.2e.pt' %(
                            'casorati', obj_type, P, noise_std))[:K+K_extra+1,:].cuda()
                self.psi_mtx = torch.autograd.Variable(self.psi_mtx, requires_grad=True)
            
        elif learning_mode == 'gen' and temporal_mode == 'z':
            self.temporal_fcts = torch.autograd.Variable(torch.randn(1, K + K_extra + 1, z_dim).cuda(), requires_grad=True)
            self.gen_temp = TempGeneratorConvTrans(z_dim, K, P)

        # FOV mask
        self.mask = torch.autograd.Variable(torch.ones(spatial_dim, spatial_dim).cuda(), requires_grad=False)
        if mask:
            for i,j in list(itertools.product(np.arange(spatial_dim), np.arange(spatial_dim))):
                if (i-spatial_dim//2)**2 + (j-spatial_dim//2)**2 >= (spatial_dim//2)**2:
                    self.mask[i,j] = 0
        
        # Spatial basis functions
        if spatial_mode == 'dip':
            self.generator = spatial_generator(z_dim, ngf=spatial_dim, nc=1)
            self.spatial_basis_fcts = torch.autograd.Variable(torch.randn(K + K_extra + 1, z_dim, 1, 1).cuda(), requires_grad=True)
            gen_total_param = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
            print(self.generator, gen_total_param)
        elif spatial_mode == 'full_basis':
            if spatial_basis_type == 'random':
                self.spatial_basis_fcts = torch.autograd.Variable(
                    torch.zeros(K + K_extra + 1, spatial_dim, spatial_dim).cuda(), requires_grad=True)
            elif spatial_basis_type == 'learned':
                self.spatial_basis_fcts = torch.zeros(K + K_extra + 1, spatial_dim, spatial_dim).cuda()
                self.spatial_basis_fcts[:K+1, :, :] = torch.load('data/spatial_basis_fcts_est_%s_%s_P_%d_noise_std_%.2e.pt' %(
                    'spline', obj_type, P, noise_std)).cuda()[:,:,:K+1].permute(2,0,1)
                self.spatial_basis_fcts = torch.autograd.Variable(self.spatial_basis_fcts, requires_grad=True)
            elif spatial_basis_type == 'casorati':
                self.spatial_basis_fcts = torch.zeros(K + K_extra + 1, spatial_dim, spatial_dim).cuda()
                self.spatial_basis_fcts[:K+1, :, :] = torch.load('data/spatial_basis_fcts_est_%s_%s_P_%d_noise_std_%.2e.pt' %(
                    'casorati', obj_type, P, noise_std)).cuda()[:,:,:K+K_extra+1].permute(2,0,1)
                self.spatial_basis_fcts = torch.autograd.Variable(self.spatial_basis_fcts, requires_grad=True)
            else:
                raise NotImplementedError('Spatial basis type for full basis mode not implemented.')
        else:
            raise NotImplementedError('Spatial mode not implemented.')
        
        f_est = torch.einsum('kp,kjs->pjs', self.psi_mtx, self.mask * self.spatial_basis_fcts)
        self.f_est = torch.autograd.Variable((self.mask * f_est), requires_grad=True)
        
        # # Plot initial spatial and temporal basis function estimates
        # plots.plot_init_estimates(K, K_extra, z_dim, P, self.spatial_basis_fcts,
        #                           self.temporal_fcts, self.D_t, self.P_t, self.mask)

    def forward(self):
        if self.temporal_mode == 'z':
            # psi_mtx = generate_psi_from_z(self.K, self.K_extra, self.P, self.temporal_fcts, self.z_dim,
            #                               self.temporal_basis, self.D_t, self.P_t, self.U_casorati, gen_temp=None)
            psi_mtx = generate_psi_from_z(
                self.K, self.P, self.temporal_fcts, self.z_dim, 
                self.learning_mode, self.D_t, self.P_t, self.DCT_U,
                gen_temp=None)
        elif self.temporal_mode == 'psi':
            psi_mtx = self.psi_mtx
        else:
            raise NotImplementedError('Temporal mode not implemented.')

        if self.spatial_mode == 'dip':
            spatial_basis_fcts = self.generator(self.mask * self.spatial_basis_fcts).squeeze()
            f_psm_est = torch.einsum('kp,kjs->pjs', psi_mtx, spatial_basis_fcts)
            f_psm_est = (self.mask * f_psm_est).permute(1,2,0)
        elif self.spatial_mode == 'full_basis':
            f_psm_est = torch.einsum('kp,kjs->pjs', psi_mtx, self.mask * self.spatial_basis_fcts)
            f_psm_est = (self.mask * f_psm_est).permute(1,2,0)
        else:
            raise NotImplementedError('Spatial mode not implemented.')
        return psi_mtx, f_psm_est


# def generate_psi_from_z(K, K_extra, P, temporal_fcts, z_dim, learning_mode, D_t, P_t, U_casorati=None, gen_temp=None):
#     """
#     Function to compute temporal basis functions Psi from their low-dimensional latent representations Z.
#     """
    
#     psi_mtx = torch.zeros([K + K_extra + 1, P], dtype=torch.float).cuda()
#     if learning_mode == 'linear':
#         for k in range(K + K_extra + 1):
#             psi_mtx[k, :] = interpolate(temporal_fcts[:, k, :].view(1, 1, z_dim), size=P, mode=learning_mode)
#     elif learning_mode == 'spline':
#         for k in range(K + K_extra + 1):
#             psi_mtx[k, :] = torch_cubic_spline_interp.interp(D_t, temporal_fcts[:, k, :].view(z_dim),
#                                                              P_t).view(1, 1, P)
#     elif learning_mode == 'casorati':
#         for k in range(K + K_extra + 1):
#             psi_mtx[k, :] = (U_casorati @ temporal_fcts[:, k, :].view(z_dim)).view(1, 1, P)
#     elif learning_mode == 'gen':
#         psi_mtx = gen_temp(temporal_fcts[0])
#     else:
#         raise NotImplementedError('Learning mode not implemented.')
#     return psi_mtx


def generate_psi_from_z(K, P, temporal_fcts, z_dim, temporal_basis, D_t, P_t, 
                        DCT_U, gen_temp=None):
    """
    Computes temporal basis Psi from the low-dimensional latent Z.
    """
    
    psi_mtx = torch.zeros([K + 1, P], dtype=torch.float).cuda()
    if temporal_basis == 'linear':
        for k in range(K + 1):
            psi_mtx[k, :] = F.interpolate(
                temporal_fcts[:, k, :].view(1, 1, z_dim), size=P, 
                mode=temporal_basis)
    elif temporal_basis == 'spline':
        for k in range(K + 1):
            psi_mtx[k, :] = torch_cubic_spline_interp.interp(
                D_t, temporal_fcts[:, k, :].view(z_dim), P_t).view(1, 1, P)
    elif temporal_basis == 'gen':
        psi_mtx = gen_temp(temporal_fcts[0])
    elif temporal_basis == 'dct':
        for k in range(K + 1):
            psi_mtx[k, :] = (DCT_U[:, :z_dim] @ temporal_fcts[:, k, :].view(z_dim)).view(1, 1, P)
    else:
        raise NotImplementedError('Learning mode not implemented.')
    return psi_mtx



class dncnn(torch.nn.Module):
    """
    DnCNN denoiser for RED-PSM framework. 
    """
    
    def __init__(self, numLayers, numChannels, filterSize, noise_est_type='direct'):
        super(dncnn, self).__init__()
        self.init_layer = nn.Sequential(nn.Conv2d(
                in_channels=1, out_channels=numChannels, kernel_size=filterSize, padding=1), nn.ReLU())
        layers = [nn.Sequential(nn.Conv2d(
                in_channels=numChannels, out_channels=numChannels, kernel_size=filterSize, padding=1), nn.ReLU()) for i in range(numLayers)]
        self.main = nn.Sequential(*layers)
        self.final_layer = nn.Conv2d(in_channels=numChannels, out_channels=1, kernel_size=filterSize, padding=1)
        self.noise_est_type = noise_est_type
        
    def forward(self, x):
        if self.noise_est_type == 'direct':
            return self.final_layer(self.main(self.init_layer(x)))
        elif self.noise_est_type == 'residual':
            return x - self.final_layer(self.main(self.init_layer(x)))


class dncnnPatchBased(torch.nn.Module):
    """
    Patch-based DnCNN denoiser for RED-PSM framework. 
    """
    
    def __init__(self, numLayers, numChannels, filterSize, patchSize, patchStride, spatial_dim, noise_est_type='direct'):
        super(dncnnPatchBased, self).__init__()
        
        self.patchSize = patchSize
        self.filterSize = filterSize
        self.patchStride = patchStride
        self.noise_est_type = noise_est_type
        
        self.init_layer = nn.Sequential(nn.Conv2d(
                in_channels=1, out_channels=numChannels, kernel_size=filterSize, padding=1), nn.ReLU())
        layers = [nn.Sequential(nn.Conv2d(
                in_channels=numChannels, out_channels=numChannels, kernel_size=filterSize, padding=1), nn.ReLU()) for i in range(numLayers)]
        self.main = nn.Sequential(*layers)
        self.final_layer = nn.Conv2d(in_channels=numChannels, out_channels=1, kernel_size=filterSize, padding=1)
        
        self.fold_params = dict(kernel_size=[patchSize, patchSize], stride=patchStride)
        self.fold = nn.Fold(output_size=[spatial_dim, spatial_dim], **self.fold_params)
        self.unfold = nn.Unfold(**self.fold_params)
        
    def forward(self, x):
        patches = self.unfold(x).permute(0,2,1)
        patches = patches.contiguous().view(x.shape[0]*patches.shape[1], 1, self.patchSize, self.patchSize)
        patches = self.final_layer(self.main(self.init_layer(patches)))
        patches = patches.contiguous().view(x.shape[0], patches.shape[0]//x.shape[0], self.patchSize*self.patchSize).permute(0,2,1)
        if self.noise_est_type == 'direct':
            return self.fold(patches)
        elif self.noise_est_type == 'residual':
            return x - self.fold(patches)












# Patch-based RED-PSM model
class SpatialBasisReconTempRegPatch(nn.Module):
    """ 
    Patch-based RED-PSM model class initializing temporal latent representations, spatial and temporal basis functions,
    and the full-rank object f. Spatial basis functions are patch-based.
    """
    
    def __init__(self, P, K, K_extra, spatial_dim, num_instances, z_dim, learning_mode, Psi=None, 
                 obj_type='walnut', temp_fcts_type='random', f_est_type='random', spatial_basis_type='random',
                 spatial_mode='full_basis', 
                 # z_dim=16, 
                 temporal_mode='z', noise_std=0, patch_size=8):
        super(SpatialBasisReconTempRegPatch, self).__init__()

        self.P = P
        self.K = K
        self.K_extra = K_extra
        self.z_dim = z_dim
        self.num_instances = num_instances
        self.learning_mode = learning_mode
        
        self.spatial_mode = spatial_mode
        self.f_est_type = f_est_type
        self.spatial_basis_type = spatial_basis_type
        
        self.temporal_mode = temporal_mode
        
        # Ground truth Casorati temporal functions for Psi
        self.U_casorati = torch.Tensor(Psi[:,:z_dim]).cuda()
        
        # Cubic spline interpolation
        self.D_t = torch.linspace(0, 1, self.z_dim).cuda()
        self.P_t = torch.linspace(0, 1, self.P).cuda()
        
        self.patch_size = patch_size
        self.num_patches = spatial_dim // self.patch_size
            
        if learning_mode in ['linear', 'spline', 'casorati']:

            if temp_fcts_type == 'random':
                self.temporal_fcts = torch.autograd.Variable(torch.randn(self.num_patches, self.num_patches, K + K_extra + 1, z_dim).cuda(), requires_grad=True)
            else:
                raise NotImplementedError('temp_fcts_type not implemented.')

            if temporal_mode == 'psi':
                self.psi_mtx = torch.randn([self.num_patches, self.num_patches, K + K_extra + 1, self.P], dtype=torch.float).cuda()
                if temp_fcts_type == 'learned':
                    if self.learning_mode == 'linear':
                        for k in range(self.K + self.K_extra + 1):
                            self.psi_mtx[:, :, k, :] = interpolate(
                                self.temporal_fcts[:, :, k, :].view(self.num_patches, self.num_patches, self.z_dim), size=self.P, mode=self.learning_mode)
                    elif self.learning_mode == 'spline':
                        for k in range(self.K + self.K_extra + 1):
                            self.psi_mtx[:, :, k, :] = torch_cubic_spline_interp.interp(self.D_t, self.temporal_fcts[:, :, k, :].view(self.num_patches, self.num_patches, self.z_dim),
                                                                 self.P_t).view(self.num_patches, self.num_patches, self.P)
                self.psi_mtx = torch.autograd.Variable(self.psi_mtx, requires_grad=True)

        # FOV mask
        self.mask = torch.autograd.Variable(torch.ones(spatial_dim, spatial_dim).cuda(), requires_grad=False)
        for i,j in list(itertools.product(np.arange(spatial_dim), np.arange(spatial_dim))):
            if (i-spatial_dim//2)**2 + (j-spatial_dim//2)**2 >= (spatial_dim//2)**2:
                self.mask[i,j] = 0
        
        # Spatial basis functions
        if spatial_mode == 'dip':
            ngf = spatial_dim
            nc = 1
            self.generator = torch.nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 4 * (nc+1)//1, 4, 1, 0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4 * (nc+1)//1, ngf * 2 * (nc+1)//1, 4, 2,1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2 * (nc+1)//1, ngf * 1 * (nc+1)//1, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 1 * (nc+1)//1, ngf * (nc+1)//2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*(nc+1)//2, ngf*(nc+1)//4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*(nc+1)//4, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            )
            self.spatial_basis_fcts = torch.autograd.Variable(torch.randn(K + K_extra + 1, z_dim, 1, 1).cuda(), requires_grad=True)
            gen_total_param = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
            print(self.generator, gen_total_param)

        elif spatial_mode == 'full_basis':
            # f_est
            if spatial_basis_type == 'random':
                self.spatial_basis_fcts = torch.autograd.Variable(
                    torch.zeros(K + K_extra + 1, spatial_dim, spatial_dim).cuda(), requires_grad=True)
            elif spatial_basis_type == 'learned':
                self.spatial_basis_fcts = torch.zeros(K + K_extra + 1, spatial_dim, spatial_dim).cuda()
                self.spatial_basis_fcts[:K+1, :, :] = torch.load('data/spatial_basis_est_%s_%s_P_%d_noise_std_%.2e.pt' %(
                    learning_mode, obj_type, P, noise_std)).cuda()[:,:,:K+1].permute(2,0,1)
                self.spatial_basis_fcts = torch.autograd.Variable(self.spatial_basis_fcts, requires_grad=True)
            else:
                raise NotImplementedError('Spatial basis type for full basis mode not implemented.')
        else:
            raise NotImplementedError('Spatial mode not implemented.')

        if f_est_type == 'random':
            self.f_est = torch.autograd.Variable(
                torch.zeros(P, spatial_dim, spatial_dim).cuda(), requires_grad=True)
        elif f_est_type == 'learned':
            self.f_est = torch.autograd.Variable(
                torch.load('data/f_prosep_est_%s_%s_P_%d_noise_std_%.2e.pt' %(learning_mode, obj_type, P, noise_std)).cuda(), requires_grad=True)
        else:
            raise NotImplementedError('f_est_type mode not implemented.')
    
        plt.figure()
        im = []
        fig, ax = plt.subplots(1, K, figsize=(K*3, 3))
        for k in range(K):
            im.append(ax[k].imshow(self.spatial_basis_fcts[k,:,:].detach().cpu().numpy()))
        plt.show()

        if learning_mode == 'casorati':
            plt.figure()
            im = []
            fig, ax = plt.subplots(1, K, figsize=(K*3, 3))
            for k in range(K):
                im.append(ax[k].plot(self.psi_mtx[k,:].detach().cpu().numpy()))
            plt.show()


    def forward(self):
        if self.temporal_mode == 'z':
            psi_mtx = torch.zeros([self.num_patches, self.num_patches, self.K + self.K_extra + 1, self.P], dtype=torch.float).cuda()
            if self.learning_mode == 'linear':
                for k in range(self.K + self.K_extra + 1):
                    psi_mtx[:, :, k, :] = interpolate(self.temporal_fcts[:, :, k, :].view(self.num_patches, self.num_patches, self.z_dim), size=self.P, mode=self.learning_mode)
            elif self.learning_mode == 'spline':
                for k in range(self.K + self.K_extra + 1):
                    psi_mtx[:, :, k, :] = torch_cubic_spline_interp.interp(self.D_t, self.temporal_fcts[:, :, k, :].view(self.num_patches, self.num_patches, self.z_dim),
                                                         self.P_t).view(self.num_patches, self.num_patches, self.P)
            else:
                raise NotImplementedError('Learning mode not implemented.')
        else:
            raise NotImplementedError('Temporal mode not implemented.')

        if self.spatial_mode == 'full_basis':
            size = self.patch_size
            stride = self.patch_size
            
            spatial_basis_fcts_patch = self.spatial_basis_fcts.unfold(1, size, stride).unfold(2, size, stride).cuda()
            f_psm_est = torch.einsum('...kp,k...js->...pjs', psi_mtx, spatial_basis_fcts_patch)
            # Reshape back #
            output_h = spatial_basis_fcts_patch.shape[1] * spatial_basis_fcts_patch.shape[3]
            output_w = spatial_basis_fcts_patch.shape[2] * spatial_basis_fcts_patch.shape[4]
            f_psm_est = f_psm_est.permute(2, 0, 3, 1, 4).contiguous()
            f_psm_est = f_psm_est.view(self.P, output_h, output_w)
            f_psm_est = (self.mask * f_psm_est).permute(1,2,0)
        else:
            raise NotImplementedError('Spatial mode not implemented.')

        return psi_mtx, f_psm_est
    

class dncnnPatchBased_patchLoss(torch.nn.Module):
    """
    Patch-based DnCNN denoiser for RED-PSM with patch-wise denoised output.
    """
    
    def __init__(self, numLayers, numChannels, filterSize, noise_est_type='direct'):
        super(dncnnPatchBased_patchLoss, self).__init__()
        self.filterSize = filterSize
        self.noise_est_type = noise_est_type
        self.init_layer = nn.Sequential(nn.Conv2d(
                in_channels=1, out_channels=numChannels, kernel_size=filterSize, padding=1), nn.ReLU())
        layers = [nn.Sequential(nn.Conv2d(
            in_channels=numChannels, out_channels=numChannels, kernel_size=filterSize, padding=1), nn.ReLU()) for i in range(numLayers)]
        self.main = nn.Sequential(*layers)
        self.final_layer = nn.Conv2d(in_channels=numChannels, out_channels=1, kernel_size=filterSize, padding=1)
        
    def forward(self, patches):
        if self.noise_est_type == 'direct':
            return self.final_layer(self.main(self.init_layer(patches)))
        elif self.noise_est_type == 'residual':
            return patches - self.final_layer(self.main(self.init_layer(patches)))


class RedPsmSimultMultiProj(nn.Module):
    """
    RED-PSM model class initializing temporal latent representations, spatial and temporal basis functions,
    and the full-rank object f. Supports multiple measurements at a given time instant.
    """
    
    def __init__(self, P, K, K_extra, spatial_dim, num_instances, z_dim, learning_mode, obj_type='walnut',
                 temp_fcts_type='random', f_est_type='random',
                 spatial_basis_type='random',
                 # z_dim=16,
                 temporal_mode='z', noise_std=0, temp_reflect=None, mask=False, rep=4):
        super(RedPsmSimultMultiProj, self).__init__()

        self.P, self.K, self.K_extra, self.z_dim, self.num_instances = P, K, K_extra, z_dim, num_instances
        self.learning_mode = learning_mode
        self.f_est_type, self.spatial_basis_type, self.temporal_mode = f_est_type, spatial_basis_type, temporal_mode
        self.rep = rep
                        
        # Cubic spline interpolation
        self.D_t = torch.linspace(0, 1, self.z_dim).cuda()
        self.P_t = torch.linspace(0, 1, self.P//self.rep).cuda()
        
        self.U_casorati = torch.randn(P,z_dim).cuda()
            
        if learning_mode in ['linear', 'spline']:
            if temporal_mode == 'z':
                self.psi_mtx = torch.zeros([self.K + self.K_extra + 1, self.P//self.rep], dtype=torch.float).cuda()
                if temp_fcts_type == 'random':
                    self.temporal_fcts = torch.autograd.Variable(torch.randn(1, K + K_extra + 1, z_dim).cuda(), requires_grad=True)
                elif temp_fcts_type == 'learned':
                    self.temp_fcts = torch.randn(1, K + K_extra + 1, z_dim)  # Prosep learned temporal fcts of dim d = z_dim
                    if temp_reflect == None:
                        self.temp_fcts[:, :K+1, :] = torch.load('data/temporal_latent_fcts_est_%s_%s_P_%d_K_%d_L_out_%d_noise_std_%.2e.pt' %(
                            learning_mode, obj_type, P, K, z_dim, noise_std))
                    elif temp_reflect == 'end':
                        temp = torch.load('data/temporal_latent_fcts_est_%s_%s_P_%d_K_%d_L_out_%d_noise_std_%.2e.pt' %(
                            learning_mode, obj_type, P//2, K, z_dim//2, noise_std))
                        print(temp.shape, torch.flip(temp, dims=[2]).shape, torch.cat((temp, torch.flip(temp, dims=[2])), dim=2).shape)
                        self.temp_fcts[:, :K+1, :] = torch.cat((temp, torch.flip(temp, dims=[2])), dim=2)
                    self.temporal_fcts = torch.randn(1, K + K_extra + 1, z_dim).cuda()
                    for k in range(self.K + self.K_extra + 1):
                        self.temporal_fcts[:, k, :] = interpolate(self.temp_fcts[:, k, :].view(
                            1, 1, self.temp_fcts.shape[2]), size=self.z_dim, mode='linear')
                    self.temporal_fcts = torch.autograd.Variable(self.temporal_fcts, requires_grad=True)
                else:
                    raise NotImplementedError('temp_fcts_type not implemented.')

        # FOV mask
        self.mask = torch.autograd.Variable(torch.ones(spatial_dim, spatial_dim).cuda(), requires_grad=False)
        if mask:
            for i,j in list(itertools.product(np.arange(spatial_dim), np.arange(spatial_dim))):
                if (i-spatial_dim//2)**2 + (j-spatial_dim//2)**2 >= (spatial_dim//2)**2:
                    self.mask[i,j] = 0
        
        # Spatial basis functions
        if spatial_basis_type == 'random':
            self.spatial_basis_fcts = torch.autograd.Variable(
                torch.zeros(K + K_extra + 1, spatial_dim, spatial_dim).cuda(), requires_grad=True)
        elif spatial_basis_type == 'learned':
            self.spatial_basis_fcts = torch.zeros(K + K_extra + 1, spatial_dim, spatial_dim).cuda()
            self.spatial_basis_fcts[:K+1, :, :] = torch.load('data/spatial_basis_fcts_est_%s_%s_P_%d_noise_std_%.2e.pt' %(
                'spline', obj_type, P, noise_std)).cuda()[:,:,:K+1].permute(2,0,1)
            self.spatial_basis_fcts = torch.autograd.Variable(self.spatial_basis_fcts, requires_grad=True)
        else:
            raise NotImplementedError('Spatial basis type for full basis mode not implemented.')
        
        f_est = torch.einsum('kp,kjs->pjs', self.psi_mtx, self.mask * self.spatial_basis_fcts)
        self.f_est = torch.autograd.Variable((self.mask * f_est), requires_grad=True)
        
        # # Plot initial estimates
        # plots.plot_init_estimates(K, K_extra, z_dim, P//rep, self.spatial_basis_fcts, 
        #                           self.temporal_fcts, self.D_t, self.P_t, self.mask)

    def forward(self):
        if self.temporal_mode == 'z':
            psi_mtx = generate_psi_from_z(self.K, self.K_extra, self.P//self.rep, self.temporal_fcts, self.z_dim,
                                          self.learning_mode, self.D_t, self.P_t, self.U_casorati, gen_temp=None)
        else:
            raise NotImplementedError('Temporal mode not implemented.')

        f_psm_est = torch.einsum('kp,kjs->pjs', psi_mtx, self.mask * self.spatial_basis_fcts)
        f_psm_est = (self.mask * f_psm_est).permute(1,2,0)
        return psi_mtx, f_psm_est