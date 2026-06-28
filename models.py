import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(input)


class WIRE(nn.Module):
    def __init__(self, w0, s0):
        super().__init__()
        self.w0 = w0
        self.s0 = s0
        self.omega = nn.Parameter(self.w0 * torch.ones(1), requires_grad=False)
        self.scale = nn.Parameter(self.s0 * torch.ones(1), requires_grad=False)

    def forward(self, input):
        return torch.cos(self.omega * input) * torch.exp(-1 * (self.scale * input) ** 2)


class NeuralFieldModel(nn.Module):
    def __init__(self, input_ch=2, output_dim=1, num_layers=4, num_channels=256, act_type='sine'):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_channels = num_channels

        if act_type == 'sine':
            self.act = Sine()
        elif act_type == 'relu':
            self.act = nn.ReLU(True)
        elif act_type == 'gelu':
            self.act = nn.GELU()

        layers = [nn.Conv2d(input_ch, num_channels, kernel_size=1, bias=True), self.act()]
        for _ in range(num_layers - 1):
            layers += [
                nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=True),
                self.act(),
                nn.BatchNorm2d(num_channels),
            ]
        layers += [
            nn.Conv2d(num_channels, output_dim, kernel_size=1),
            self.act if act_type == 'sine' else nn.Sigmoid(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(coords)


class NeuralFieldModel3D(nn.Module):
    def __init__(
        self,
        input_ch=3,
        output_dim=1,
        num_layers=4,
        num_channels=256,
        spatial_dim=128,
        mask=True,
        P=32,
        act_type='sine',
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_channels = num_channels

        if act_type == 'sine':
            self.act = Sine()
        elif act_type == 'relu':
            self.act = nn.ReLU(True)
        else:
            raise NotImplementedError

        layers = [nn.Conv3d(input_ch, num_channels, kernel_size=1, bias=True), self.act]
        for _ in range(num_layers - 1):
            layers += [
                nn.Conv3d(num_channels, num_channels, kernel_size=1, bias=True),
                self.act,
                nn.BatchNorm3d(num_channels),
            ]
        layers += [
            nn.Conv3d(num_channels, output_dim, kernel_size=1),
            nn.Sigmoid(),
        ]
        self.net = nn.Sequential(*layers)

        # FOV mask
        self.mask = torch.autograd.Variable(
            torch.ones(spatial_dim, spatial_dim).cuda(), requires_grad=False
        )
        if mask:
            for i, j in list(itertools.product(np.arange(spatial_dim), np.arange(spatial_dim))):
                if (i - spatial_dim // 2) ** 2 + (j - spatial_dim // 2) ** 2 >= (
                    spatial_dim // 2
                ) ** 2:
                    self.mask[i, j] = 0
        self.f_est = torch.autograd.Variable(
            (self.mask[..., None] * torch.zeros(spatial_dim, spatial_dim, P).cuda()),
            requires_grad=True,
        )

    def forward(self, coords):
        return self.mask[None, None, ..., None] * self.net(coords)


class NeuralFieldModel3D_fc(nn.Module):
    def __init__(
        self,
        input_ch=3,
        output_dim=1,
        num_layers=7,
        num_channels=64,
        spatial_dim=128,
        mask=True,
        P=32,
        act_type='relu',
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_channels = num_channels

        if act_type == 'sine':
            self.act = Sine()
        elif act_type == 'relu':
            self.act = nn.ReLU(True)
        elif act_type == 'gelu':
            self.act = nn.GELU(True)
        else:
            raise NotImplementedError

        layers = [nn.Linear(input_ch, num_channels, bias=True), self.act]
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(num_channels, num_channels, bias=True),
                self.act,
                nn.BatchNorm1d(num_channels),
            ]
        layers += [
            nn.Linear(num_channels, output_dim),
            # nn.Sigmoid(),
            nn.ReLU(),
        ]
        self.net = nn.Sequential(*layers)

        # FOV mask
        self.mask = torch.autograd.Variable(
            torch.ones(spatial_dim, spatial_dim).cuda(), requires_grad=False
        ).cuda()
        if mask:
            for i, j in list(itertools.product(np.arange(spatial_dim), np.arange(spatial_dim))):
                if (i - spatial_dim // 2) ** 2 + (j - spatial_dim // 2) ** 2 >= (
                    spatial_dim // 2
                ) ** 2:
                    self.mask[i, j] = 0
        self.f_est = torch.zeros(spatial_dim, spatial_dim, P).cuda()
        self.gamma_est = torch.zeros(spatial_dim, spatial_dim, P).cuda()

    def forward(self, coords):
        x = coords.flatten(2, 4).permute(0, 2, 1).squeeze()
        x = self.net(x)
        x = torch.unflatten(x, -2, (coords.shape[2], coords.shape[3], coords.shape[4])).permute(
            3, 0, 1, 2
        )[None, ...]
        return self.mask[None, None, ..., None] * x


class NeuralFieldModel3D_fc_sinogram(nn.Module):
    def __init__(
        self,
        input_ch=3,
        output_dim=1,
        num_layers=4,
        num_channels=256,
        spatial_dim=128,
        mask=True,
        P=32,
        act_type='sine',
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_channels = num_channels

        if act_type == 'sine':
            self.act = Sine()
        elif act_type == 'relu':
            self.act = nn.ReLU(True)
        else:
            raise NotImplementedError

        layers = [
            nn.Linear(input_ch, num_channels, bias=True),
            self.act,
            nn.BatchNorm1d(num_channels),
        ]
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(num_channels, num_channels, bias=True),
                self.act,
                nn.BatchNorm1d(num_channels),
            ]
        layers += [
            nn.Linear(num_channels, output_dim),
            nn.Sigmoid(),
        ]
        self.net = nn.Sequential(*layers)

        self.g_est = torch.autograd.Variable(
            (torch.zeros(spatial_dim, P_static, P).cuda()), requires_grad=True
        )

    def forward(self, coords):
        x = coords.flatten(2, 4).permute(0, 2, 1).squeeze()
        x = self.net(x)
        x = torch.unflatten(x, -2, (coords.shape[2], coords.shape[3], coords.shape[4])).permute(
            3, 0, 1, 2
        )[None, ...]
        return x


class SineEncoding(nn.Module):
    def __init__(self, input_dim=2, sine_order=1):
        super().__init__()
        self.input_dim = input_dim
        self.sine_order = [2 ** (i + 1) for i in range(sine_order)]

    def forward(self, coords):
        return torch.cat(
            (torch.sin(2 * torch.pi * coords), torch.cos(2 * torch.pi * coords)), dim=1
        )


class GaussianFourierEncoding(nn.Module):
    def __init__(self, input_dim, mapping_size, scale):
        super().__init__()
        self.input_dim = input_dim
        self._mapping_size = mapping_size
        self._B = (torch.randn(input_dim, mapping_size) * scale).cuda()

    def forward(self, coords):
        batches, channels, height, width = coords.shape
        coords = coords.permute(0, 2, 3, 1).reshape(batches * height * width, channels)
        coords = coords @ self._B
        coords = coords.reshape(batches, height, width, self._mapping_size).permute(0, 3, 1, 2)
        return torch.cat(
            (torch.sin(2 * torch.pi * coords), torch.cos(2 * torch.pi * coords)), dim=1
        )


class GaussianFourierEncoding3D(nn.Module):
    def __init__(self, input_dim, mapping_size, scale):
        super().__init__()
        self.input_dim = input_dim
        self._mapping_size = mapping_size
        self._B = (torch.randn(input_dim, mapping_size) * scale).cuda()

    def forward(self, coords):
        batches, channels, height, width, P = coords.shape
        coords = coords.permute(0, 2, 3, 4, 1).reshape(batches * height * width * P, channels)
        coords = coords @ self._B
        coords = coords.reshape(batches, height, width, P, self._mapping_size).permute(
            0, 4, 1, 2, 3
        )
        return torch.cat(
            (torch.sin(2 * torch.pi * coords), torch.cos(2 * torch.pi * coords)), dim=1
        )


class FourierEncoding3D(nn.Module):
    def __init__(self, input_dim, mapping_size):
        super().__init__()
        self.input_dim = input_dim
        self._mapping_size = mapping_size
        self.coefs = (
            (torch.arange(start=1, end=mapping_size + 1e-12) * torch.pi * 0.5)
            .view(1, mapping_size, 1, 1, 1)
            .cuda()
        )

    def forward(self, coords):
        arg = torch.kron(self.coefs, coords.contiguous())
        return torch.hstack((torch.sin(arg), torch.cos(arg)))


class FourierEncoding2D_t(nn.Module):
    def __init__(self, input_dim, mapping_size, t_freq_ratio):
        super().__init__()

        self.input_dim = input_dim
        self._mapping_size = mapping_size
        self.coefs_spatial = (
            (torch.arange(start=1, end=mapping_size + 1e-12) * torch.pi * 0.5)
            .view(1, mapping_size, 1, 1, 1)
            .cuda()
        )
        self.coefs_temporal = (
            (torch.arange(start=1, end=mapping_size + 1e-12) * torch.pi * 0.5 * t_freq_ratio)
            .view(1, mapping_size, 1, 1, 1)
            .cuda()
        )

    def forward(self, coords):
        arg_spatial = torch.kron(self.coefs_spatial, coords[:, :2, :, :, :].contiguous())
        arg_temporal = torch.kron(self.coefs_temporal, coords[:, 2:, :, :, :].contiguous())
        return torch.hstack(
            (
                torch.sin(torch.cat((arg_spatial, arg_temporal), dim=1)),
                torch.cos(torch.cat((arg_spatial, arg_temporal), dim=1)),
            )
        )


class RedStaticRecon(nn.Module):
    def __init__(self, spatial_dim, mask=False):
        """
        Initializes the static reconstruction model with RED.

        Parameters:
        ----------
        P (int): Number of measurements.
        mask (bool): If True, apply FOV mask for tomographic objects.
        """
        super().__init__()

        # FOV mask
        self.mask = torch.autograd.Variable(
            torch.ones(spatial_dim, spatial_dim).cuda(), requires_grad=False
        )
        if mask:
            for i, j in list(itertools.product(np.arange(spatial_dim), np.arange(spatial_dim))):
                if (i - spatial_dim // 2) ** 2 + (j - spatial_dim // 2) ** 2 >= (
                    spatial_dim // 2
                ) ** 2:
                    self.mask[i, j] = 0

        self.f_est = torch.autograd.Variable(
            torch.zeros(spatial_dim, spatial_dim).cuda(), requires_grad=True
        )
        self.z_est = torch.autograd.Variable(
            torch.zeros(spatial_dim, spatial_dim).cuda(), requires_grad=True
        )
        self.gamma_est = torch.autograd.Variable(
            torch.zeros(spatial_dim, spatial_dim).cuda(), requires_grad=True
        )

    def forward(self):
        return self.f_est, self.z_est, self.gamma_est


class RedStaticReconFull(nn.Module):
    def __init__(self, spatial_dim, P, f_init=None, mask=False):
        """
        Initializes the static reconstruction model with RED.

        Parameters:
        ----------
        P (int): Number of measurements.
        mask (bool): If True, apply FOV mask for tomographic objects.
        """
        super().__init__()

        # FOV mask
        self.mask = torch.autograd.Variable(
            torch.ones(spatial_dim, spatial_dim).cuda(), requires_grad=False
        )
        if mask:
            for i, j in list(itertools.product(np.arange(spatial_dim), np.arange(spatial_dim))):
                if (i - spatial_dim // 2) ** 2 + (j - spatial_dim // 2) ** 2 >= (
                    spatial_dim // 2
                ) ** 2:
                    self.mask[i, j] = 0

        if f_init == None:
            self.f_est = torch.autograd.Variable(
                torch.zeros(spatial_dim, spatial_dim, P).cuda(), requires_grad=True
            )
            self.z_est = torch.autograd.Variable(
                torch.zeros(spatial_dim, spatial_dim, P).cuda(), requires_grad=False
            )
        else:
            self.f_est = torch.autograd.Variable(f_init.clone().detach(), requires_grad=True)
            self.z_est = torch.autograd.Variable(f_init.clone().detach(), requires_grad=False)

        self.gamma_est = torch.autograd.Variable(
            torch.zeros(spatial_dim, spatial_dim, P).cuda(), requires_grad=False
        )

    def forward(self):
        return self.f_est, self.z_est, self.gamma_est


class NF3DProjDomain(nn.Module):
    def __init__(
        self,
        input_ch=3,
        output_dim=1,
        num_layers=4,
        num_channels=256,
        spatial_dim=128,
        P=32,
        act_type='sine',
        P_static=128,
        init=None,
        w0=10,
        s0=5,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_channels = num_channels

        if act_type == 'sine':
            self.act = Sine()
        elif act_type == 'relu':
            self.act = nn.ReLU(True)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(True)
        elif act_type == 'WIRE':
            self.act = WIRE(w0=w0, s0=s0)
        else:
            raise NotImplementedError

        layers = [
            nn.Linear(input_ch, num_channels, bias=True),
            self.act,
            nn.BatchNorm1d(num_channels),
        ]
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(num_channels, num_channels, bias=True),
                self.act,
                nn.BatchNorm1d(num_channels),
            ]
        layers += [
            nn.Linear(num_channels, output_dim),
            nn.ReLU(),
        ]
        self.net = nn.Sequential(*layers)
        if isinstance(init, np.ndarray):
            self.g_est = init
        else:
            self.g_est = torch.zeros(spatial_dim, P_static, P).cuda()
        self.gamma_est = torch.zeros(spatial_dim, P_static, P).cuda()
        self.unflatten = nn.Unflatten(-2, (spatial_dim, P_static, P))
        self.unflatten_sub = nn.Unflatten(-2, (spatial_dim, P, 1))

    def forward(self, coords):
        x = coords.flatten(2, 4).permute(0, 2, 1).squeeze()
        x = self.net(x)
        # x = torch.unflatten(x, -2, (
        #         coords.shape[2], coords.shape[3], coords.shape[4])).permute(3,0,1,2)[None, ...]
        if coords.shape[-1] != 1:
            x = self.unflatten(x).permute(3, 0, 1, 2)[None, ...]
        else:
            x = self.unflatten_sub(x).permute(3, 0, 1, 2)[None, ...]
        return x


class NeuralFieldModel3D_fc(nn.Module):
    def __init__(
        self, input_ch, output_dim, num_layers, num_channels, spatial_dim, mask, P, act_type
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_channels = num_channels

        if act_type == 'sine':
            self.act = Sine()
        elif act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU()
        else:
            raise NotImplementedError

        layers = [
            nn.Linear(input_ch, num_channels, bias=True),
            nn.BatchNorm1d(num_channels),
            self.act,
        ]
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(num_channels, num_channels, bias=True),
                nn.BatchNorm1d(num_channels),
                self.act,
            ]
        layers += [
            nn.Linear(num_channels, output_dim, bias=True),
            nn.ReLU(),
        ]
        self.net = nn.Sequential(*layers)

        # FOV mask
        self.mask = torch.ones(spatial_dim, spatial_dim).cuda()
        if mask:
            for i, j in list(itertools.product(np.arange(spatial_dim), np.arange(spatial_dim))):
                if (i - spatial_dim // 2) ** 2 + (j - spatial_dim // 2) ** 2 >= (
                    spatial_dim // 2
                ) ** 2:
                    self.mask[i, j] = 0
        self.mask = torch.autograd.Variable(self.mask, requires_grad=False)
        self.f_est = F.relu(1e-5 * torch.randn(spatial_dim, spatial_dim, P).cuda())
        self.gamma_est = torch.zeros(spatial_dim, spatial_dim, P).cuda()

    def forward(self, coords):
        x = coords.flatten(2, 4).permute(0, 2, 1).squeeze()
        x = self.net(x)
        x = torch.unflatten(x, -2, (coords.shape[2], coords.shape[3], coords.shape[4])).permute(
            3, 0, 1, 2
        )[None, ...]
        return self.mask[None, None, ..., None] * x


# class NeuralFieldModel3D_fc_res(nn.Module):
#     def __init__(self, input_ch, output_dim, num_layers, num_channels,
#                  spatial_dim, mask, P, act_type, f_max, final_act):
#         super(NeuralFieldModel3D_fc_res, self).__init__()
#         self.output_dim = output_dim
#         self.num_layers = num_layers
#         self.num_channels = num_channels
#         self.f_max = f_max
#         self.final_act = final_act

#         if act_type == 'sine':
#             self.act = Sine()
#         elif act_type == 'relu':
#             self.act = nn.ReLU()
#         elif act_type == 'gelu':
#             self.act = nn.GELU()
#         elif act_type == 'leakyrelu':
#             self.act = nn.LeakyReLU()
#         else:
#             raise NotImplementedError

#         self.layers = []
#         self.layers.append(nn.Linear(input_ch, num_channels, bias=True))
#         for _ in range(num_layers-1):
#             self.layers.append(nn.Linear(num_channels, num_channels, bias=True))
#         self.layers.append(nn.Linear(num_channels, output_dim, bias=True))
#         self.layers = nn.ModuleList(self.layers)

#         with torch.no_grad():
#             w = self.num_channels**(-1/2)
#             for lyr in self.layers[1:-1]:
#                 lyr.weight.uniform_(-w, w)
#                 lyr.bias.uniform_(-1, 1)
#             self.layers[-1].weight.uniform_(-1, 1)

#         # FOV mask #
#         self.mask = torch.ones(spatial_dim, spatial_dim).cuda()
#         if mask:
#             for i,j in list(itertools.product(
#                 np.arange(spatial_dim), np.arange(spatial_dim))):
#                 if (i-spatial_dim//2)**2 + (
#                     j-spatial_dim//2)**2 >= (spatial_dim//2)**2:
#                     self.mask[i,j] = 0
#         self.mask = torch.autograd.Variable(self.mask, requires_grad=False)
#         self.f_est = F.relu(
#             f_max/16 * torch.randn(spatial_dim, spatial_dim, P)).cuda()
#         self.gamma_est = torch.zeros(spatial_dim, spatial_dim, P).cuda()

#     def forward(self, coords):
#         x = coords.flatten(2, 4).permute(0,2,1).squeeze()

#         y = self.layers[0](x)
#         x = self.layers[0](x)
#         for lyr in self.layers[1:-1]:
#             x = self.act(lyr(x)) + y
#         # x = self.layers[-1](x) * 2 * self.output_dim
#         x = self.layers[-1](x)
#         # x = -x**2

#         if self.final_act == 'sigmoid':
#             x = self.f_max * torch.sigmoid(x)
#         elif self.final_act == 'relu':
#             x = self.f_max * F.relu(x)
#         elif self.final_act == 'gelu':
#             x = self.f_max * F.gelu(x)
#         elif self.final_act == 'same':
#             x = self.f_max * self.act(x)
#         elif self.final_act == 'none':
#             x = self.f_max * x

#         x = torch.unflatten(x, -2, (
#             coords.shape[2], coords.shape[3],
#             coords.shape[4])).permute(3,0,1,2)[None, ...]
#         return self.mask[None, None, ..., None] * x


class NeuralFieldModel3D_fc_res(nn.Module):
    def __init__(
        self,
        input_ch,
        output_dim,
        num_layers,
        num_channels,
        spatial_dim,
        mask,
        P,
        act_type,
        f_max,
        final_act,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.f_max = f_max
        self.final_act = final_act

        act_dict = {
            'sine': Sine(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leakyrelu': nn.LeakyReLU(),
        }
        if act_type not in act_dict:
            raise NotImplementedError
        self.act = act_dict[act_type]

        self.layers = nn.ModuleList(
            [nn.Linear(input_ch, num_channels, bias=True)]
            + [nn.Linear(num_channels, num_channels, bias=True) for _ in range(num_layers - 1)]
            + [nn.Linear(num_channels, output_dim, bias=True)]
        )

        with torch.no_grad():
            w = self.num_channels ** (-1 / 2)
            for lyr in self.layers[1:-1]:
                lyr.weight.uniform_(-w, w)
                lyr.bias.uniform_(-1, 1)
            self.layers[-1].weight.uniform_(-1, 1)

        self.mask = torch.ones(spatial_dim, spatial_dim).cuda()
        if mask:
            for i, j in itertools.product(np.arange(spatial_dim), repeat=2):
                if (i - spatial_dim // 2) ** 2 + (j - spatial_dim // 2) ** 2 >= (
                    spatial_dim // 2
                ) ** 2:
                    self.mask[i, j] = 0
        self.mask = torch.autograd.Variable(self.mask, requires_grad=False)
        self.f_est = F.relu(f_max / 16 * torch.randn(spatial_dim, spatial_dim, P).cuda())
        self.gamma_est = torch.zeros(spatial_dim, spatial_dim, P).cuda()

    def forward(self, coords):
        x = coords.flatten(2, 4).permute(0, 2, 1).squeeze()
        y = self.layers[0](x)
        x = y
        for i, lyr in enumerate(self.layers[1:-1]):
            x = self.act(lyr(x)) + 0.5 * y
        x = self.layers[-1](x)

        final_act_dict = {
            'sigmoid': torch.sigmoid,
            'relu': F.relu,
            'gelu': F.gelu,
            'same': self.act,
            'none': lambda x: x,
        }
        if self.final_act not in final_act_dict:
            raise NotImplementedError
        x = self.f_max * final_act_dict[self.final_act](x)

        x = torch.unflatten(x, -2, (coords.shape[2], coords.shape[3], coords.shape[4])).permute(
            3, 0, 1, 2
        )[None, ...]
        return self.mask[None, None, ..., None] * x
