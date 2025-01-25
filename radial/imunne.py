import sys
import os
import functools
import random

import numpy as np
import matplotlib.pyplot as plt
from sympy import limit

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchkbnufft as tkbn
import pytorch_lightning as pl

from rad_compsense import CompressedSensing as CS_Rad

from regularizer import DL_Regularizer, TTV_Regularizer, TPCA_Regularizer
from cu_net.complex_unet import Complex2DtUNet
from rad_cu_net_pl import CUNet as CUNetRad

N_ITER = 5

norm_strategy = None #'unit_circle'

class IMUNNE(pl.LightningModule):
    
    def __init__(self, lr=1e-3):
        super().__init__()
        side = 320
        grid_side = 320
        self.im_size = (side,side)
        grid_size = (grid_side, grid_side)
        
        self.t = nn.parameter.Parameter(data=torch.Tensor([0.8,]).to(self.device), requires_grad=False)
        self.t_decay = nn.parameter.Parameter(data=torch.Tensor([1.0,]).to(self.device), requires_grad=False)
        
        self.cs = CS_Rad(
            lmbda=0,
            truncate_comp_graph=True, 
            update_t=False,
            train_t=False,
            t=None,
            regs=[
                DL_Regularizer( # DL_Reg must be regs[0] (see rad_compsense.py l.66 following)
                    lmbda=0.15,
                    convert=False,
                    mode='output',
                    #cnn = Complex2DtUNet(dim=3, n_filters=20, n_enc_stages=4, res_connection=False, truncate_batch=True)
                    #cnn = CineNetRad.load_from_checkpoint('./lightning_logs/version_100/checkpoints/epoch=999-step=15000.ckpt')
                    cnn = Complex2DtUNet(dim=3, n_filters=4, n_enc_stages=3, res_connection=False, truncate_batch=True),  
                ),
                #TTV_Regularizer(),
            ],
        )
        self.cs.lmbda_ttv = 0#.001
        self.cs.lmbda_pca = 0
        self.pre_transform = tkbn.KbNufftAdjoint(
            numpoints=3, im_size=self.im_size, grid_size=grid_size
        ).to(device=self.device)
        
        self.vgg16 = torchvision.models.vgg16(pretrained=True).features[0:15]
        # adapt to 1-channel input
        vgg_w = self.vgg16[0].weight
        self.vgg16[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg16[0].weight = nn.Parameter(vgg_w.mean(dim=1).unsqueeze(1))
        self.vgg16.eval()
        
        self.n_iter = None
        w = torch.Tensor(range(-160, 160)).abs().to(device=self.device, dtype=torch.complex64)
        w += 2
        w /= w.abs().max()
        w = torch.concat([w for i in range(42)])
        w = w.unsqueeze(0).unsqueeze(0)
        fact = 0.3
        w = torch.min(torch.Tensor([fact]).to(device=self.device), w.real).to(dtype=torch.complex64)
        w /= fact
        self.dcomp = w #torch.ones_like(w)
        self.learning_rate = lr

    def forward(self, batch, n_iter):
        with torch.no_grad():
            kspace, smaps, traj, cs_recon = batch
            kspace = kspace.squeeze()
            smaps = smaps.squeeze()
            traj = traj.squeeze()
            if str(self.dcomp.device) == 'cpu':
                self.dcomp = self.dcomp.to(self.device)
            kdata = kspace.permute(2,1,3,0).reshape(kspace.shape[2], kspace.shape[1], -1)
            
            n_sample,_,_,n_sp = kspace.shape
            # Rad perfusion specific
            w = torch.Tensor(range(-n_sample//2, n_sample//2)).abs().to(device=self.device, dtype=torch.complex64)
            w += 2
            w /= w.abs().max()
            w = torch.concat([w for i in range(n_sp)])
            w = w.unsqueeze(0).unsqueeze(0)
            fact = 0.3
            w = torch.min(torch.Tensor([fact]).to(device=self.device), w.real).to(dtype=torch.complex64)
            w /= fact
            self.dcomp = w
            # Rad perfusion specific
#             if self.add_noise:
#                 siglevel = torch.abs(self.dcomp*kdata).max()
#                 kdata = self.dcomp*kdata + self.sigma * siglevel * torch.randn((self.dcomp*kdata).shape, dtype=torch.complex64).to(kdata)
#                 kdata /= self.dcomp
            x_hat = self.pre_transform(self.dcomp*kdata, traj, smaps=smaps, norm='ortho')
            
            if norm_strategy == 'cs_recon_abs_mean':
                ratio = cs_recon.abs().mean()/x_hat.abs().mean()
                x_hat = ratio * x_hat
                kdata = ratio * kdata
            elif norm_strategy == 'unit_circle':
                x_hat_max = x_hat.abs().max().item()
                cs_recon_max = cs_recon.abs().max().item()
                x_hat = x_hat / x_hat_max
                kdata = kdata / cs_recon_max
                cs_recon = cs_recon / cs_recon_max
            
            self.cs.load_batch(kdata, traj, smaps, self.dcomp)
        for i in range(n_iter):
            #self.cs._t = self.t * (self.t_decay ** n_iter)
            x_hat = self.cs(n_iter=1, t_forward = self.t * (self.t_decay ** i), x0=x_hat, verbose=False, show_frame=0, differentiable=False)
            print(i, "_iter: ", F.mse_loss(
                torch.view_as_real(x_hat[:].detach()), torch.view_as_real(cs_recon.detach()), reduction='sum'
            ).item())
        return x_hat
        
    def configure_optimizers(self):
        # try different optimizers
        optimizer = torch.optim.Adam([
            {'params': self.cs.parameters()},
            {'params': self.t, 'lr': 10*self.learning_rate}, 
            {'params': self.t_decay, 'lr': 10*self.learning_rate}
        ], lr=self.learning_rate)
        lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.95)
        #lr_scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return {"optimizer" : optimizer, "lr_scheduler": lr_scheduler}
         
    def training_step(self, train_batch, batch_idx):
        with torch.no_grad():
            kspace, smaps, traj, cs_recon, f_name = train_batch #cs_recon = batch
            print(f_name[0])
            if not torch.is_complex(kspace) or not torch.is_complex(smaps):
                kspace = torch.view_as_complex(kspace)
                smaps = torch.view_as_complex(smaps)
            kspace = kspace.squeeze()
            smaps = smaps.squeeze()
            traj = traj.squeeze()
            if str(self.dcomp.device) == 'cpu':
                self.dcomp = self.dcomp.to(self.device)
            kdata = kspace.permute(2,1,3,0).reshape(kspace.shape[2], kspace.shape[1], -1)
            
            n_sample,_,_,n_sp = kspace.shape
            # Rad perfusion specific
            w = torch.Tensor(range(-n_sample//2, n_sample//2)).abs().to(device=self.device, dtype=torch.complex64)
            w += 2
            w /= w.abs().max()
            w = torch.concat([w for i in range(n_sp)])
            w = w.unsqueeze(0).unsqueeze(0)
            fact = 0.3
            w = torch.min(torch.Tensor([fact]).to(device=self.device), w.real).to(dtype=torch.complex64)
            w /= fact
            self.dcomp = w
            # Rad perfusion specific
            
            x_hat = self.pre_transform(self.dcomp*kdata, traj, smaps=smaps, norm='ortho')

            if norm_strategy == 'cs_recon_abs_mean':
                ratio = cs_recon.abs().mean()/x_hat.abs().mean()
                x_hat = ratio * x_hat
                kdata = ratio * kdata
            elif norm_strategy == 'unit_circle':
                x_hat_max = x_hat.abs().max().item()
                cs_recon_max = cs_recon.abs().max().item()
                x_hat = x_hat / x_hat_max
                kdata = kdata / cs_recon_max
                cs_recon = cs_recon / cs_recon_max
                
                
            self.cs.load_batch(kdata, traj, smaps, self.dcomp)
        
        missing_iterations = 0 #random.randint(0,3)
        
        for i in range(N_ITER-missing_iterations-1):
            #self.cs._t = t * (t_decay ** i)
            x_hat = self.cs(n_iter=1, t_forward = self.t * (self.t_decay ** i), x0=x_hat, verbose=False, show_frame=0, differentiable=False)
            print(i, "_iter: ", F.mse_loss(
                torch.view_as_real(x_hat[:].detach()), torch.view_as_real(cs_recon.detach()), reduction='sum'
            ).item())
        
        #self.cs._t = t * (t_decay ** (N_ITER-missing_iterations-1))
        x_hat = self.cs(
            n_iter=1, 
            t_forward = self.t * (self.t_decay ** (N_ITER-missing_iterations-1)), 
            x0=x_hat, 
            verbose=False, 
            show_frame=0, 
            differentiable=True
        )
        print(N_ITER-missing_iterations-1, "_iter: ", F.mse_loss(
                torch.view_as_real(x_hat.detach()), torch.view_as_real(cs_recon.detach()), reduction='sum'
            ).item())
        #x_hat_coils = x_hat # * smaps
        #cs_recon_coils = cs_recon[0]# * smaps
        offset_val = 80
        x_hat_coils = x_hat#[...,offset_val:-offset_val,offset_val:-offset_val] # * smaps
        cs_recon_coils = cs_recon[0]#, ..., offset_val:-offset_val,offset_val:-offset_val]# * smaps

#         perc_x_real = self.vgg16(x_hat_coils.abs())
#         perc_gt_real = self.vgg16(cs_recon_coils.abs())
        #perc_x_imag = self.vgg16(x_hat_coils.imag)
        #perc_gt_imag = self.vgg16(cs_recon_coils.imag)
        
        loss_1 = F.mse_loss(
            torch.view_as_real(x_hat_coils), torch.view_as_real(cs_recon_coils), reduction='sum'
        )
        
        #loss_2 = 1e-02 * (F.mse_loss(perc_x_real, perc_gt_real, reduction='sum'))# + F.mse_loss(perc_x_imag, perc_gt_imag, reduction='sum'))
                       
        combined_loss = loss_1# + loss_2
        
        self.log('L2', loss_1, on_step=True, on_epoch=True)
        #self.log('L2', loss_1 * cs_recon_max, on_step=True, on_epoch=True)
        #self.log('Perc_loss', loss_2, on_step=True, on_epoch=True)
        self.log('combined_loss', combined_loss, on_step=True, on_epoch=True)
        self.log('t', self.t, on_step=True, on_epoch=True)
        self.log('t_decay', self.t_decay, on_step=True, on_epoch=True)
        #self.log('lambda', self.cs.lmbda_t, on_step=True, on_epoch=True)
        return combined_loss
        
    def validation_step(self, val_batch, batch_idx):
        kspace, smaps, traj, cs_recon, f_name = val_batch   
        kspace = kspace.squeeze()
        smaps = smaps.squeeze()
        traj = traj.squeeze()
       
        #cs_recon = cs_recon.squeeze()
        if str(self.dcomp.device) == 'cpu':
                self.dcomp = self.dcomp.to(self.device)
        kdata = kspace.permute(2,1,3,0).reshape(kspace.shape[2], kspace.shape[1], -1)
        n_sample,_,_,n_sp = kspace.shape
        # Rad perfusion specific
        w = torch.Tensor(range(-n_sample//2, n_sample//2)).abs().to(device=self.device, dtype=torch.complex64)
        w += 2
        w /= w.abs().max()
        w = torch.concat([w for i in range(n_sp)])
        w = w.unsqueeze(0).unsqueeze(0)
        fact = 0.3
        w = torch.min(torch.Tensor([fact]).to(device=self.device), w.real).to(dtype=torch.complex64)
        w /= fact
        self.dcomp = w
        
        # Rad perfusion specific
        x_hat = self.pre_transform(self.dcomp*kdata, traj, smaps=smaps, norm='ortho')
        
        if norm_strategy == 'cs_recon_abs_mean':
            ratio = cs_recon.abs().mean()/x_hat.abs().mean()
            x_hat = ratio * x_hat
            kdata = ratio * kdata
        elif norm_strategy == 'unit_circle':
            x_hat_max = x_hat.abs().max().item()
            cs_recon_max = cs_recon.abs().max().item()
            x_hat = x_hat / x_hat_max
            kdata = kdata / cs_recon_max
            cs_recon = cs_recon / cs_recon_max
        
        self.cs.load_batch(kdata, traj, smaps, self.dcomp)
        
        with torch.set_grad_enabled(True):
            for i in range(N_ITER):
                #self.cs._t = self.cs._t * (t_decay ** i)
                x_hat = self.cs(n_iter=1, t_forward= self.t*(self.t_decay**i), x0=x_hat, verbose=False, show_frame=0, differentiable=False)
                print(i, "_iter: ", F.mse_loss(
                    torch.view_as_real(x_hat.detach()), torch.view_as_real(cs_recon.detach()), reduction='sum'
                ).item())
        offset_val = 80
        x_hat_coils = x_hat[...,offset_val:-offset_val,offset_val:-offset_val] # * smaps
        cs_recon_coils = cs_recon[0, ..., offset_val:-offset_val,offset_val:-offset_val]# * smaps
        

        perc_x_real = self.vgg16(x_hat_coils.real)
        perc_gt_real = self.vgg16(cs_recon_coils.real)
        perc_x_imag = self.vgg16(x_hat_coils.imag)
        perc_gt_imag = self.vgg16(cs_recon_coils.imag)
        
        loss_1 = F.mse_loss(
            torch.view_as_real(x_hat_coils), torch.view_as_real(cs_recon_coils), reduction='sum'
        )
        loss_2 = 1e-04 * (F.mse_loss(perc_x_real, perc_gt_real, reduction='sum') + F.mse_loss(perc_x_imag, perc_gt_imag, reduction='sum'))
        combined_loss = loss_1 + loss_2

        self.log('val_L2', loss_1, on_step=True, on_epoch=True)
        self.log('val_Perc_loss', loss_2, on_step=True, on_epoch=True)
        self.log('val_t', self.t, on_step=True, on_epoch=True)
        self.log('t_decay', self.t_decay, on_step=True, on_epoch=True)
        #self.log('val_lambda', self.cs.lmbda_t, on_step=True, on_epoch=True)
        self.log('val_combined_loss', combined_loss, on_step=True, on_epoch=True)
