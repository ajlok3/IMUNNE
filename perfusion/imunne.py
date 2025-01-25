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

from compsense import CompressedSensing as CS_Cart

from regularizer import DL_Regularizer, TTV_Regularizer, TPCA_Regularizer
from cu_net.complex_unet import Complex2DtUNet
import sys
sys.path.append('..')
from load_scan import calc_cs_mask
from dyn_fft import DynFFT
from dyn_ifft import DyniFFT

N_ITER = 4

class IMUNNE(pl.LightningModule):
    
    def __init__(self, lr=1e-3):
        super().__init__()
        side = 384
        grid_side = 384
        self.im_size = (side,side)
        grid_size = (grid_side, grid_side)        
        self.pre_transform = DyniFFT()
        
        self.cs = CS_Cart(
            lmbda=0, 
            truncate_comp_graph=True, 
            update_t=False,
            train_t_lmbda=True,
            t=1.3,
            regs=[
                DL_Regularizer( # DL_Reg must be regs[0] (see rad_compsense.py l.66 following)
                    lmbda=0.12,
                    convert=False,
                    mode='output',
                    #cnn = Complex2DtUNet(dim=3, n_filters=20, n_enc_stages=4, res_connection=False, truncate_batch=True)
                    #cnn = CineNetRad.load_from_checkpoint('./lightning_logs/version_100/checkpoints/epoch=999-step=15000.ckpt')
                    cnn = Complex2DtUNet(dim=3, n_filters=4, n_enc_stages=3, res_connection=False, truncate_batch=True),  
                ),
                TTV_Regularizer(),
            ],
        )
        self.cs.lmbda_ttv = 0.001
        self.cs.lmbda_pca = 0
        
        self.vgg16 = torchvision.models.vgg16(pretrained=True).features[0:15]
        # adapt to 1-channel input
        vgg_w = self.vgg16[0].weight
        self.vgg16[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg16[0].weight = nn.Parameter(vgg_w.mean(dim=1).unsqueeze(1))
        self.vgg16.eval()
        
        self.n_iter = None
        self.learning_rate = lr

    def forward(self, batch, n_iter):
        with torch.no_grad():
            kspace, smaps, cs_recon = batch
            kspace = kspace.squeeze()
            smaps = smaps.squeeze()
            cs_mask_3d = calc_cs_mask(kspace)
            self.pre_transform.load_data(smaps, cs_mask_3d)
            kdata = kspace.permute(2,1,3,0).reshape(kspace.shape[2], kspace.shape[1], -1)
                
            x_hat = self.pre_transform(kdata)

            self.cs.load_batch(kdata, smaps)
        
        for i in range(n_iter):
            x_hat = self.cs(n_iter=1, x0=x_hat, verbose=False, show_frame=0, differentiable=False)
            print(i, "_iter: ", F.mse_loss(
                torch.view_as_real(x_hat[:].detach()), torch.view_as_real(cs_recon.detach()), reduction='sum'
            ).item())
        return x_hat
        
    def configure_optimizers(self):
        # try different optimizers
        # import pdb;pdb.set_trace()
        optimizer = torch.optim.Adam(list(self.cs.parameters()), lr=self.learning_rate)
        lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.95)
        #lr_scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return {"optimizer" : optimizer, "lr_scheduler": lr_scheduler}
         
    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            kspace, smaps, cs_recon = batch
            kspace = kspace.squeeze()
            smaps = smaps.squeeze()
            cs_mask_3d = calc_cs_mask(kspace)
            self.pre_transform.load_data(smaps, cs_mask_3d)

            x_hat = self.pre_transform(kspace)
            self.cs.load_batch(kspace, smaps)
        
        missing_iterations = 0 #random.randint(0,3)
        for i in range(N_ITER-missing_iterations-1):
            x_hat = self.cs(n_iter=1, x0=x_hat, verbose=False, show_frame=0, differentiable=False)
            print(i, "_iter: ", F.mse_loss(
                torch.view_as_real(x_hat[:].detach()), torch.view_as_real(cs_recon.detach()), reduction='sum'
            ).item())
        
        x_hat = self.cs(n_iter=1, x0=x_hat, verbose=False, show_frame=0, differentiable=True)
        print(N_ITER-missing_iterations-1, "_iter: ", F.mse_loss(
                torch.view_as_real(x_hat.detach()), torch.view_as_real(cs_recon.detach()), reduction='sum'
            ).item())
        #x_hat_coils = x_hat # * smaps
        #cs_recon_coils = cs_recon[0]# * smaps
        offset_val = 80
        x_hat_coils = x_hat#[...,offset_val:-offset_val,offset_val:-offset_val] # * smaps
        cs_recon_coils = cs_recon[0]#, ..., offset_val:-offset_val,offset_val:-offset_val]# * smaps

        #perc_x_real = self.vgg16(x_hat_coils.abs())
        #perc_gt_real = self.vgg16(cs_recon_coils.abs())
        #perc_x_imag = self.vgg16(x_hat_coils.imag)
        #perc_gt_imag = self.vgg16(cs_recon_coils.imag)
        
        loss_1 = F.mse_loss(
            torch.view_as_real(x_hat_coils), torch.view_as_real(cs_recon_coils), reduction='sum'
        )
        
        #loss_2 = 1e-02 * (F.mse_loss(perc_x_real, perc_gt_real, reduction='sum'))# + F.mse_loss(perc_x_imag, perc_gt_imag, reduction='sum'))
                       
        combined_loss = loss_1# + loss_2
        self.log('L2', loss_1, on_step=True, on_epoch=True)
        #self.log('Perc_loss', loss_2, on_step=True, on_epoch=True)
        self.log('combined_loss', combined_loss, on_step=True, on_epoch=True)
        self.log('t', self.cs.t, on_step=True, on_epoch=True)
        #self.log('t_decay', self.t_decay, on_step=True, on_epoch=True)
        self.log('lambda', self.cs.lmbda_t, on_step=True, on_epoch=True)
        return combined_loss
        
    def validation_step(self, batch, batch_idx):
        kspace, smaps, cs_recon = batch
        kspace = kspace.squeeze()
        smaps = smaps.squeeze()
        cs_mask_3d = calc_cs_mask(kspace)
        self.pre_transform.load_data(smaps, cs_mask_3d)

        x_hat = self.pre_transform(kspace)

        self.cs.load_batch(kspace, smaps)
        
        with torch.set_grad_enabled(True):
            for i in range(N_ITER):
                x_hat = self.cs(n_iter=1, x0=x_hat, verbose=False, show_frame=0, differentiable=False)
                print(i, "_iter: ", F.mse_loss(
                    torch.view_as_real(x_hat.detach()), torch.view_as_real(cs_recon.detach()), reduction='sum'
                ).item())
        offset_val = 80
        
        x_hat_coils = x_hat.unsqueeze(1)
        cs_recon_coils = cs_recon[0].unsqueeze(1)# * smaps
        
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
        self.log('val_t', self.cs.t, on_step=True, on_epoch=True)
        #self.log('t_decay', self.t_decay, on_step=True, on_epoch=True)
        self.log('val_lambda', self.cs.lmbda_t, on_step=True, on_epoch=True)
        self.log('val_combined_loss', combined_loss, on_step=True, on_epoch=True)
