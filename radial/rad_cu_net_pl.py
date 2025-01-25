import sys
import os
import functools

import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import torchvision

from torch.optim.lr_scheduler import StepLR
import torchkbnufft as tkbn

sys.path.append('..')
from cu_net.complex_unet import Complex2DtUNet
from regularizer import DL_Regularizer


class CUNet(pl.LightningModule):
    '''
        Pytorch lightning wrapper for 
    '''
    def __init__(self, lr=1e-03):
        super().__init__()
        side = 320
        grid_side = 320
        self.im_size = (side,side)
        grid_size = (grid_side, grid_side)
        self.pre_transform = tkbn.KbNufftAdjoint(
            numpoints=3, im_size=self.im_size, grid_size=grid_size
        ).to(device=self.device)
        self.cnn = Complex2DtUNet(dim=3, n_filters=4, n_enc_stages=3, res_connection=False, truncate_batch=True)
        self.learning_rate = lr
        
        self.vgg16 = torchvision.models.vgg16(pretrained=True).features[0:15]
        # adapt to 1-channel input
        vgg_w = self.vgg16[0].weight
        self.vgg16[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg16[0].weight = nn.Parameter(vgg_w.mean(dim=1).unsqueeze(1))
        self.vgg16.eval()
        
        #self.dcomp = tkbn.calc_density_compensation_function(traj, self.im_size)
        w = torch.Tensor(range(-160, 160)).abs().to(device=self.device, dtype=torch.complex64)
        w += 2
        w /= w.abs().max()
        w = torch.concat([w for i in range(11)])
        w = w.unsqueeze(0).unsqueeze(0)
        fact = 0.3
        w = torch.min(torch.Tensor([fact]).to(device=self.device), w.real).to(dtype=torch.complex64)
        w /= fact
        self.dcomp = w
        self.offset = 16
        
    def forward(self, batch):
        if len(batch) == 3:
            kspace, smaps, traj = batch
            kspace = kspace.squeeze()
            smaps = smaps.squeeze()
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
            
            x0 = self.pre_transform(self.dcomp*kdata, traj, smaps=smaps, norm='ortho')
        else:
            x0 = batch
        x_hat = self.cnn(x0.squeeze())
        return x_hat
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.cnn.parameters(), lr=self.learning_rate)
        lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.95)
        return {"optimizer" : optimizer, "lr_scheduler": lr_scheduler}
    
    def training_step(self, train_batch, batch_idx):
        
        with torch.no_grad():
            kspace, smaps, traj, gt, f_name = train_batch
            #print(f_name)
            
            kspace = kspace.squeeze()
            smaps = smaps.squeeze()
            traj = traj.squeeze()
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
            
            if str(self.dcomp.device) == 'cpu':
                self.dcomp = self.dcomp.to(self.device)
#             if self.dcomp is None:
#                 self.dcomp = tkbn.calc_density_compensation_function(traj, self.im_size)
            x0 = self.pre_transform(self.dcomp*kdata, traj, smaps=smaps, norm='ortho').squeeze()
            x0 = x0[:40, ...,80:240, 80:240]

#             factor = torch.quantile(x0.abs().flatten(), 0.95)
#             x0 = x0 / factor

            gt_coils = gt[0, :40, ...,80:240, 80:240]# * smaps


#             factor = torch.quantile(gt_coils.abs().flatten(), 0.95)
#             gt_coils = gt_coils / factor

        x_hat = self.cnn(x0)   
        
        x_hat_coils = x_hat.unsqueeze(1)# * smaps

        perc_x_real = self.vgg16(x_hat_coils.abs())
        perc_gt_real = self.vgg16(gt_coils.abs())
#         perc_x_imag = self.vgg19(x_hat_coils.imag)
#         perc_gt_imag = self.vgg19(gt_coils.imag)
        
        loss_1 = F.mse_loss(
            torch.view_as_real(x_hat_coils), torch.view_as_real(gt_coils), reduction='sum'
        )
        loss_2 = 1e-05 * (F.mse_loss(perc_x_real, perc_gt_real, reduction='sum'))# + F.mse_loss(perc_x_imag, perc_gt_imag, reduction='sum'))
                       
        combined_loss = loss_1 + loss_2
        self.log('L2', loss_1, on_step=True, on_epoch=True)
        self.log('Perc_loss', loss_2, on_step=True, on_epoch=True)
        self.log('combined_loss', combined_loss, on_step=True, on_epoch=True)
        return combined_loss
        
    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
             
            kspace, smaps, traj, gt, f_name = val_batch
            #print(f_name[0])
            
            kspace = kspace.squeeze()
            smaps = smaps.squeeze()
            traj = traj.squeeze()
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
            
            if str(self.dcomp.device) == 'cpu':
                self.dcomp = self.dcomp.to(self.device)
#             if self.dcomp is None:
#                 self.dcomp = tkbn.calc_density_compensation_function(traj, self.im_size)
            x0 = self.pre_transform(self.dcomp*kdata, traj, smaps=smaps, norm='ortho').squeeze()
            offset_val = 80
            x0 = x0[:40, ...,offset_val:-offset_val,offset_val:-offset_val]

#             factor = torch.quantile(x0.abs().flatten(), 0.95)
#             x0 = x0 / factor

            gt_coils = gt[0, :40, ...,offset_val:-offset_val,offset_val:-offset_val]# * smaps


#             factor = torch.quantile(gt_coils.abs().flatten(), 0.95)
#             gt_coils = gt_coils / factor
        x_hat = self.cnn(x0)

        # multi-coil loss
        x_hat_coils = x_hat.unsqueeze(1)# * smaps
        
        perc_x_real = self.vgg16(x_hat_coils.real)
        perc_gt_real = self.vgg16(gt_coils.real)
        perc_x_imag = self.vgg16(x_hat_coils.imag)
        perc_gt_imag = self.vgg16(gt_coils.imag)
        
        loss_1 = F.mse_loss(
            torch.view_as_real(x_hat_coils), torch.view_as_real(gt_coils), reduction='sum'
        )
        loss_2 = 1e-04 * (F.mse_loss(perc_x_real, perc_gt_real, reduction='sum') + F.mse_loss(perc_x_imag, perc_gt_imag, reduction='sum'))
        combined_loss = loss_1 + loss_2

        self.log('val_L2', loss_1, on_step=True, on_epoch=True)
        self.log('val_Perc_loss', loss_2, on_step=True, on_epoch=True)
        self.log('val_combined_loss', combined_loss, on_step=True, on_epoch=True)
