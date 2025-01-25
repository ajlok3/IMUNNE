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

sys.path.append('..')
sys.path.append('../util')

from cinenet.complex_unet import Complex2DtUNet
from regularizer import DL_Regularizer
from load_scan import calc_cs_mask
from dyn_fft import DynFFT
from dyn_ifft import DyniFFT

class CineNet(pl.LightningModule):
    '''
        Pytorch lightning wrapper for 
    '''
    def __init__(self, lr=1e-03):
        super().__init__()
        side = 384
        grid_side = 384
        self.im_size = (side,side)
        grid_size = (grid_side, grid_side)        
        self.pre_transform = DyniFFT()
        self.cnn = Complex2DtUNet(dim=3, n_filters=4, n_enc_stages=3, res_connection=False, truncate_batch=True)
        self.learning_rate = lr
        
        self.vgg16 = torchvision.models.vgg16(pretrained=True).features[0:15]
        # adapt to 1-channel input
        vgg_w = self.vgg16[0].weight
        self.vgg16[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg16[0].weight = nn.Parameter(vgg_w.mean(dim=1).unsqueeze(1))
        self.vgg16.eval()
        
        
    def forward(self, batch):
        if len(batch) == 3:
            kspace, smaps = batch
            kspace = kspace.squeeze()
            smaps = smaps.squeeze()
            cs_mask_3d = calc_cs_mask(kspace)
            self.pre_transform.load_data(smaps, cs_mask_3d)
            
            kdata = kspace.permute(2,1,3,0).reshape(kspace.shape[2], kspace.shape[1], -1)    
            x0 = self.pre_transform(kdata)
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
            kspace, smaps, gt, = train_batch            
            kspace = kspace.squeeze()
            smaps = smaps.squeeze()
            cs_mask_3d = calc_cs_mask(kspace)
            self.pre_transform.load_data(smaps, cs_mask_3d)
            
           
            x0 = self.pre_transform(kspace).squeeze()

#             factor = torch.quantile(x0.abs().flatten(), 0.95)
#             x0 = x0 / factor

            gt_coils = gt[0].unsqueeze(1)# * smaps


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
            kspace, smaps, gt, = val_batch            
            kspace = kspace.squeeze()
            smaps = smaps.squeeze()
            cs_mask_3d = calc_cs_mask(kspace)
            self.pre_transform.load_data(smaps, cs_mask_3d)
           
            x0 = self.pre_transform(kspace).squeeze()

#             factor = torch.quantile(x0.abs().flatten(), 0.95)
#             x0 = x0 / factor

            gt_coils = gt[0].unsqueeze(1)# * smaps


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
