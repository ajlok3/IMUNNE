import math
import torch
import torch.nn as nn
import numpy as np
import cu_net.complex_pytorch_functions as cpf
import cu_net.complex_pytorch_layers as cpl

pad_mode_global = 'circular'
batch_norm = False
layer_norm = False
instance_norm = False
channel_attention = False

class ECA(nn.Module):
    
    def __init__(self, gamma=2, b=1, c=2, dev='cpu'):
        super(ECA, self).__init__()
        # gamma, b: parameters of mapping function
        t = int(abs((math.log2(c) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1).to(dev)
        self.conv = nn.Conv1d(1,1, kernel_size=k, padding=int(k/2), bias=False).to(dev)
        self.sigmoid = nn.Sigmoid().to(dev)
        
    def forward(self, x):
        # x: input features with shape [N, C, D, H, W]
        
        y = self.avg_pool(x.abs())
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1,-2))
        y = y.transpose(-1,-2).unsqueeze(-1).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)
    
    
class ConvBlock(nn.Module):

    """
    A block of convolutional layers (1D, 2D or 3D)
    """
    
    def __init__(self, dim, n_ch_in, n_ch_out, n_convs, sp_kernel_size=3, t_kernel_size=3, bias = False, dev='cpu'):
        super(ConvBlock, self).__init__()

        if dim==1:
            conv_op = nn.Conv1d
        if dim==2:
            conv_op = nn.Conv2d
        elif dim==3:
            conv_op = cpl.ComplexConv3d
                
        sp_padding = [int(np.floor(sp_kernel_size/2)), int(np.floor(sp_kernel_size/2)), 0]
        t_padding = [0, 0, int(np.floor(t_kernel_size/2))] 
        #sp_padding = int(np.floor(sp_kernel_size/2))
        #t_padding = int(np.floor(t_kernel_size/2))
        
        conv_block_list = []
        conv_block_list.extend(
            [conv_op(
                n_ch_in, 
                n_ch_out, 
                kernel_size=(sp_kernel_size,sp_kernel_size,1), 
                padding = sp_padding, 
                padding_mode=pad_mode_global, 
                bias=bias
            ),                   
             conv_op(
                 n_ch_out, 
                 n_ch_out, 
                 kernel_size=(1,1,t_kernel_size), 
                 padding = t_padding, 
                 padding_mode=pad_mode_global, 
                 bias=bias
             ),
                                
             #cpl.ComplexReLU()]
             cpl.ComplexLReLU()]
        )
        if channel_attention:
            conv_block_list.insert(-2, ECA(c=n_ch_out, dev=dev))
            conv_block_list.insert(-1, ECA(c=n_ch_out, dev=dev))
        
        if batch_norm:
            conv_block_list.insert(-1,cpl.ComplexBatchNorm3d(n_ch_out))
        if layer_norm:
            conv_block_list.insert(-1,cpl.ComplexLayerNorm3d([n_ch_out, 160//n_ch_out, 1280//n_ch_out, 1280//n_ch_out]))
        if instance_norm:
            conv_block_list.insert(-1,cpl.ComplexLayerNorm3d([160//n_ch_out, 1280//n_ch_out, 1280//n_ch_out]))
            
        
        for i in range(n_convs-1):
            conv_block_list.extend(
                [conv_op(
                    n_ch_out, 
                    n_ch_out, 
                    kernel_size=(sp_kernel_size,sp_kernel_size,1),
                    padding = sp_padding, 
                    padding_mode=pad_mode_global, 
                    bias=bias
                ),
                                
                 conv_op(
                     n_ch_out, 
                     n_ch_out, 
                     kernel_size=(1,1,t_kernel_size), 
                     padding = t_padding, 
                     padding_mode=pad_mode_global, 
                     bias=bias
                 ),
                                
                 #cpl.ComplexReLU()]
                 cpl.ComplexLReLU()]
                 
            )
            if channel_attention:
                conv_block_list.insert(-2, ECA(c=n_ch_out, dev=dev))
                conv_block_list.insert(-1, ECA(c=n_ch_out, dev=dev))
            if batch_norm:
                conv_block_list.insert(-1, cpl.ComplexBatchNorm3d(n_ch_out))
            if layer_norm:
                conv_block_list.insert(-1,cpl.ComplexLayerNorm3d([n_ch_out, 160//n_ch_out, 1280//n_ch_out, 1280//n_ch_out]))
            if instance_norm:
                conv_block_list.insert(-1,cpl.ComplexLayerNorm3d([160//n_ch_out, 1280//n_ch_out, 1280//n_ch_out]))
                
        self.conv_block = nn.Sequential(*conv_block_list)

    def forward(self, x):
        return self.conv_block(x)


class Encoder(nn.Module):
    def __init__(self, dim, n_ch_in, n_enc_stages, n_convs_per_stage, n_filters, sp_kernel_size=3, t_kernel_size=3, bias = False, dev='cpu'):
        super(Encoder, self).__init__()

        n_ch_list = [n_ch_in]
        for ne in range(n_enc_stages):
            n_ch_list.append(int(n_filters)*2**ne)

        self.enc_blocks = nn.ModuleList([
            ConvBlock(
                dim, 
                n_ch_list[i], 
                n_ch_list[i+1], 
                n_convs_per_stage, 
                sp_kernel_size=sp_kernel_size,
                t_kernel_size=t_kernel_size,
                dev=dev
            ) for i in range(len(n_ch_list)-1)])

        if dim == 1:
            pool_op = nn.MaxPool1d(2)
        elif dim == 2:
            pool_op = nn.MaxPool2d(2)
        elif dim == 3:
            #pool_op = cpl.ComplexMaxPool3d(2)
            pool_op = cpl.ComplexAvgPool3d(2)
            
        self.pool = pool_op

    def forward(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features


class Decoder(nn.Module):
    def __init__(self, dim, n_ch_in,  n_dec_stages, n_convs_per_stage, n_filters, sp_kernel_size=3, t_kernel_size=3, bias=False, dev='cpu'):
        super(Decoder, self).__init__()

        n_ch_list = []
        for ne in range(n_dec_stages):
            n_ch_list.append(int(n_ch_in*(1/2)**ne))

        if dim == 1:
            interp_mode = 'linear'
            conv_op = nn.Conv1d
        elif dim == 2:
            conv_op = nn.Conv2d
            interp_mode = 'bilinear'
        elif dim == 3:
            interp_mode = 'trilinear'
            conv_op = cpl.ComplexConv3d

        self.interp_mode = interp_mode

        sp_padding = [int(np.floor(sp_kernel_size/2)), int(np.floor(sp_kernel_size/2)), 0]
        t_padding = [0, 0, int(np.floor(t_kernel_size/2))]
        #sp_padding = int(np.floor(sp_kernel_size/2))
        #t_padding = int(np.floor(t_kernel_size/2))
        self.sp_upconvs = nn.ModuleList([conv_op(n_ch_list[i], n_ch_list[i+1],  kernel_size=(sp_kernel_size, sp_kernel_size, 1), padding=sp_padding, padding_mode=pad_mode_global, bias=bias) for i in range(len(n_ch_list)-1)])
        self.t_upconvs = nn.ModuleList([conv_op(n_ch_list[i+1], n_ch_list[i+1],  kernel_size=(1,1,t_kernel_size), padding=t_padding, padding_mode=pad_mode_global, bias=bias) for i in range(len(n_ch_list)-1)])
        
        self.dec_blocks = nn.ModuleList([
            ConvBlock(
                dim, 
                n_ch_list[i], 
                n_ch_list[i+1], 
                n_convs_per_stage, 
                sp_kernel_size=sp_kernel_size, 
                t_kernel_size=t_kernel_size, 
                bias=bias,
                dev=dev
            ) for i in range(len(n_ch_list)-1)])


    def forward(self, x, encoder_features):

        for i in range(len(self.dec_blocks)):
            #x        = self.upconvs[i](x)
            enc_features = encoder_features[i]
            enc_features_shape = enc_features.shape
            x = cpf.interpolate(x, enc_features_shape[2:], mode = self.interp_mode)
            x = self.sp_upconvs[i](x)
            x = self.t_upconvs[i](x)
            x = torch.cat([x, enc_features], dim=1)
            x = self.dec_blocks[i](x)
        return x


class Complex2DtUNet(nn.Module):
    def __init__(self, dim, n_ch_in=1, n_ch_out = 1, n_enc_stages=3, n_convs_per_stage=2, n_filters=8, sp_kernel_size=3, t_kernel_size=3, res_connection=False, bias=False, truncate_batch=True, dev='cpu'):
        super(Complex2DtUNet, self).__init__()
        self.truncate_batch = truncate_batch
        self.encoder = Encoder(dim, n_ch_in, n_enc_stages,  n_convs_per_stage, n_filters, sp_kernel_size=sp_kernel_size, t_kernel_size=t_kernel_size, bias = bias, dev=dev)
        self.decoder = Decoder(dim, n_filters*(2**(n_enc_stages-1)), n_enc_stages, n_convs_per_stage, n_filters*(n_enc_stages*2), sp_kernel_size=sp_kernel_size, t_kernel_size=t_kernel_size, bias=bias, dev=dev)

        if dim == 1:
            conv_op = nn.Conv1d
        elif dim == 2:
            conv_op = nn.Conv2d
        elif dim == 3:
            conv_op = cpl.ComplexConv3d

        self.c1x1     = conv_op(n_filters, n_ch_out, kernel_size=1, padding_mode=pad_mode_global, bias=bias)
        self.res_connection  = res_connection

    def forward(self, x):
        if self.truncate_batch:
            while len(x.shape) < 5:
                x = x.unsqueeze(0)
        if self.res_connection:
            x_in = x.clone()
        
        enc_features = self.encoder(x)
        x      = self.decoder(enc_features[-1], enc_features[::-1][1:])
        x      = self.c1x1(x)
        if self.res_connection:
            x = x_in + x
        if self.truncate_batch:
            x = x.squeeze()
        return x
