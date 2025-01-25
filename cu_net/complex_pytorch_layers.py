import torch
import torch.nn as nn
from cu_net.complex_pytorch_functions import complex_max_pool2d, complex_avg_pool2d, complex_relu, complex_lrelu

def apply_complex(fr, fi, x, dtype = torch.complex64):
    return (fr(x.real)-fi(x.imag)).type(dtype) \
            + 1j*(fr(x.imag)+fi(x.real)).type(dtype)


class ComplexMaxPool3d(nn.Module):

    def __init__(self,kernel_size, stride= None, padding = 0,
                 dilation = 1, return_indices = False, ceil_mode = False):
        super(ComplexMaxPool3d,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self,x):
        return complex_max_pool2d(x,kernel_size = self.kernel_size,
                                stride = self.stride, padding = self.padding,
                                dilation = self.dilation, ceil_mode = self.ceil_mode,
                                return_indices = self.return_indices)

class ComplexAvgPool3d(nn.Module):

    def __init__(self,kernel_size, stride= None, padding = 0,
                 dilation = 1, return_indices = False, ceil_mode = False):
        super(ComplexAvgPool3d,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self,x):
        return complex_avg_pool2d(x,kernel_size = self.kernel_size,
                                stride = self.stride, padding = self.padding,
                                dilation = self.dilation, ceil_mode = self.ceil_mode,
                                return_indices = self.return_indices)
    
    
class ComplexReLU(nn.Module):
    
    def forward(self,x):
        return complex_relu(x)

class ComplexLReLU(nn.Module):
    
    def forward(self,x):
        return complex_lrelu(x)
    

class ComplexConv3d(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0, padding_mode = 'zeros',
                 dilation=1, groups=1, bias=True):
        super(ComplexConv3d, self).__init__()
        self.conv_r = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.conv_i = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        
    def forward(self,x):    
        return apply_complex(self.conv_r, self.conv_i, x)

class ComplexBatchNorm3d(nn.Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, \
                 track_running_stats=True):
        super(ComplexBatchNorm3d, self).__init__()
        self.bn_r = nn.BatchNorm3d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = nn.BatchNorm3d(num_features, eps, momentum, affine, track_running_stats)
    def forward(self,input):
        return self.bn_r(input.real).type(torch.complex64) +1j*self.bn_i(input.imag).type(torch.complex64)

class ComplexLayerNorm3d(nn.Module):
    '''
    Naive approach to complex layer norm, perform layer norm independently on real and imaginary part.
    '''
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(ComplexLayerNorm3d, self).__init__()
        self.ln_r = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
        self.ln_i = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self,input):
        return self.ln_r(input.real).type(torch.complex64) +1j*self.ln_i(input.imag).type(torch.complex64)