import torch
import torch.nn.functional as nn

def _retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=-3)
    output = flattened_tensor.gather(dim=-1, index=indices.flatten(start_dim=-3)).view_as(indices)
    return output

def interpolate(x, size=None, scale_factor=None, mode='nearest',
                             align_corners=None, recompute_scale_factor=None):
    '''
        Performs upsampling by separately interpolating the real and imaginary part and recombining
    '''
    outp_real = nn.interpolate(x.real,  size=size, scale_factor=scale_factor, mode=mode,
                                    align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    outp_imag = nn.interpolate(x.imag,  size=size, scale_factor=scale_factor, mode=mode,
                                    align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    
    return outp_real.type(torch.complex64) + 1j * outp_imag.type(torch.complex64)

def complex_max_pool2d(x,kernel_size, stride=None, padding=0,
                                dilation=1, ceil_mode=False, return_indices=False):
    '''
    Perform complex max pooling by selecting on the absolute value on the complex values.
    '''
    _, indices =  nn.max_pool3d(
                               x.abs(), 
                               kernel_size = kernel_size, 
                               stride = stride, 
                               padding = padding, 
                               dilation = dilation,
                               ceil_mode = ceil_mode, 
                               return_indices = True
                            )
    # performs the selection on the absolute values
    #breakpoint()
    # absolute_value = absolute_value.type(torch.complex64)
    # retrieve the corresonding phase value using the indices
    # unfortunately, the derivative for 'angle' is not implemented
    # angle = torch.atan2(x.imag,x.real)
    # get only the phase values selected by max pool
    return _retrieve_elements_from_indices(x, indices)
    #return absolute_value * (torch.cos(angle).type(torch.complex64)+1j*torch.sin(angle).type(torch.complex64))

def complex_avg_pool2d(x,kernel_size, stride=None, padding=0,
                                dilation=1, ceil_mode=False, return_indices=False):
    '''
    Perform complex max pooling by selecting on the absolute value on the complex values.
    '''
    x_real =  nn.avg_pool3d(
                               x.real, 
                               kernel_size = kernel_size, 
                               stride = stride, 
                               padding = padding,
                               ceil_mode = ceil_mode
                            )
    x_img =  nn.avg_pool3d(
                               x.imag, 
                               kernel_size = kernel_size, 
                               stride = stride, 
                               padding = padding,
                               ceil_mode = ceil_mode
                            )
    x = x_real + 1j*x_img
    # performs the selection on the absolute values
    #breakpoint()
    # absolute_value = absolute_value.type(torch.complex64)
    # retrieve the corresonding phase value using the indices
    # unfortunately, the derivative for 'angle' is not implemented
    # angle = torch.atan2(x.imag,x.real)
    # get only the phase values selected by max pool
    #return _retrieve_elements_from_indices(x, indices)
    #return absolute_value * (torch.cos(angle).type(torch.complex64)+1j*torch.sin(angle).type(torch.complex64))
    return x

def complex_relu(x):
    return nn.relu(x.real).type(torch.complex64)+1j*nn.relu(x.imag).type(torch.complex64)

def complex_lrelu(x):
    return nn.leaky_relu(x.real).type(torch.complex64)+1j*nn.leaky_relu(x.imag).type(torch.complex64)