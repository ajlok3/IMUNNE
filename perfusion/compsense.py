import torch
from torch.autograd import grad as auto_grad
import torch.nn as nn
from torch.nn import MSELoss
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from dyn_fft import DynFFT
from dyn_ifft import DyniFFT
from util import show
from regularizer import TTV_Regularizer, TV2D_Regularizer, TPCA_Regularizer

class CompressedSensing(nn.Module):
    '''
        New class for compressed sensing for Cartesian data
    '''
    def __init__(
        self, 
        kdata=None, 
        smaps=None,
        lmbda=0.025, 
        regs=TTV_Regularizer(), 
        truncate_comp_graph=False, 
        update_t=True, 
        train_t_lmbda=False, 
        t=1,
    ):
        super().__init__()
        self.kdata = kdata
        if kdata is None:
            cs_mask = None
        else:
            cs_mask = (kdata != 0)
        self.forward_op = DynFFT(smaps=smaps, cs_mask=cs_mask)
        self.adj_op = DyniFFT(smaps=smaps, cs_mask=cs_mask)
        self.lmbda_ttv = lmbda
        self.lmbda_pca = lmbda*0.5
        self.lmbda_tv2d = lmbda*0
        if not isinstance(regs, list):
            regs = [regs]
        self.regs = regs
        
        self.alpha = 0.01
        self.beta = 0.6
        self.truncate_comp_graph = truncate_comp_graph
        self.update_t = update_t
        self.train_t_lmbda = train_t_lmbda

        # TODO: make this more beautiful
        try:
            if self.regs[0].cnn is not None:
                self.cnn = self.regs[0].cnn
                self.t = nn.parameter.Parameter(data=torch.Tensor([t,]), requires_grad=self.train_t_lmbda)
                if train_t_lmbda:
                    self.lmbda_t = nn.parameter.Parameter(
                        data=torch.Tensor([self.regs[0].lmbda,]), requires_grad=self.train_t_lmbda
                    )
                else:
                    self.lmbda_t = self.regs[0].lmbda
                
        except AttributeError:
            pass
        
    def load_batch(self, kdata, smaps):
        self.kdata = kdata
        cs_mask = (kdata != 0)
        self.forward_op.load_data(smaps, cs_mask)
        self.adj_op.load_data(smaps, cs_mask)
        
    def calculate_loss(self, *args, include_dc=False, update_weights=False):
        ''' Calculation of the CS loss function

            lambda*R(image)
            R(.): generic regularizer

            Args:
                *args: function call parameters passed to the regularizer
                include_dc (Bool): whether to add the data consistency term 
                                   || F(image) - kspace||_2^2 or not
                update_weights (Bool): if True the regularization weights 

            Returns:
                loss (torch.tensor): loss value with the autograd fuctionality
        '''
        # image state is assumed to be the first argument always
        image = args[0]
        # regularizer term
        if update_weights:
            with torch.no_grad():
                tmp_max = image.detach().abs().max()
                for reg in self.regs:
                    if isinstance(reg, TTV_Regularizer):
                        reg.set_lmbda(tmp_max * self.lmbda_ttv)
                    elif isinstance(reg, TV2D_Regularizer):
                        reg.set_lmbda(tmp_max * self.lmbda_tv2d)                        
                    elif isinstance(reg, TPCA_Regularizer):
                        reg.set_lmbda(tmp_max * self.lmbda_pca)
                     
        reg_term = 0
        for reg in self.regs:
            if reg.mode == 'loss':
                reg_term += reg.lmbda * reg.loss_fct(*args)

        if reg_term != 0:
            with torch.no_grad():
                print("Regularization = ", '{0:.8f}'.format(reg_term.item()), end="")
        
        loss = reg_term

        if include_dc:
            # L2 loss in k-space
            ksp_new = self.forward_op(image)
            kspace_r = self.kdata            
            
            data_fidelity = nn.functional.mse_loss(torch.view_as_real(ksp_new), torch.view_as_real(kspace_r), reduction='sum')
            loss += data_fidelity
            print(", Data consistency = ", '{0:.8f}'.format(data_fidelity.item()))
        else:
            print("")
        # TODO: more effcient memory usage that makes the emptying of cache unnecessary
        # torch.cuda.empty_cache()
        print("Loss: ", '{0:.8f}'.format(loss.detach().item()))
        return loss


    def forward(self, *args, n_iter, x0=None, show_frame=10, verbose=False, 
                           differentiable=False, eps=0):
        ''' Multicoil compressed sensing algorithm for MRI implemented with pytorch

            Args:
                n_iter (int): number of iterations: TODO: stopping criterion
                x0 (torch.tensor): initial image to start the CS reconstruction;
                                   if None the internal image state is taken as x0
                show_frame (int): number of the frame displayed every second iteration;
                                  only valid if verbose=True
                verbose (bool):   controls the amount of information printed to the console
                                  during reconstruction

            Returns:
                out (torch.tensor): reconstructed 2D+t-data
        '''
        use_eps = (n_iter == 0 or n_iter is None)
        if use_eps:
            n_iter = 100000 # enough iterations
        print("lambda = [ttv:", self.lmbda_ttv, ", tv2d:", self.lmbda_tv2d, ", pca:", self.lmbda_pca, "]", flush=True)
        # initial guess
    #     if x0 is None:
    #         image = self.image
    #     else:
    #         image = x0
        image = x0
        loss_args = [image]
        if len(args) > 0:
            loss_args.append(args[0])
        if self.train_t_lmbda:
            t0 = self.t
        else:
            t0 = 0.9
        if verbose:
            print("Zero-filled recon:")
            show(image.detach().abs().cpu()[show_frame].squeeze().T)
            plt.show()
        
        # main reconctruction loop
        for i in range(n_iter):
            
            print("iteration: ", i+1)
            t = t0
            if not image.requires_grad:
                image.requires_grad = True
            # update regularization weights every 10 iterations
            update_weights = True           
            loss = self.calculate_loss(*loss_args, include_dc=True, update_weights=update_weights)
            
            # auto_grad for second order derivative
            create_graph = differentiable and (not self.truncate_comp_graph or i == n_iter - 1)
            im_grad, = auto_grad(outputs=loss, inputs=image, create_graph=create_graph)

            for reg in self.regs:
                if reg.mode == 'grad':
                    with torch.set_grad_enabled(create_graph):
                        im_grad += reg.lmbda * reg.grad_fct(*loss_args)
            # optimizer: conjugate gradient descent scheme
            if i==0:
                grad = im_grad
                delta_m = -grad
            else:
                tmp = grad
                grad = im_grad
                gamma = torch.sum(torch.abs(grad)**2)/torch.sum(torch.abs(tmp)**2)
                delta_m = -grad + gamma*delta_m
            # this part can be detached from autograd because its purpose is
            # to find t, a scalar constant
            with torch.no_grad():
                if self.update_t:
                    grad_desc = torch.mean(torch.real(grad.conj()*delta_m))
                    new_loss_args = [image+t*delta_m]
                    if len(args) > 0:
                        new_loss_args.append(args[0])
                    new_loss = self.calculate_loss(*new_loss_args, include_dc=True)
                    if use_eps and new_loss.item() < eps:
                        print("Epsilon condition fulfilled!", flush=True)
                        self.image = image.detach().clone()
                        return image
                
                    lower_bound = loss + self.alpha*t*grad_desc
                    lsiter = 0
                    while new_loss > lower_bound:
                        t = self.beta*t
                        new_loss_args[0] = image+t*delta_m
                        new_loss = self.calculate_loss(*new_loss_args, include_dc=True)
                        if use_eps and new_loss.item() < eps:
                            print("Epsilon condition fulfilled!", flush=True)
                            self.image = image.detach().clone()
                            return image
                        print("t=", '{0:.8f}'.format(t))
                        if t < 1e-15:
                            print("Calculated gradient is not a descent descent direction. Resetting t to it's original value: t=", t0)
                            t = t0
                            break
                        lsiter += 1
                    if lsiter == 0:
                        t0 = t0 / self.beta
                    if lsiter > 2:
                        t0 = t0 * self.beta
                    print("iteration loss = ", new_loss.item())
            differentiable_iteration = differentiable and (not self.truncate_comp_graph or i == n_iter - 1)
            with torch.set_grad_enabled(differentiable_iteration):
                image = image + t*delta_m
                tmp_max = image.abs().max()
                
                for reg in self.regs:
                    if reg.mode == 'output':
                        reg_term = reg.grad_fct(*loss_args).squeeze()                      
                        #reg_term = reg_term * tmp_max / reg_term.abs().max()
                        image = (1-self.lmbda_t)*image + self.lmbda_t*reg_term
            
            loss_args[0] = image
            
            if verbose and i % 5 == 0:
                show(image.detach().abs().cpu()[show_frame].squeeze().T)
                plt.show()
        self.image = image.detach().clone()
        return image