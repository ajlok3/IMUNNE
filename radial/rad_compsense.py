import sys
import torch
from torch.autograd import grad as auto_grad
import torch.nn as nn
from torch.nn import MSELoss
import torchkbnufft as tkbn
import matplotlib.pyplot as plt
sys.path.append('..')
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
        traj=None, 
        lmbda=0.025, 
        regs=TTV_Regularizer(), 
        truncate_comp_graph=False, 
        update_t=True, 
        train_t=False,
        view_sharing=False,
        t=0.2,
    ):
        super().__init__()
        self.kdata = kdata
        self.smaps = smaps
        self.traj = traj
        
        ## TODO: fix hard coded
        self.numpoints = 3
        side = 320
        factor = 1
        self.im_size = (side,side)
        self.grid_size = (320,320)

        ##
        
        self.forward_op = tkbn.KbNufft(
            numpoints=self.numpoints, im_size=self.im_size, grid_size=self.grid_size
        ) #.to(device=self.device, dtype=self.dtype)
        
        self.adj_op = tkbn.KbNufftAdjoint(
            numpoints=self.numpoints, im_size=self.im_size, grid_size=self.grid_size
        ) #.to(device=self.device, dtype=self.dtype)
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
        self.train_t = train_t
        self.view_sharing = view_sharing
        
        self._dl_value = 0
        self._t = t
        # TODO: make this more beautiful
        try:
            if self.regs[0].cnn is not None:
                self.cnn = self.regs[0].cnn
                #self.t = nn.parameter.Parameter(data=torch.Tensor([t,]), requires_grad=self.train_t_lmbda)
#                 if train_t:
#                     self.lmbda_t = nn.parameter.Parameter(
#                         data=torch.Tensor([self.regs[0].lmbda,]), requires_grad=self.train_t_lmbda
#                     )
#                 else:
#                     self.lmbda_t = self.regs[0].lmbda
                
        except AttributeError:
            pass
        
    def load_batch(self, kdata, traj, smaps, dcomp):
        self.kdata = kdata
        self.traj = traj
        self.smaps = smaps
        self.dcomp = dcomp #tkbn.calc_density_compensation_function(traj, (40,40))
        
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
                print("tmp_max: ", tmp_max)
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
            ksp_new = self.forward_op(image, self.traj, smaps=self.smaps, norm='ortho')*self.dcomp
            with torch.no_grad():
                ksp_new /= self.dcomp
                
            if self.view_sharing:
                kspace_r = self.kdata
                ksp_new *= (kspace_r != 0)
            else:
                kspace_r = self.kdata #.permute(2,1,3,0).reshape(self.kdata.shape[2], self.kdata.shape[1], -1)   
           # with torch.no_grad():
                #ksp_new *= self.dcomp
                #kspace_r *= self.dcomp

            data_fidelity = nn.functional.mse_loss(torch.view_as_real(ksp_new), torch.view_as_real(kspace_r), reduction='sum')
            loss += data_fidelity
            print(", Data consistency = ", '{0:.8f}'.format(data_fidelity.item()))
        else:
            print("")
        # TODO: more effcient memory usage that makes the emptying of cache unnecessary
        # torch.cuda.empty_cache()
        print("Loss: ", '{0:.8f}'.format(loss.detach().item()))
        return loss


    def forward(self, *args, n_iter, t_forward=None, x0=None, show_frame=10, verbose=False, 
                           differentiable=False):
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
        print("lambda = [ttv:", self.lmbda_ttv, ", tv2d:", self.lmbda_tv2d, ", pca:", self.lmbda_pca, "]", flush=True)
        image = x0
        loss_args = [image]
        if len(args) > 0:
            loss_args.append(args[0])
        if self.train_t:
            t0 = 0.5 #self.t.data.item()
        else:
            t0 = t_forward #70000
        if verbose:
            print("Zero-filled recon:")
            show(image.detach().abs().cpu()[show_frame].squeeze().T)
            plt.show()

        # main reconctruction loop
        for i in range(n_iter):
            
            print("iteration: ", i+1)
            t = t0
            print("t = ", t)#.item())
            
            if not image.requires_grad:
                image.requires_grad = True
            # update regularization weights every 10 iterations
            update_weights = True           
            loss = self.calculate_loss(*loss_args, include_dc=True, update_weights=update_weights)
            #t = self._t
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
                if self._dl_value == 0:
                    self._dl_value = torch.abs(delta_m).sum()
                    dl_factor = 1
                else:
                    dl_factor = 1
                    #dl_factor = torch.abs(delta_m).sum()/self._dl_value
                    
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
                   
                
                    lower_bound = loss + self.alpha*t*grad_desc
                    lsiter = 0
                    while new_loss > lower_bound:
                        t = self.beta*t
                        new_loss_args[0] = image+ t*delta_m
                        new_loss = self.calculate_loss(*new_loss_args, include_dc=True)
                        
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
                
                #image = image + t*delta_m
                
                for reg in self.regs:
                    if reg.mode == 'output':
                        reg_term = reg.grad_fct(*loss_args)
                        #reg_term = reg_term * tmp_max / reg_term.abs().max()
                        
                        #image = (1-self.lmbda_t*dl_factor)*image + self.lmbda_t*dl_factor*reg_term
                        
                        image = image + t*(delta_m - reg_term)
                        #image = image + t*delta_m + self.lmbda_t*reg_term
                        #image = image / image.abs().max()

            loss_args[0] = image
            
            if verbose and i % 5 == 0:
                show(image.detach().abs().cpu()[show_frame].squeeze().T)
                plt.show()
        self.image = image.detach().clone()
        return image