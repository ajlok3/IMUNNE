import torch
from torch.linalg import eig
from torch_pca import torch_pca

def coil_pca_torch(kspace, virtual_coils, dim=0):
        ''' 
            performs dimensionality reduction in the coil domain to reduce the amount of data
            Args: kspace (torch.tensor): original kspace data to be coil-reduced
                  vitual_coils (int): number of virtual coils to which the coil dimension is to be reduced
                  dim (int): location of the coil dimension
            Returns: kspace data with reduced coil dimension
        '''
        # prep
        rest_ind = [i for i in range(len(kspace.shape)) if i != dim] # all indices
        
        # flatten the original kspace data for each coil
        orig = kspace.permute(dim, *rest_ind)
        orig = orig.reshape(orig.shape[0], -1)

        ## eigenvalue based
        #covariance = torch.cov(orig)
        #val, vec = eig(covariance)
        #vec = vec[:, :virtual_coils]
        #new_data = (orig.T @ vec).T
        
        ## svd based
        new_data = torch_pca(orig.T, virtual_coils).T
        
        # reshape the new_data into the original shape
        new_data = new_data.reshape(
            virtual_coils, *[kspace.shape[i] for i in rest_ind])
        new_data = new_data.permute(
            *range(1, dim+1),0, *range(dim+1, len(kspace.shape)))
        return new_data