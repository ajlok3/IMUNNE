import os
import torch
import functools
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

# enumerate the entries in the directory
path = os.environ["DATA_PATH"]
modality = 'Rad_noise/sigma_0.1/'
crop = True

path = path + modality
ksp_sfx = 'kspace/'
smaps_sfx = 'smaps/'
traj_sfx = 'traj/'

cs_recon_sfx = 'cs_recon/'

testing_list = sorted(os.listdir(path + 'testing/' + cs_recon_sfx))
n_testing_samples = len(testing_list)
import pdb;pdb.set_trace()

pre_cutoff = 5
cutoff = 45


#########
# Radial
#########

@functools.lru_cache(1, typed=True)
def getTestingTupleRadial(n):
    mode = "testing/"
    return torch.load(
            path + mode + ksp_sfx + testing_list[n].replace("cs_recon", "kspace")
        )[:,:,pre_cutoff:cutoff], torch.load(
            path + mode + smaps_sfx + testing_list[n].replace("cs_recon", "smaps").replace('_sigma0.1', '')
        ), torch.load(
            path + mode + traj_sfx + testing_list[n].replace("cs_recon", "traj").replace('_sigma0.1', '')
        )[pre_cutoff:cutoff], torch.load(
            path + mode + cs_recon_sfx + testing_list[n] #.replace("cs_recon", "cs_recon")
        )[pre_cutoff:cutoff], testing_list[n]


######################
# Data sets & loaders
######################    
    
class TestingDataset(Dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        return n_testing_samples
    
    def __getitem__(self, ndx):
        return getTestingTupleRadial(ndx)