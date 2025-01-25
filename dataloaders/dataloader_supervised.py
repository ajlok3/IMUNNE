import os
import torch
import functools
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

# enumerate the entries in the directory
path = os.environ["DATA_PATH"]
modality = 'Rad_crossvalidation/'
crop = True

path = path + modality
ksp_sfx = 'kspace/'
smaps_sfx = 'smaps/'
traj_sfx = 'traj/'

cs_recon_sfx = 'cs_recon/'


training_list = os.listdir(path + 'training/' + cs_recon_sfx)#[:1]
validation_list = os.listdir(path + 'validation/' + cs_recon_sfx)

testing_list = sorted(os.listdir(path + 'testing/' + cs_recon_sfx))


#n_training_samples = 1
n_training_samples = len(training_list)
n_testing_samples = len(testing_list)
n_validation_samples = len(validation_list)

pre_cutoff = 5
cutoff = 45

###########
# Cartesian

@functools.lru_cache(1, typed=True)
def getTrainingTuple(n):
    mode = "training/"
    return torch.load(
            path + mode + ksp_sfx + training_list[n].replace("cs_recon", "kspace")
        )[pre_cutoff:cutoff], torch.load(
            path + mode + smaps_sfx + training_list[n].replace("cs_recon", "smaps")
        ), torch.load(
            path + mode + cs_recon_sfx + training_list[n] #.replace("cs_recon", "cs_recon")
        )[pre_cutoff:cutoff]

@functools.lru_cache(1, typed=True)
def getValidationTuple(n):
    mode = "validation/"
    return torch.load(
            path + mode + ksp_sfx + validation_list[n].replace("cs_recon", "kspace")
        )[pre_cutoff:cutoff], torch.load(
            path + mode + smaps_sfx + validation_list[n].replace("cs_recon", "smaps")
        ), torch.load(
            path + mode + cs_recon_sfx + validation_list[n] #.replace("cs_recon", "cs_recon")
        )[pre_cutoff:cutoff]

@functools.lru_cache(1, typed=True)
def getTestingTuple(n):
    mode = "testing/"
    return torch.load(
            path + mode + ksp_sfx + testing_list[n].replace("cs_recon", "kspace")
        )[pre_cutoff:cutoff], torch.load(
            path + mode + smaps_sfx + testing_list[n].replace("cs_recon", "smaps")
        ), torch.load(
            path + mode + cs_recon_sfx + testing_list[n] #.replace("cs_recon", "cs_recon")
        )[pre_cutoff:cutoff]

#########
# Radial
#########
@functools.lru_cache(1, typed=True)
def getTrainingTupleRadial(n):
    mode = "training/"
    return torch.load(
            path + mode + ksp_sfx + training_list[n].replace("cs_recon", "kspace")
        )[:,:,pre_cutoff:cutoff], torch.load(
            path + mode + smaps_sfx + training_list[n].replace("cs_recon", "smaps")
        ), torch.load(
            path + mode + traj_sfx + training_list[n].replace("cs_recon", "traj")
        )[pre_cutoff:cutoff], torch.load(
            path + mode + cs_recon_sfx + training_list[n] #.replace("cs_recon", "cs_recon")
        )[pre_cutoff:cutoff,:], training_list[n]

@functools.lru_cache(1, typed=True)
def getValidationTupleRadial(n):
    mode = "validation/"
    return torch.load(
            path + mode + ksp_sfx + validation_list[n].replace("cs_recon", "kspace")
        )[:,:,pre_cutoff:cutoff], torch.load(
            path + mode + smaps_sfx + validation_list[n].replace("cs_recon", "smaps")
        ), torch.load(
            path + mode + traj_sfx + validation_list[n].replace("cs_recon", "traj")
        )[pre_cutoff:cutoff], torch.load(
            path + mode + cs_recon_sfx + validation_list[n] #.replace("cs_recon", "cs_recon")
        )[pre_cutoff:cutoff], validation_list[n]

@functools.lru_cache(1, typed=True)
def getTestingTupleRadial(n):
    mode = "testing/"
    return torch.load(
            path + mode + ksp_sfx + testing_list[n].replace("cs_recon", "kspace")
        )[:,:,pre_cutoff:cutoff], torch.load(
            path + mode + smaps_sfx + testing_list[n].replace("cs_recon", "smaps")
        ), torch.load(
            path + mode + traj_sfx + testing_list[n].replace("cs_recon", "traj")
        )[pre_cutoff:cutoff], torch.load(
            path + mode + cs_recon_sfx + testing_list[n] #.replace("cs_recon", "cs_recon")
        )[pre_cutoff:cutoff], testing_list[n]

#####
# GRE
#####
def _getTrainingTupleGRE(n):    
    mode = "training/"
    return torch.load(
            path + mode + ksp_sfx + training_list[n].replace("cs_recon", "kspace")
        )[:,:,pre_cutoff:cutoff], torch.load(
            path + mode + smaps_sfx + training_list[n].replace("cs_recon", "smaps")
        ), torch.load(
            path + mode + traj_sfx + training_list[n].replace("cs_recon", "traj")
        )[pre_cutoff:cutoff], torch.load(
            path + mode + cs_recon_sfx + training_list[n] #.replace("cs_recon", "cs_recon")
        )[pre_cutoff:cutoff,:], training_list[n]

@functools.lru_cache(1, typed=True)
def getTrainingTupleGRE(n):
    kspace, smaps, traj, gt, name = _getTrainingTupleGRE(n)
    if crop:
        # adjust GRE data
        offs = 32
        kspace = kspace[offs:-offs]
        smaps = smaps[:,offs:-offs,offs:-offs]
        gt = gt[...,offs:-offs,offs:-offs]
        traj = traj.reshape(*traj.shape[:2],-1,384)
        traj = traj[...,offs:-offs]
        traj = traj.reshape(*traj.shape[:2],-1)
        # adjust GRE data
    return kspace, smaps, traj, gt, name
    
def _getValidationTupleGRE(n):
    mode = "validation/"
    return torch.load(
            path + mode + ksp_sfx + validation_list[n].replace("cs_recon", "kspace")
        )[:,:,pre_cutoff:cutoff], torch.load(
            path + mode + smaps_sfx + validation_list[n].replace("cs_recon", "smaps")
        ), torch.load(
            path + mode + traj_sfx + validation_list[n].replace("cs_recon", "traj")
        )[pre_cutoff:cutoff], torch.load(
            path + mode + cs_recon_sfx + validation_list[n] #.replace("cs_recon", "cs_recon")
        )[pre_cutoff:cutoff], validation_list[n]

@functools.lru_cache(1, typed=True)
def getValidationTupleGRE(n):
    kspace, smaps, traj, gt, name = _getValidationTupleGRE(n)
    if crop:
        # adjust GRE data
        offs = 32
        kspace = kspace[offs:-offs]
        smaps = smaps[:,offs:-offs,offs:-offs]
        gt = gt[...,offs:-offs,offs:-offs]
        traj = traj.reshape(*traj.shape[:2],-1,384)
        traj = traj[...,offs:-offs]
        traj = traj.reshape(*traj.shape[:2],-1)
        # adjust GRE data
    return kspace, smaps, traj, gt, name

def _getTestingTupleGRE(n):
    mode = "testing/"
    return torch.load(
            path + mode + ksp_sfx + testing_list[n].replace("cs_recon", "kspace")
        )[:,:,pre_cutoff:cutoff], torch.load(
            path + mode + smaps_sfx + testing_list[n].replace("cs_recon", "smaps")
        ), torch.load(
            path + mode + traj_sfx + testing_list[n].replace("cs_recon", "traj")
        )[pre_cutoff:cutoff], torch.load(
            path + mode + cs_recon_sfx + testing_list[n] #.replace("cs_recon", "cs_recon")
        )[pre_cutoff:cutoff], testing_list[n]

@functools.lru_cache(1, typed=True)
def getTestingTupleGRE(n):
    kspace, smaps, traj, gt, name = _getTestingTupleGRE(n)
    if crop:
        # adjust GRE data
        offs = 32
        kspace = kspace[offs:-offs]
        smaps = smaps[:,offs:-offs,offs:-offs]
        gt = gt[...,offs:-offs,offs:-offs]
        traj = traj.reshape(*traj.shape[:2],-1,384)
        traj = traj[...,offs:-offs]
        traj = traj.reshape(*traj.shape[:2],-1)
        # adjust GRE data
    return kspace, smaps, traj, gt, name

######################
# Data sets & loaders
######################
class TrainingDataset(Dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        return n_training_samples
    
    def __getitem__(self, ndx):
        if modality == 'Cart/' or modality == 'Perfusion/':
            return getTrainingTuple(ndx)
        elif modality == 'Rad/' or modality == 'PC/' or modality.startswith('Rad_'):
            return getTrainingTupleRadial(ndx)
        elif modality == 'GRE/' or modality == 'Radial_perfusion/':
            return getTrainingTupleGRE(ndx)
            
            
class ValidationDataset(Dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        return n_validation_samples
    
    def __getitem__(self, ndx):
        if modality == 'Cart/' or modality == 'Perfusion/':
            return getValidationTuple(ndx)
        elif modality == 'Rad/' or modality == 'PC/' or modality.startswith('Rad_'):
            return getValidationTupleRadial(ndx)
        elif modality == 'GRE/' or modality == 'Radial_perfusion/':
            return getValidationTupleGRE(ndx)
    
    
class TestingDataset(Dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        return n_testing_samples
    
    def __getitem__(self, ndx):
        if modality == 'Cart/' or modality == 'Perfusion/':
            return getTestingTuple(ndx)
        elif modality == 'Rad/' or modality == 'PC/' or modality.startswith('Rad_'):
            return getTestingTupleRadial(ndx)
        elif modality == 'GRE/' or modality == 'Radial_perfusion/':
            return getTestingTupleGRE(ndx)