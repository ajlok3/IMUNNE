import os
import torch
import functools
from torch.utils.data import Dataset, DataLoader, random_split

# enumerate the entries in the directory
path = os.environ["DATA_PATH"]
ksp_sfx = 'kspace/'
smaps_sfx = 'smaps/'

training_list = [f for f in os.listdir(path + ksp_sfx) if os.path.isfile(os.path.join(path + ksp_sfx, f))]

n_training_samples = len(training_list)

@functools.lru_cache(1, typed=True)
def getTrainingTuple(n):
    mode = ""
    cutoff = 80
    return torch.load(
            path + ksp_sfx + training_list[n]
        )[:cutoff], torch.load(
            path + smaps_sfx + training_list[n].replace("kspace", "smaps")
        )[:cutoff],


class TrainingDataset(Dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        return n_training_samples
    
    def __getitem__(self, ndx):
        return getTrainingTuple(ndx)