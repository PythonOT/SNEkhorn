# Example of SNEkhorn on COIL dataset

#%%
import matplotlib.pyplot as plt
from snekhorn.utils import COIL_dataset
import torch
#%%
X_coil, Y_coil = COIL_dataset('../data/coil-20-proc')
perp = 30
n = X_coil.shape[0]
X_process = PCA(X_coil, q=50) #reduce dimensionality
C = torch.cdist(X_process, X_process,2)**2