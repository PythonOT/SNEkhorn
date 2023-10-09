#%% tests classes
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from snekhorn import SNEkhorn
from snekhorn.affinities import NormalizedGaussianAndStudentAffinity
from snekhorn.dimension_reduction import SNE
#%%
seed=2
torch.manual_seed(seed)
n=20
X1 = torch.Tensor([-8,-8])[None,:] + torch.normal(0, 1, size=(n, 2), dtype=torch.double)
X2 = torch.Tensor([0,8])[None,:] + torch.normal(0, 3, size=(n, 2), dtype=torch.double)
X3 = torch.Tensor([8,-8])[None,:] + torch.normal(0, 2, size=(n, 2), dtype=torch.double)
X = torch.cat([X1,X2,X3], 0)
#%%
tsnekhorn = SNEkhorn(perp=5, student_kernel=True, tolog=True, square_parametrization=False, lr_sea=1e-3)
tsne = SNE(perp=5, student_kernel=True, tolog=True)
# %%
tsnekhorn.fit(X)
#%%
tsne.fit(X)
# %%
Zsnekhorn = tsnekhorn.embedding_
Zsne = tsne.embedding_
plt.scatter(X[:, 0], X[:, 1], label='data')
plt.scatter(Zsnekhorn[:, 0], Zsnekhorn[:, 1], label='tsnekhorn embedding')
plt.scatter(Zsne[:, 0], Zsne[:, 1], label='tsne embedding')
plt.legend()
# %%
plt.plot(tsnekhorn.log_['log_affinity_in_X']['loss'])
# %%
