#%% tests classes
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from snekhorn import SNEkhorn
#%%
seed=2
torch.manual_seed(seed)
n=20
X1 = torch.Tensor([-8,-8])[None,:] + torch.normal(0, 1, size=(n, 2), dtype=torch.double)
X2 = torch.Tensor([0,8])[None,:] + torch.normal(0, 3, size=(n, 2), dtype=torch.double)
X3 = torch.Tensor([8,-8])[None,:] + torch.normal(0, 2, size=(n, 2), dtype=torch.double)
X = torch.cat([X1,X2,X3], 0)
#%%
tsnekhorn = SNEkhorn(perp=5)
# %%
tsnekhorn.fit(X)
# %%
Z = tsnekhorn.embedding
plt.scatter(X[:, 0], X[:, 1], label='data')
plt.scatter(Z[:, 0], Z[:, 1], label='embedding')
plt.legend()
# %%
