# %% Very simple script to take in charge the framework
from snekhorn.affinities import BistochasticAffinity, SymmetricEntropicAffinity
from snekhorn.dimension_reduction import AffinityMatcher
import torch
import matplotlib.pyplot as plt
import numpy as np
from snekhorn import SNEkhorn
from snekhorn.dimension_reduction import SNE
# %% Make simple 2D data
seed = 2
torch.manual_seed(seed)
n = 20
X1 = torch.Tensor([-8, -8])[None, :] + torch.normal(0,
                                                    1, size=(n, 2), dtype=torch.double)
X2 = torch.Tensor([0, 8])[None, :] + torch.normal(0,
                                                  3, size=(n, 2), dtype=torch.double)
X3 = torch.Tensor([8, -8])[None, :] + torch.normal(0,
                                                   2, size=(n, 2), dtype=torch.double)
X = torch.cat([X1, X2, X3], 0)
# %%
tsnekhorn = SNEkhorn(perp=5,
                     student_kernel=True,
                     lr_sea=1e-1,
                     max_iter_sea=500,
                     square_parametrization=True,
                     tolog=True)
# %%
tsnekhorn.fit(X)
# %%
Zsnekhorn = tsnekhorn.embedding_
plt.scatter(X[:, 0], X[:, 1], label='data')
plt.scatter(Zsnekhorn[:, 0], Zsnekhorn[:, 1], label='tsnekhorn embedding')
plt.legend()
# %% We can also use the precomputed method this way

symmetric_entropic_affinity = SymmetricEntropicAffinity(
    perp=5, lr=1e-1, max_iter=500, square_parametrization=True)
PX = symmetric_entropic_affinity.compute_affinity(X)
# %% Match the fixed affinity PX with a bistochastic kernel
bisto_affinity = BistochasticAffinity()
snekhorn_precomputed = AffinityMatcher(affinity_in_Z=bisto_affinity,
                                       affinity_in_X="precomputed",
                                       lr=1e-1)
# the fit method when precomputed should be applied on a affinity matrix
snekhorn_precomputed.fit(PX)
#%% 
Zsnekhorn = snekhorn_precomputed.embedding_
plt.scatter(X[:, 0], X[:, 1], label='data')
plt.scatter(Zsnekhorn[:, 0], Zsnekhorn[:, 1], label='tsnekhorn embedding')
plt.legend()
# %%
