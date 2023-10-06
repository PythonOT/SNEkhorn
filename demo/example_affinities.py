#%%
import torch
import matplotlib.pyplot as plt
import ot
from matplotlib import cm
import numpy as np
from snekhorn.affinities import sne_affinity, symmetric_entropic_affinity
#%%
seed=2
torch.manual_seed(seed)
n=20
X1 = torch.Tensor([-8,-8])[None,:] + torch.normal(0, 1, size=(n, 2), dtype=torch.double)
X2 = torch.Tensor([0,8])[None,:] + torch.normal(0, 3, size=(n, 2), dtype=torch.double)
X3 = torch.Tensor([8,-8])[None,:] + torch.normal(0, 2, size=(n, 2), dtype=torch.double)
X = torch.cat([X1,X2,X3], 0)
C = torch.cdist(X,X,2)**2
n = 3*n
nu = 2
K = torch.exp(-C/nu)
#%%
plt.close('all')
plt.figure(1, (10,3))

plt.subplot(1,3,1)
plt.scatter(X[:,0], X[:,1],alpha=0.5)
plt.title("Data")

plt.subplot(1,3,2)
plt.imshow(K.detach().numpy(), cmap=cm.Greys,interpolation='none')
plt.title(r"Gaussian Kernel ($\nu$=2)")

plt.subplot(1,3,3)

Kvisu = K.detach().numpy()
for i in range(n):
    for j in range(i):
        if K[i,j]>1e-2:
            plt.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], color='black', alpha=Kvisu[i,j].item())
plt.scatter(X[:,0], X[:,1],alpha=0.5)
plt.title("Affinity graph")
plt.show()
# %%
nu = 2
Pds = ot.sinkhorn(torch.ones(n), torch.ones(n), C, nu)

#%%

plt.close('all')
plt.figure(1, (10,3))

Kvisu = Pds.detach().numpy()
vmax=1
scale=5

plt.subplot(1,3,1)
plt.scatter(X[:,0], X[:,1],alpha=0.5)
plt.title("Data")

plt.subplot(1,3,2)
plt.imshow(Kvisu, cmap=cm.Greys,interpolation='none',vmax=vmax)
plt.title(r"Doubly Stochastic Aff. ($\nu$=2)")

plt.subplot(1,3,3)


for i in range(n):
    for j in range(i):
        if K[i,j]>1e-6:
            plt.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], color='black', alpha=min(Kvisu[i,j]*scale,1))
plt.scatter(X[:,0], X[:,1],alpha=0.5)
plt.title("Affinity graph")
plt.show()
# %%
perp=5

Pe = sne_affinity(C,perp)

#%%

plt.close('all')
plt.figure(1, (10,3))

Kvisu = Pe.detach().numpy()
Kvisu = Kvisu/np.max(Kvisu)
vmax=0.5
scale=1

plt.subplot(1,3,1)
plt.scatter(X[:,0], X[:,1],alpha=0.5)
plt.title("Data")

plt.subplot(1,3,2)
plt.imshow(Kvisu, cmap=cm.Greys,interpolation='none',vmax=vmax)
plt.title(r"Entropic Aff. ($\xi$=5)")

plt.subplot(1,3,3)


for i in range(n):
    for j in range(n):
        if K[i,j]*scale>1e-5:
            plt.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], color='black', alpha=min(Kvisu[i,j]*scale,1))
plt.scatter(X[:,0], X[:,1],alpha=0.5)
plt.title("Affinity graph")
plt.show()

#%%

perp=5

Pse = symmetric_entropic_affinity(C, perp, lr=1e-1)

plt.close('all')
plt.figure(1, (10,3))

Kvisu = Pse.detach().numpy()
Kvisu = Kvisu/np.max(Kvisu)
vmax=0.5
scale=1

plt.subplot(1,3,1)
plt.scatter(X[:,0], X[:,1],alpha=0.5)
plt.title("Data")

plt.subplot(1,3,2)
plt.imshow(Kvisu, cmap=cm.Greys,interpolation='none',vmax=vmax)
plt.title(r"Sym. Entropic Aff. ($\xi$=5)")

plt.subplot(1,3,3)


for i in range(n):
    for j in range(n):
        if K[i,j]*scale>1e-5:
            plt.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], color='black', alpha=min(Kvisu[i,j]*scale,1))
plt.scatter(X[:,0], X[:,1],alpha=0.5)
plt.title("Affinity graph")
plt.show()
# %%