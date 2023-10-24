#%%
import torch 
import numpy as np
import matplotlib.pyplot as plt

from snekhorn.dimension_reduction import SNE, SNEkhorn
from snekhorn.affinities import EntropicAffinity
from matplotlib.colors import LogNorm, PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker as mticker
from snekhorn.utils import entropy

# Simulate Data
torch.manual_seed(16)

n = 1000
m = 10000

z1 = np.random.uniform(size=m)
p1 = z1/z1.sum()
z2 = np.random.uniform(size=m)
p2 = z2/z2.sum()

X1 = np.random.multinomial(1000, pvals=p1, size=int(n/2))
X1 = X1/X1.sum(-1, keepdims=True)
X2 = np.random.multinomial(1000, pvals=p2, size=int(n/4))
X2 = X2/(X2.sum(-1, keepdims=True))
X3 = np.random.multinomial(2000, pvals=p2, size=int(n/4))
X3 = X3/(X3.sum(-1, keepdims=True))

X = torch.from_numpy(np.concatenate((X1,X2,X3),0))
X = (X-X.mean())/X.std()
Y = [0]*int(n/2) + [1]*int(n/2)
Z = torch.normal(0, 1, size=(n, 2), dtype=torch.double)

#%%
perp=30
sne = SNE(perp=perp, lr=1e-1, max_iter=500)
sne.fit(X)
P_SNE = sne.PX_
print('Perplexities of the rows are {}'.format(torch.exp(entropy(P_SNE)-1)))

#%%
snekhorn = SNEkhorn(perp=perp, lr=1e-1, max_iter=500, lr_sea=1e1, max_iter_sea=3000)
snekhorn.fit(X)
P_SE = snekhorn.PX_
print('Perplexities of the rows are {}'.format(torch.exp(entropy(P_SNE)-1)))

#%%
Z_sne = sne.embedding_
Z_snekhorn = snekhorn.embedding_

#%%
params = {'text.usetex': True}
plt.rcParams.update(params)

vmin = 1e-6
vmax = 1e-3
plt.rc('font', family='Times New Roman')
fs = 24
imshow_kwargs = {'cmap':'Blues', 'norm':PowerNorm(1.5, vmin=vmin, vmax=vmax)}

fig, axs = plt.subplots(1, 4, figsize=(19,4.5), constrained_layout=True) #gridspec_kw = {'height_ratios':[3,1]})

im0 = axs[0].imshow(P_SNE, aspect="auto", **imshow_kwargs)
axs[0].set_title(r'$\overline{\mathbf{P}^{\mathrm{e}}}$', fontsize=fs)
axs[0].set_xticks([])
axs[0].set_yticks([])

divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="3%", pad=0.05)
cb0 = fig.colorbar(im0, cax=cax, orientation='vertical')
ticks_loc = cb0.ax.get_yticks().tolist()
cb0.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
cb0.ax.set_yticklabels([f'{i:.0e}' for i in cb0.get_ticks()])
cb0.ax.tick_params(labelsize=fs-5)
# cb0.set_ticks([])

axs[2].set_title(r'$\mathbf{P}^{\mathrm{se}}$', fontsize=fs)
axs[2].set_xticks([])
axs[2].set_yticks([])
# divider = make_axes_locatable(axs[0,1])
# cax = divider.append_axes("right", size="3%", pad=0.03)
im2 = axs[2].imshow(P_SE, aspect="auto", **imshow_kwargs)

divider = make_axes_locatable(axs[2])
cax = divider.append_axes("right", size="3%", pad=0.05)
cb2 = fig.colorbar(im2, cax=cax, orientation='vertical')
ticks_loc = cb2.ax.get_yticks().tolist()
cb2.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
cb2.ax.set_yticklabels([f'{i:.0e}' for i in cb2.get_ticks()])
cb2.ax.tick_params(labelsize=fs-5)
# cb2.set_ticks([])

# clb2 = fig.colorbar(im2, ax=axs[2])

axs[1].scatter(Z_sne[np.array(Y) == 0,0], Z_sne[np.array(Y) == 0,1], alpha=0.7, c='blue', edgecolor='k', label='$\mathbf{p}_1$')
axs[1].scatter(Z_sne[np.array(Y) == 1,0], Z_sne[np.array(Y) == 1,1], alpha=0.7, c='red', edgecolor='k', label='$\mathbf{p}_2$')
axs[1].set_title('Symmetric-SNE', fontsize=fs)
axs[1].set_xticks([])
axs[1].set_yticks([])

axs[3].scatter(Z_snekhorn[np.array(Y) == 0,0], Z_snekhorn[np.array(Y) == 0,1], alpha=0.7, c='blue', edgecolor='k', label='$\mathbf{p}_1$')
axs[3].scatter(Z_snekhorn[np.array(Y) == 1,0], Z_snekhorn[np.array(Y) == 1,1], alpha=0.7, c='red', edgecolor='k', label='$\mathbf{p}_2$')
axs[3].set_title('SNEkhorn', fontsize=fs)
axs[3].legend(loc = 'lower right', fontsize=fs-3)
axs[3].set_xticks([])
axs[3].set_yticks([])

# fig.tight_layout(pad=1.0)
#plt.savefig('heteroscedastic_noise.pdf', bbox_inches='tight')  
# %%
