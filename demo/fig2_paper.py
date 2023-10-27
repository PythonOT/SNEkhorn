import torch
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import math
from snekhorn.affinities import SymmetricEntropicAffinity, BistochasticAffinity
from snekhorn.utils import entropy

from matplotlib import cm

OPTIMIZERS = {'SGD': torch.optim.SGD,
              'Adam': torch.optim.Adam,
              'NAdam': torch.optim.NAdam
              }

# Sample data
n=30
X1 = torch.Tensor([-8,-8])[None,:] + torch.normal(0, 1, size=(n, 2), dtype=torch.double)
X2 = torch.Tensor([0,8])[None,:] + torch.normal(0, 3, size=(n, 2), dtype=torch.double)
X3 = torch.Tensor([8,-8])[None,:] + torch.normal(0, 2, size=(n, 2), dtype=torch.double)
X = torch.cat([X1,X2,X3], 0)
C = torch.cdist(X,X,2)**2


# Code EOT with fixed global entropy
def log_EOT(C: torch.Tensor,
        mu: torch.Tensor,
        nu: float):
    return (mu[:, None] + mu[None, :] - 2*C) / nu

def EOT_mean_perp(C: torch.Tensor,
        perp: int = 10,
        lr: float = 1e-3,
        nu0: float= 1e-3,
        tol: int = 1e-3,
        max_iter: int = 10000,
        optimizer: bool = 'Adam',
        verbose: bool = True):
    """
        Performs dual ascent to compute EOT with given global entropy n * (log(perp) + 1) .

        Parameters
        ----------
        C: array (n,n) 
            symmetric cost matrix
        perp: int 
            value of the perplexity parameter K
        lr: float
            learning rate used for gradient ascent
        nu0: float
            initial dual variable
        tol: float
            precision threshold at which the algorithm stops
        max_iter: int
            maximum iterations of binary search
        optimizer: bool
            specifies which pytorch optimizer to use
        verbose: bool
            if True, prints current mean and std perplexity values and binary search bounds
    """
    n = C.shape[0]
    mu = torch.zeros(n, dtype=torch.double)
    nu = nu0 * torch.ones(1)
    log_P = log_EOT(C, mu, nu.item())
    target_entropy = n*(math.log(perp) + 1)

    optimizer = OPTIMIZERS[optimizer]([mu,nu], lr=lr)

    if verbose:
        print('---------- Computing the Affinity Matrix ----------')

    one = torch.ones(n, dtype=torch.double)
    pbar = tqdm(range(max_iter))
    for k in pbar:
        with torch.no_grad():
            optimizer.zero_grad()
            P_sum = torch.exp(torch.logsumexp(log_P, -1, keepdim=False))
            mu.grad = P_sum - one
            H = entropy(log_P, log=True, ax=(0,1))
            nu.grad = torch.Tensor([H - target_entropy])
            optimizer.step()
            nu.clamp_(min=0)
            log_P = log_EOT(C, mu, nu.item())

            if verbose:
                pbar.set_description(
                    f'marginal sum : {float(P_sum.mean().item()): .3e}, '
                    f'marginal std : {float(P_sum.std().item()): .3e}, '
                    f'perp est. : {float(math.exp((H/n)-1)): .3e}, ')

            if (torch.abs(P_sum - one) < tol).all() and torch.abs(H-target_entropy)<tol:
                if verbose:
                    print(f'breaking at iter {k}')
                break

            if k == max_iter-1 and verbose:
                print('---------- Max iter attained ----------')

    return log_P


# Graph Laplacian
def Laplacian(P):
    return torch.diag(P.sum(-1)) - P

# Desired Perplexity
perp = 5

# Compute Sinkhorn Affinity
log_Ps = EOT_mean_perp(C, perp, nu0=1e-1, lr=1e-1, max_iter=500)
Ls, _ = torch.linalg.eig(Laplacian(torch.exp(log_Ps)))
cs = torch.exp(entropy(log_Ps, log=True, ax=-1)-1)

# Compute Symmetric Entropic Affinity
sym_aff = SymmetricEntropicAffinity(perp=perp, verbose=True, lr=1e-2, square_parametrization=False, max_iter=10000)
log_PSE = sym_aff.compute_log_affinity(X)
Lse, _ = torch.linalg.eig(Laplacian(torch.exp(log_PSE)))
cSE = torch.exp(entropy(log_PSE, log=True, ax=-1)-1)

norm = cm.colors.Normalize(vmax=cs.max(), vmin=cs.min())
cmap = cm.RdBu

params = {'text.usetex': True}
        #   'text.latex.preamble': [r'\usepackage{cmbright}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)
plt.rc('font', family='Times New Roman')

markers = n*['o'] + n*['s'] + n*['^']

def plot_log_graph(C, x, c, ax=None, s=10, markers=markers, scale=1):
    C = np.array(C)
    if ax is None:
        ax = plt
    for j in range(C.shape[0]):
        for i in range(j):
            ax.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], alpha=np.min([scale*np.exp(C[i, j]),1],0), color='k', linewidth=2)
        im = ax.scatter(x[j, 0], x[j, 1], c=c[j], s=s, zorder=10, edgecolors='k', norm=norm, marker=markers[j], cmap=cmap)
    return im

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4), width_ratios= [2, 2, 1.5])

im0 = plot_log_graph(log_Ps, X, cs, s=40, ax=axes[0], scale=4, markers=markers)
clb0 = fig.colorbar(im0, ax=axes[0])
clb0.set_label('Perplexity', labelpad=-30, y=1.07, rotation=0, fontsize=15)
cbar0_yticks = plt.getp(clb0.ax.axes, 'yticklabels')
clb0.ax.tick_params(labelsize=15) 
plt.setp(cbar0_yticks[1], color='r')

axes[0].set_title(r'$\mathbf{P}^{\mathrm{ds}}$', fontsize=20)
im2 = plot_log_graph(log_PSE, X, cSE, s=40, ax=axes[1], scale=4,  markers=markers)
clb2 = fig.colorbar(im2, ax=axes[1])
clb2.set_label('Perplexity', labelpad=-30, y=1.07, rotation=0, fontsize=15)
cbar2_yticks = plt.getp(clb2.ax.axes, 'yticklabels')
clb2.ax.tick_params(labelsize=15) 
plt.setp(cbar2_yticks[1], color='r')

axes[1].set_title(r'$\mathbf{P}^{\mathrm{se}}$', fontsize=20)
axes[2].set_title('Laplacian Eigenvalues', fontsize=20)
axes[2].scatter(np.arange(1,16), torch.sort(torch.real(Ls), descending=True).values[-15:], label=r'$\mathbf{P}^{\mathrm{ds}}$', marker='o', s=50, alpha=0.5)
axes[2].scatter(np.arange(1,16), torch.sort(torch.real(Lse), descending=True).values[-15:], label=r'$\mathbf{P}^{\mathrm{se}}$', marker='d', s=50, alpha=0.5)
axes[2].axhline(y=0, color='black', linestyle='--')
axes[2].legend(fontsize=20)

fig.tight_layout()
plt.savefig('Ps_vs_Pse.pdf', bbox_inches='tight')
plt.show()