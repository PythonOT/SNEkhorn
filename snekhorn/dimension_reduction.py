import torch
import numpy as np
from tqdm import tqdm
from snekhorn.affinities import sne_affinity, symmetric_entropic_affinity, log_selfsink
import snekhorn.root_finding as root_finding


OPTIMIZERS = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD}

class Affinity_matcher():
    def __init__(self, kernel_in_Z, optimizer='Adam', verbose=True, tol=1e-4, max_iter=100, lr=1e-1, to_log=False):
        self.optimizer = optimizer
        self.verbose = verbose
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.kernel_in_Z = kernel_in_Z
        if self.to_log:
            self.log = {}

    def match(self, PX, Z):
        losses = []
        Z.requires_grad = True
        f = None
        optimizer = OPTIMIZERS[self.optimizer]([Z], lr=self.lr)

        pbar = tqdm(range(self.max_iter))
        for k in pbar:
            optimizer.zero_grad()
            log_Q = self.kernel_in_Z(Z) #not the best way

            # pytorch reverse the standard definition of the KL div and impose that the input is in log space to avoid overflow
            loss = torch.nn.functional.kl_div(log_Q, PX, reduction='sum')
            if torch.isnan(loss):
                raise Exception(f'NaN in loss at iteration {k}')

            loss.backward()
            optimizer.step()

            if k > 1:
                delta = abs(losses['loss'][-1] - losses['loss'][-2]) / \
                    abs(losses['loss'][-2])
                if delta < self.tol:
                    if self.verbose:
                        print('---------- delta loss convergence ----------')
                    break
                if self.verbose:
                    pbar.set_description(f'Loss : {float(loss.item()): .3e}, '
                                        f'delta : {float(delta): .3e} '
                                        )
        if self.to_log:
            self.log['loss'] = losses
        return Z.detach()

# ---------- Dimension reduction methods related to stochastic neighbor embedding, including ours ----------

def affinity_coupling(P0, Z, kernel=None, eps=1.0,  lr=1, max_iter=1000, optimizer='Adam', loss='KL', verbose=True, tol=1e-4, pz=2, exaggeration=False):

    Z.requires_grad = True
    f = None
    optimizer = OPTIMIZERS[optimizer]([Z], lr=lr)
    counter_cv = 0

    log = {}
    log['loss'] = []
    #log['Z'] = []

    pbar = tqdm(range(max_iter))
    for k in pbar:
        C = torch.cdist(Z, Z, p=pz)**2
        # C.fill_diagonal_(0)
        optimizer.zero_grad()

        if exaggeration and k < 100:
            P = 12*P0
        else:
            P = P0

        if kernel == 'student' or kernel == 'gaussian':
            log_k = log_kernel(C, kernel=kernel)
            log_Q = log_k - torch.logsumexp(log_k, dim=(0, 1))
        else:
            assert eps is not None
            student = (kernel == 'tsnekhorn')
            log_Q, f = log_selfsink(C=C, eps=eps, f=f, student=student)
        # else:
        #     raise ValueError('Kernel not implemented')

        loss = torch.nn.functional.kl_div(log_Q, P, reduction='sum')
        if torch.isnan(loss):
            raise Exception(f'NaN in loss at iteration {k}')

        loss.backward()
        optimizer.step()

        log['loss'].append(loss.item())
        # log['Z'].append(Z.clone().detach().cpu())

        if k > 1:
            delta = abs(log['loss'][-1] - log['loss'][-2]) / \
                abs(log['loss'][-2])
            if delta < tol:
                counter_cv += 1
                if counter_cv > 10:  # Convergence criterion satisfied for more than 10 iterations in a row -> stop the algorithm
                    if verbose:
                        print('---------- delta loss convergence ----------')
                    break
            else:
                counter_cv = 0

            if verbose:
                pbar.set_description(f'Loss : {float(loss.item()): .3e}, '
                                     f'delta : {float(delta): .3e} '
                                     )

    return Z.detach(), log

def log_kernel(C: torch.Tensor, kernel: str):
    if kernel == 'student':
        return - torch.log(1 + C)
    else:  # Gaussian
        return - 0.5 * C


def SNE(X, Z, perp, **coupling_kwargs):
    C = torch.cdist(X, X, p=2)**2
    P = sne_affinity(C, perp)
    return affinity_coupling(P, Z, **coupling_kwargs)


def DSNE(X, Z, eps, **coupling_kwargs):
    C = torch.cdist(X, X, p=2)**2
    T0 = torch.exp(log_selfsink(C, eps=eps)[0])
    return affinity_coupling(T0, Z, **coupling_kwargs)


def SSNE(X, Z, perp, **coupling_kwargs):
    C = torch.cdist(X, X, p=2)**2
    P = symmetric_entropic_affinity(C, perp=perp)
    return affinity_coupling(P, Z, **coupling_kwargs)

