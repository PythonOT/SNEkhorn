import torch
import numpy as np
from tqdm import tqdm
from snekhorn.affinities import sne_affinity, symmetric_entropic_affinity, log_selfsink
import snekhorn.root_finding as root_finding


OPTIMIZERS = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD}

class Affinity_matcher():
    def __init__(self, 
                 affinity_in_Z, #both are two classes that computes the affinities
                 affinity_in_X, 
                 output_dim=2, 
                 optimizer='Adam', 
                 verbose=True, 
                 tol=1e-4, 
                 max_iter=100, 
                 lr=1e-1, 
                 to_log=False):
        
        assert optimizer in ['Adam', 'SGD']
        self.optimizer = optimizer
        self.verbose = verbose
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.affinity_in_Z = affinity_in_Z #should be in log space
        self.output_dim = output_dim
        self.affinity_in_X = affinity_in_X
        if to_log:
            self.log = {}

    def fit(self, X):
        n = X.shape[0]
        P_X = self.affinity_in_X.compute_affinity(X)
        losses = []
        Z = torch.normal(0, 1, size=(n, self.output_dim), dtype=torch.double)
        Z.requires_grad = True
        f = None
        optimizer = OPTIMIZERS[self.optimizer]([Z], lr=self.lr)

        pbar = tqdm(range(self.max_iter))
        for k in pbar:
            optimizer.zero_grad()
            log_Q = self.affinity_in_Z.compute_affinity(Z) #not the best way

            # pytorch reverse the standard definition of the KL div and impose that the input is in log space to avoid overflow
            loss = torch.nn.functional.kl_div(log_Q, P_X, reduction='sum')
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
    
class SNEkhorn():
    def __init__(self,
                 perp, 
                 output_dim=2, 
                 optimizer='Adam', 
                 verbose=True, 
                 tol=1e-4, 
                 max_iter=100, 
                 lr=1e-1, 
                 to_log=False,
                 **args_sym_entropic):
        self.perp = perp
        self.output_dim = output_dim
        self.optimizer = optimizer
        self.verbose = verbose
        self.tol = tol
        self.max_iter = max_iter
        self.lr = lr
        self.to_log = to_log

        def kernel_in_X(X):
            C = torch.cdist(X, X, p=2)**2
            P = symmetric_entropic_affinity(C, perp=perp, **args_sym_entropic)
            return
        
        def kernel_in_Z(Z):
            C = torch.cdist(Z, Z, p=2)**2
            log_Q, f = log_selfsink(C=C, eps=eps, f=f, student=student)
        affinity_matcher = Affinity_matcher()


def SSNE(X, Z, perp, **coupling_kwargs):
    C = torch.cdist(X, X, p=2)**2
    P = symmetric_entropic_affinity(C, perp=perp)
    return affinity_coupling(P, Z, **coupling_kwargs)

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

