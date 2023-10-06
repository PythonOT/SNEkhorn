# Affinity matrices
import torch
import numpy as np
import snekhorn.root_finding as root_finding
from tqdm import tqdm
import time
from snekhorn.utils import entropy

OPTIMIZERS = {'SGD': torch.optim.SGD,
              'Adam': torch.optim.Adam, 
              'NAdam': torch.optim.NAdam,
              'LBFGS': torch.optim.LBFGS}


def log_Pe(C: torch.Tensor,
           eps: torch.Tensor):
    """
        Returns the log of the directed affinity matrix of SNE.

        Parameters
        ----------
        C: array (n,n) 
            distance matrix
        eps: array (n)
            kernel bandwidths vector
    """
    log_P = - C / (eps[:, None])
    return log_P - torch.logsumexp(log_P, -1, keepdim=True)



class Entropic_affinity():
    def __init__(self, perp, tol=1e-5, max_iter=1000, verbose=True, begin=None, end=None, output_log=False, normalize_as_sne=True):
        self.perp = perp
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.begin = begin
        self.end = end
        self.output_log = output_log
        self.normalize_as_sne=normalize_as_sne

    def compute_affinity(self, X):
        log_P = self.compute_log_affinity(X)
        return torch.exp(log_P)
    
    def compute_log_affinity(self, X):
        C = torch.cdist(X, X, 2)**2
        log_P = self.entropic_affinity(C)
        if self.normalize_as_sne:# does P+P.T/2 in log space
            log_P_SNE = torch.logsumexp(torch.stack([log_P,log_P.T], 0), 0, keepdim=False) - np.log(2)
            return log_P_SNE
        else:
            return log_P

    def entropic_affinity(self, C):
        """
            Performs a binary search to solve the dual problem of entropic affinities in log space.
            Returns the entropic affinity matrix that is **not** symmetrized.
            Parameters
            ----------
            C: array (n, n) 
                distance matrix
            perp: int 
                value of the perplexity parameter
            tol: float
                precision threshold at which the algorithm stops
            max_iter: int
                maximum iterations for the binary search
            verbose: bool
                if True, prints current mean and std entropy values and current bounds 
            output_log: bool
                if True, outputs the affinity in log domain 
        """
        target_entropy = np.log(self.perp) + 1
        n = C.shape[0]
            
        def f(eps):
            return entropy(log_Pe(C, eps), log=True) - target_entropy

        eps_star, _, _  = root_finding.false_position(f=f, n=n, begin=self.begin, end=self.end, tol=self.tol, max_iter=self.max_iter, verbose=self.verbose)
        log_affinity = log_Pe(C, eps_star)

        return log_affinity
    

class Symmetric_entropic_affinity():
    def __init__(self, perp, lr=1e-3, tol=1e-3, max_iter=10000, optimizer='Adam', verbose=True, tolog=False, squared_parametrization=True):
        self.perp = perp
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.verbose = verbose
        if tolog:
            self.log = {}
        self.squared_parametrization = squared_parametrization

    def compute_affinity(self, X):
        log_P = self.compute_log_affinity(X)
        return torch.exp(log_P)
    
    def compute_log_affinity(self, X):
        C = torch.cdist(X, X, 2)**2
        log_P = self.symmetric_entropic_affinity(C)
        return log_P

    def symmetric_entropic_affinity(self, C):
        n = C.shape[0]
        assert 1 <= self.perp <= n
        target_entropy = np.log(self.perp) + 1
        eps = torch.ones(n, dtype=torch.double)
        mu = torch.zeros(n, dtype=torch.double)
        log_P = log_Pse(C, eps, mu, to_square=self.squared_parametrization)

        optimizer = OPTIMIZERS[optimizer]([eps, mu], lr=self.lr)

        if self.tolog:
            self.log['eps'] = [eps.clone().detach()]
            self.log['mu'] = [mu.clone().detach()]
            self.log['loss'] = []

        if self.verbose:
            print('---------- Computing the Affinity Matrix ----------')

        one = torch.ones(n, dtype=torch.double)
        pbar = tqdm(range(self.max_iter))
        for k in pbar:
            with torch.no_grad():
                optimizer.zero_grad()
                H = entropy(log_P, log=True)

                if self.squared_parametrization:
                    eps.grad = 2*eps.clone().detach()*(H - target_entropy) 
                else:
                    eps.grad = H - target_entropy

                P_sum = torch.exp(torch.logsumexp(log_P, -1, keepdim=False))
                mu.grad = P_sum - one
                optimizer.step()
                if not self.squared_parametrization:
                    eps.clamp_(min=0)
                
                log_P = log_Pse(C, eps, mu, to_square=self.squared_parametrization)

                if torch.isnan(eps).any() or torch.isnan(mu).any():
                    raise Exception(f'NaN in dual variables at iteration {k}')

                if self.tolog:
                    self.log['eps'].append(eps.clone().detach())
                    self.log['mu'].append(mu.clone().detach())
                    eps_ = eps.clone().detach()
                    if self.squared_parametrization:
                        eps_ = eps_**2
                        mu_ = mu.clone().detach()
                        self.log['loss'].append(-Lagrangian(C, torch.exp(log_P.clone().detach()),
                                                eps_, mu_, self.perp, squared_parametrization=self.squared_parametrization).item())

                perps = torch.exp(H-1)
                if self.verbose:
                    pbar.set_description(
                        f'perps mean : {float(perps.mean().item()): .3e}, '
                        f'perps std : {float(perps.std().item()): .3e}, '
                        f'marginal sum : {float(P_sum.mean().item()): .3e}, '
                        f'marginal std : {float(P_sum.std().item()): .3e}, ')

                if (torch.abs(H - np.log(self.perp)-1) < self.tol).all() and (torch.abs(P_sum - one) < self.tol).all():
                    if self.verbose:
                        print(f'breaking at iter {k}')
                    break

                if k == self.max_iter-1 and self.verbose:
                    print('---------- Max iter attained ----------')

        return log_P

# ----- Symmetric Entropic Affinity -----



def log_Pse(C: torch.Tensor,
            eps: torch.Tensor,
            mu: torch.Tensor,
            to_square: bool=False):
    """
        Returns the log of the symmetric entropic affinity matrix with specified parameters epsilon and mu.

        Parameters
        ----------
        C: array (n,n) 
            distance matrix
        eps: array (n)
            symmetric entropic affinity dual variables associated to the entropy constraint
        mu: array (n)
            symmetric entropic affinity dual variables associated to the marginal constraint
        to_square: TBD
    """
    if to_square:
        return (mu[:, None] + mu[None, :] - 2*C)/(eps[:, None]**2 + eps[None, :]**2)
    else:
        return (mu[:, None] + mu[None, :] - 2*C)/(eps[:, None] + eps[None, :])

# Alternative method to compute symmetric entropic affinities.

def Lagrangian(C, log_P, eps, mu, perp=30):
    # TBD
    one = torch.ones(C.shape[0], dtype=torch.double)
    target_entropy = np.log(perp) + 1
    HP = entropy(log_P, log=True, ax=1)
    return torch.exp(torch.logsumexp(log_P + torch.log(C), (0,1), keepdim=False)) + torch.inner(eps, (target_entropy - HP)) + torch.inner(mu, (one - torch.exp(torch.logsumexp(log_P, -1, keepdim=False))))

def log_selfsink(C: torch.Tensor, 
                eps: float=1.,
                f: torch.Tensor=None,
                tol: float=1e-5,
                max_iter: int=1000,
                student: bool=False,
                tolog: bool=False):
    """ 
        Performs Sinkhorn iterations in log domain to solve the entropic "self" (or "symmetric") OT problem with symmetric cost C and entropic regularization epsilon.
        Returns the transport plan and dual variable at convergence.

        Parameters
        ----------
        C: array (n,n)
            symmetric distance matrix
        eps: float
            entropic regularization coefficient
        f: array(n)
            initial dual variable
        tol: float
            precision threshold at which the algorithm stops
        max_iter: int
            maximum number of Sinkhorn iterations
        student: bool
            if True, a Student-t kernel is considered instead of Gaussian
        tolog: bool
            if True, log and returns intermediate variables
    """
    n = C.shape[0]

    # Allows a warm-start if a dual variable f is provided
    f = torch.zeros(n) if f is None else f.clone().detach()

    if tolog:
        log = {}
        log['f'] = [f.clone().detach()]

    # If student is True, considers the Student-t kernel instead of Gaussian
    if student:
        C = torch.log(1+C)

    # Sinkhorn iterations
    for k in range(max_iter+1):
        f = 0.5 * (f - eps*torch.logsumexp((f - C) / eps,-1))

        if tolog:
            log['f'].append(f.clone().detach())        

        if torch.isnan(f).any():
            raise Exception(f'NaN in self-Sinkhorn dual variable at iteration {k}')

        log_T = (f[:,None] + f[None,:] - C) / eps
        if (torch.abs(torch.exp(torch.logsumexp(log_T, -1))-1) < tol).all():
            break

        if k == max_iter-1:
            print('---------- Max iter attained ----------')

    if tolog:
        return (f[:,None] + f[None,:] - C) / eps, f, log
    else:
        return (f[:,None] + f[None,:] - C) / eps, f