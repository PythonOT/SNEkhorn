# Affinity matrices
import torch
import math
import snekhorn.root_finding as root_finding
from tqdm import tqdm
from snekhorn.utils import entropy

OPTIMIZERS = {'SGD': torch.optim.SGD,
              'Adam': torch.optim.Adam,
              'NAdam': torch.optim.NAdam
              }


class NanError(Exception):
    pass


class BadPerplexity(Exception):
    pass


class BaseAffinity():
    def __init__(self):
        self.log_ = {} #BaseAffinity contains a dictionary of different results 

    def compute_affinity(self, X):
        """Computes an affinity matrix from an affinity matrix in log space.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        P: torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix.
        """
        log_P = self.compute_log_affinity(X)
        return torch.exp(log_P)


class NormalizedGaussianAndStudentAffinity(BaseAffinity):
    """This class computes the normalized affinity associated to a Gaussian or t-Student kernel. The affinity matrix is normalized by its total sum.

    Parameters
    ----------
    student : bool, optional
        If True computes a t-Student kernel, by default False.
    sigma : float, optional
        The length scale of the Gaussian kernel, by default 1.0.
    p : int, optional
        p value for the p-norm distance to calculate between each vector pair, by default 2.
    """

    def __init__(self, student=False, sigma=1.0, p=2):
        self.student = student
        self.sigma = sigma
        self.p = p

    def compute_log_affinity(self, X):
        """Computes the pairwise affinity matrix in log space and normalize it by its total sum.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        log_P: torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix in log space.
        """
        C = torch.cdist(X, X, self.p)**2
        if self.student:
            log_P = - torch.log(1 + C)
        else:
            log_P = - C / (2*self.sigma)
        return log_P - torch.logsumexp(log_P, dim=(0, 1))


class EntropicAffinity(BaseAffinity):
    """This class computes the entropic affinity used in SNE and tSNE in log domain. It corresponds also to the Pe matrix in [1] in log domain (see also [2]). 
    When normalize_as_sne = True, the affinity is symmetrized as (Pe + Pe.T) /2.

    Parameters
    ----------
    perp : int
        Perplexity parameter, related to the number of nearest neighbors that is used in other manifold learning algorithms. 
        Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and the number of samples. 
        Different values can result in significantly different results. The perplexity must be less than the number of samples.
    tol : _type_, optional
        Precision threshold at which the root finding algorithm stops, by default 1e-5.
    max_iter : int, optional
        Number of maximum iterations for the root finding algorithm, by default 1000.
    verbose : bool, optional
        Verbosity, by default True.
    begin : _type_, optional
        Initial lower bound of the root, by default None.
    end : _type_, optional
        Initial upper bound of the root, by default None.
    normalize_as_sne : bool, optional
        If True the entropic affinity is symmetrized as (Pe + Pe.T) /2, by default True.

    References
    ----------
    [1] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities, Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
    [2] Entropic Affinities: Properties and Efficient Numerical Computation, Max Vladymyrov, Miguel A. Carreira-Perpinan, ICML 2013.
    """

    def __init__(self,
                 perp,
                 tol=1e-5,
                 max_iter=1000,
                 verbose=True,
                 begin=None,
                 end=None,
                 normalize_as_sne=True):

        self.perp = perp
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.begin = begin
        self.end = end
        self.normalize_as_sne = normalize_as_sne

    def compute_log_affinity(self, X):
        """Computes the pairwise entropic affinity matrix in log space. If normalize_as_sne is True returns the symmetrized version.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed

        Returns
        -------
        log_P: torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix in log space. If normalize_as_sne is True returns the symmetrized affinty in log space.
        """
        C = torch.cdist(X, X, 2)**2
        log_P = self.entropic_affinity(C)
        if self.normalize_as_sne:  # does P+P.T/2 in log space
            log_P_SNE = torch.logsumexp(torch.stack(
                [log_P, log_P.T], 0), 0, keepdim=False) - math.log(2)
            return log_P_SNE
        else:
            return log_P

    def entropic_affinity(self, C):
        """Performs a binary search to solve the dual problem of entropic affinities in log space.
        It solves the problem (EA) in [1] and returns the entropic affinity matrix in log space (which is **not** symmetric).

        Parameters
        ----------
        C: torch.Tensor of shape (n_samples, n_samples)
            Distance matrix between the samples.

        References
        ----------
        [1] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities, Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
        """
        target_entropy = math.log(self.perp) + 1
        n = C.shape[0]

        if not 1 <= self.perp <= n:
            BadPerplexity(
                'The perplexity parameter must be between 1 and number of samples')

        def f(eps):
            return entropy(log_Pe(C, eps), log=True) - target_entropy

        eps_star, _, _ = root_finding.false_position(
            f=f, n=n, begin=self.begin, end=self.end, tol=self.tol, max_iter=self.max_iter, verbose=self.verbose)
        log_affinity = log_Pe(C, eps_star)

        return log_affinity


class SymmetricEntropicAffinity(BaseAffinity):
    def __init__(self,
                 perp,
                 lr=1e-3,
                 squared_parametrization=True,
                 tol=1e-3,
                 max_iter=10000,
                 optimizer='Adam',
                 verbose=True,
                 tolog=False):
        """This class computes the solution to the symmetric entropic affinity problem described in [1], in log space. 
        More precisely, it solves equation (SEA) in [1] with the dual ascent procedure described in the paper and returns the log of the affinity matrix.

        Parameters
        ----------
        perp : int
            Perplexity parameter, related to the number of nearest neighbors that is used in other manifold learning algorithms. 
            Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and the number of samples. 
            Different values can result in significantly different results. The perplexity must be less than the number of samples.
        lr : float, optional
            Learning rate for the algorithm, usually in the range [1e-5, 10], by default 1e-3.
        squared_parametrization : bool, optional
            Whether to optimize on the square of the dual variables. 
            If True the algorithm is not convex anymore but is more stable in practice, by default True.
        tol : float, optional
            Precision threshold at which the algorithm stops, by default 1e-5.
        max_iter : int, optional
            Number of maximum iterations for the algorithm, by default 1000.
        optimizer : str, optional
            Which pytorch optimizer to use among ['SGD', 'Adam', 'NAdam'], by default 'Adam'.
        verbose : bool, optional
            Verbosity, by default True.
        tolog : bool, optional
            Whether to store intermediate result in a dictionary, by default False.

        Attributes
        ----------
        log_ : dictionary
            Contains the loss and the dual variables at each iteration of the optimization algorithm when tolog = True.
        n_iter_: int
            Number of iterations run.

        References
        ----------
        [1] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities, Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
        """
        self.perp = perp
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.verbose = verbose
        self.tolog = tolog
        self.n_iter_ = 0
        self.squared_parametrization = squared_parametrization

    def compute_log_affinity(self, X):
        """Computes the pairwise symmetric entropic affinity matrix in log space.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        log_P: torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix in log space. 
        """
        C = torch.cdist(X, X, 2)**2
        log_P = self.symmetric_entropic_affinity(C)
        return log_P

    def symmetric_entropic_affinity(self, C):
        """Solves the dual optimization problem (Dual-SEA) in [1] and returns the corresponding symmetric entropic affinty in log space.

        Parameters
        ----------
        C : torch.Tensor of shape (n_samples, n_samples)
            Distance matrix between samples.

        Returns
        -------
        log_P: torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix in log space. 

        References
        ----------
        [1] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities, Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
        """
        n = C.shape[0]
        if not 1 <= self.perp <= n:
            BadPerplexity(
                'The perplexity parameter must be between 1 and number of samples')
        target_entropy = math.log(self.perp) + 1
        # dual variable corresponding to the entropy constraint
        eps = torch.ones(n, dtype=torch.double)
        # dual variable corresponding to the marginal constraint
        mu = torch.zeros(n, dtype=torch.double)
        log_P = log_Pse(C, eps, mu, to_square=self.squared_parametrization)

        optimizer = OPTIMIZERS[self.optimizer]([eps, mu], lr=self.lr)

        if self.tolog:
            self.log_['eps'] = [eps.clone().detach()]
            self.log_['mu'] = [mu.clone().detach()]
            self.log_['loss'] = []

        if self.verbose:
            print(
                '---------- Computing the symmetric entropic affinity matrix ----------')

        one = torch.ones(n, dtype=torch.double)
        pbar = tqdm(range(self.max_iter))
        for k in pbar:
            with torch.no_grad():
                optimizer.zero_grad()
                H = entropy(log_P, log=True)

                if self.squared_parametrization:
                    # the Jacobian must be corrected by 2* diag(eps) in the case of square parametrization.
                    eps.grad = 2*eps.clone().detach()*(H - target_entropy)
                else:
                    eps.grad = H - target_entropy

                P_sum = torch.exp(torch.logsumexp(log_P, -1, keepdim=False))
                mu.grad = P_sum - one
                optimizer.step()
                if not self.squared_parametrization:  # optimize on eps > 0
                    eps.clamp_(min=0)

                log_P = log_Pse(
                    C, eps, mu, to_square=self.squared_parametrization)

                if torch.isnan(eps).any() or torch.isnan(mu).any():
                    raise NanError(
                        f'NaN in dual variables at iteration {k}, consider decreasing the learning rate of SymmetricEntropicAffinity')

                if self.tolog:
                    self.log_['eps'].append(eps.clone().detach())
                    self.log_['mu'].append(mu.clone().detach())
                    eps_ = eps.clone().detach()
                    if self.squared_parametrization:
                        eps_ = eps_**2
                        mu_ = mu.clone().detach()
                        self.log_['loss'].append(-Lagrangian(C, torch.exp(log_P.clone().detach()),
                                                            eps_, mu_, self.perp, squared_parametrization=self.squared_parametrization).item())

                perps = torch.exp(H-1)
                if self.verbose:
                    pbar.set_description(
                        f'perps mean : {float(perps.mean().item()): .3e}, '
                        f'perps std : {float(perps.std().item()): .3e}, '
                        f'marginal sum : {float(P_sum.mean().item()): .3e}, '
                        f'marginal std : {float(P_sum.std().item()): .3e}, ')

                if (torch.abs(H - math.log(self.perp)-1) < self.tol).all() and (torch.abs(P_sum - one) < self.tol).all():
                    self.log_['n_iter'] = k
                    self.n_iter_ = k
                    if self.verbose:
                        print(f'breaking at iter {k}')
                    break

                if k == self.max_iter-1 and self.verbose:
                    print('---------- Max iter attained ----------')

        return log_P


class BistochasticAffinity(BaseAffinity):
    """This class computes the symmetric doubly stochastic affinity matrix in log domain with Sinkhorn algorithm.
    It normalizes a Gaussian RBF kernel or t-Student kernel to satisfy the doubly stochasticity constraints.

    Parameters
    ----------
    eps : float, optional
        The strength of the regularization for the Sinkhorn algorithm. 
        It corresponds to the square root of the length scale of the Gaussian kernel when student = False, by default 1.0.
    f : torch.Tensor of shape (n_samples), optional
        Initialization for the dual variable of the Sinkhorn algorithm, by default None.
    tol : float, optional
        Precision threshold at which the algorithm stops, by default 1e-5.    
    max_iter : int, optional
        Number of maximum iterations for the algorithm, by default 100.
    student : bool, optional
        Whether to use a t-Student kernel instead of a Gaussian kernel, by default False.
    verbose : bool, optional
        Verbosity, by default False.
    tolog : bool, optional
        Whether to store intermediate result in a dictionary, by default False.

    Attributes
    ----------
    log_ : dictionary
        Contains the dual variables at each iteration of the optimization algorithm when tolog = True.
    n_iter_: int
        Number of iterations run.

    """

    def __init__(self,
                 eps=1.0,
                 f=None,
                 tol=1e-5,
                 max_iter=100,
                 student=False,
                 verbose=False,
                 tolog=False):
        self.eps = eps
        self.f = f
        self.tol = tol
        self.max_iter = max_iter
        self.student = student
        self.tolog = tolog
        self.n_iter_ = 0
        self.verbose = verbose


    def compute_log_affinity(self, X):
        """Computes the doubly stochastic affinity matrix in log space. 
        Returns the log of the transport plan at convergence.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        log_P: torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix in log space. 
        """
        C = torch.cdist(X, X, 2)**2
        # If student is True, considers the Student-t kernel instead of Gaussian RBF
        if self.student:
            C = torch.log(1+C)
        log_P = self.log_selfsink(C)
        return log_P

    def log_selfsink(self, C):
        """Performs Sinkhorn iterations in log domain to solve the entropic "self" (or "symmetric") OT problem with symmetric cost C and entropic regularization eps.

        Parameters
        ----------
        C : torch.Tensor of shape (n_samples, n_samples)
            Distance matrix between samples.

        Returns
        -------
        log_P: torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix in log space. 
        """        

        if self.verbose:
            print(
                '---------- Computing the doubly stochastic affinity matrix ----------')
        n = C.shape[0]

        # Allows a warm-start if a dual variable f is provided
        f = torch.zeros(n) if self.f is None else self.f

        if self.tolog:
            self.log_['f'] .append(f.clone())

        # Sinkhorn iterations
        for k in range(self.max_iter+1):
            f = 0.5 * (f - self.eps*torch.logsumexp((f - C) / self.eps, -1))

            if self.tolog:
                self.log_['f'].append(f.clone())

            if torch.isnan(f).any():
                raise NanError(
                    f'NaN in self-Sinkhorn dual variable at iteration {k}')

            log_T = (f[:, None] + f[None, :] - C) / self.eps
            if (torch.abs(torch.exp(torch.logsumexp(log_T, -1))-1) < self.tol).all():
                self.n_iter_ = k
                if self.verbose:
                    print(f'breaking at iter {k}')
                break

            if k == self.max_iter-1:
                print('---------- Max iter attained ----------')

        return (f[:, None] + f[None, :] - C) / self.eps


def log_Pe(C, eps):
    """Returns the log of the directed affinity matrix of SNE with prescribed kernel bandwidth.

    Parameters
    ----------
    C : torch.Tensor of shape (n_samples, n_samples)
        Distance matrix between samples.
    eps : torch.Tensor of shape (n_samples)
        Kernel bandwidths vector.

    Returns
    -------
    log_P: torch.Tensor of shape (n_samples, n_samples)
        log of the directed affinity matrix of SNE.
    """

    log_P = - C / (eps[:, None])
    return log_P - torch.logsumexp(log_P, -1, keepdim=True)


def log_Pse(C, eps, mu, to_square=False):
    """Returns the log of the symmetric entropic affinity matrix with specified parameters epsilon and mu.

    Parameters
    ----------
    C: torch.Tensor of shape (n_samples, n_samples)
        Distance matrix between samples.
    eps: torch.Tensor of shape (n_samples)
        Symmetric entropic affinity dual variables associated to the entropy constraint.
    mu: torch.Tensor of shape (n_samples)
        Symmetric entropic affinity dual variables associated to the marginal constraint.
    to_square: bool, optional
        Whether to use the square of the dual variables associated to the entropy constraint, by default False. 
    """
    if to_square:
        return (mu[:, None] + mu[None, :] - 2*C)/(eps[:, None]**2 + eps[None, :]**2)
    else:
        return (mu[:, None] + mu[None, :] - 2*C)/(eps[:, None] + eps[None, :])


def Lagrangian(C, log_P, eps, mu, perp):
    """Computes the Lagrangian associated to the symmetric entropic affinity optimization problem.

    Parameters
    ----------
    C: torch.Tensor of shape (n_samples, n_samples)
        Distance matrix between samples.
    log_P: torch.Tensor of shape (n_samples, n_samples)
        log of the symmetric entropic affinity matrix.
    eps: torch.Tensor of shape (n_samples)
        Symmetric entropic affinity dual variables associated to the entropy constraint.
    mu: torch.Tensor of shape (n_samples)
        Symmetric entropic affinity dual variables associated to the marginal constraint.
    perp : int
        Perplexity parameter.

    Returns
    -------
    cost: float
        Value of the Lagrangian.
    """
    one = torch.ones(C.shape[0], dtype=torch.double)
    target_entropy = math.log(perp) + 1
    HP = entropy(log_P, log=True, ax=1)
    return torch.exp(torch.logsumexp(log_P + torch.log(C), (0, 1), keepdim=False)) + torch.inner(eps, (target_entropy - HP)) + torch.inner(mu, (one - torch.exp(torch.logsumexp(log_P, -1, keepdim=False))))
