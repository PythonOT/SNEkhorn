import torch
from tqdm import tqdm
from snekhorn.affinities import SymmetricEntropicAffinity, BistochasticAffinity, BaseAffinity, NormalizedGaussianAndStudentAffinity, EntropicAffinity, NanError


class NotImplemenedError(Exception):
    pass


class NotBaseAffinityError(Exception):
    pass


OPTIMIZERS = {'SGD': torch.optim.SGD,
              'Adam': torch.optim.Adam,
              'NAdam': torch.optim.NAdam
              }


class AffinityMatcher():
    """This class matches two affinities together for dimension reduction purposes. 
    Given a dataset X, and a corresponding affinity matrix P_X in the input space, it computes the embedding Z whose corresponding affinity Q_Z is the closest to P_X w.r.t. the KL divergence. 
    It solves min_Z KL(P_X | Q_Z) with first order methods.

    Parameters
    ----------
    affinity_in_Z : BaseAffinity
        The affinity in the embedding space that computes Q_Z.
    affinity_in_X : BaseAffinity
        The affinity in the embedding space that computes P_X.
    output_dim : int, optional
        Dimension of the embedded space (corresponds to the number of features of Z), by default 2.
    optimizer : str, optional
        Which pytorch optimizer to use among ['SGD', 'Adam', 'NAdam'], by default 'Adam'.
    lr : float, optional
        Learning rate for the algorithm, usually in the range [1e-5, 10], by default 1e-1.
    init : str, optional
        Initialization of embedding Z among ['random'], default 'random'.
    verbose : bool, optional
        Verbosity, by default True.
    tol : float, optional
        Precision threshold at which the algorithm stops, by default 1e-4.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm, by default 100.
    tolog : bool, optional
        Whether to store intermediate results in a dictionary, by default False.

    Attributes
    ----------
    log_ : dictionary
        Contains the log of affinity_in_Z, affinity_in_X and the loss at each iteration (if tolog is True).
    n_iter_: int
        Number of iterations run.
    embedding_ : torch.Tensor of shape (n_samples, output_dim)
        Stores the embedding vectors.
    PX_ :  torch.Tensor of shape (n_samples, n_samples)
        Fitted affinity matrix in the input space.
    """    
    def __init__(self,
                 affinity_in_Z,  # BaseAffinity objects that computes the affinities in the embedding space Z
                 affinity_in_X,  # BaseAffinity objects that computes the affinities in the input space Z
                 output_dim=2,
                 optimizer='Adam',
                 lr=1e-1,
                 init='random',
                 verbose=True,
                 tol=1e-4,
                 max_iter=100,
                 tolog=False): 

        assert optimizer in ['Adam', 'SGD', 'NAdam']
        self.optimizer = optimizer
        self.verbose = verbose
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.P_X = None
        if not isinstance(affinity_in_X, BaseAffinity) or not isinstance(affinity_in_X, BaseAffinity):
            raise NotBaseAffinityError(
                'affinity_in_Z and affinity_in_X must be BaseAffinity and implement a compute_log_affinity method')
        self.affinity_in_X = affinity_in_X
        self.affinity_in_Z = affinity_in_Z  # should be in log space
        self.output_dim = output_dim
        if init not in ['random']:
            raise NotImplementedError(
                '{} initialisation strategy is not valid'.format(init))
        self.init = init
        self.tolog = tolog
        self.log_ = {}

    def fit(self, X, y=None):
        """Fit X into an embedded space.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data to embed.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted Estimator.
        """        
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        """_summary_

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data to embed.
        y : None
            Ignored.

        Returns
        -------
        Z : torch.Tensor of shape (n_samples, output_dim)
            Embedding of the training data in low-dimensional space.
        """        
        n = X.shape[0]
        PX_ = self.affinity_in_X.compute_affinity(X)
        self.PX_ = PX_
        losses = []
        if self.init == "random": #to add different initialization strategies (like PCA)
            embedding = torch.normal(0, 1, size=(
                n, self.output_dim), dtype=torch.double) #Z embedding
        embedding.requires_grad = True
        optimizer = OPTIMIZERS[self.optimizer]([embedding], lr=self.lr)

        pbar = tqdm(range(self.max_iter))
        for k in pbar:
            optimizer.zero_grad()
            log_Q = self.affinity_in_Z.compute_log_affinity(
                embedding)

            # pytorch reverse the standard definition of the KL div and impose that the input is in log space to avoid overflow
            loss = torch.nn.functional.kl_div(log_Q, PX_, reduction='sum')
            if torch.isnan(loss):
                raise NanError(f'NaN in loss at iteration {k}, consider decreasing the learning rate')

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if k > 1:
                delta = abs(losses[-1] - losses[-2]) / abs(losses[-2])
                if delta < self.tol:
                    if self.verbose:
                        print('---------- delta loss convergence ----------')
                    break
                if self.verbose:
                    pbar.set_description(f'Loss : {float(loss.item()): .3e}, '
                                         f'delta : {float(delta): .3e} '
                                         )
        self.embedding_ = embedding.clone().detach()
        self.n_iter_ = k 
        if self.tolog:
            self.log_['loss'] = losses
            self.log_['log_affinity_in_X'] = self.affinity_in_X.log_
            self.log_['log_affinity_in_Z'] = self.affinity_in_Z.log_
            self.log_['embedding'] = self.embedding_
        return self.embedding_


class SNEkhorn(AffinityMatcher):
    """This class computes the dimension reduction method presented in [1].  Given a dataset X, it first computes the corresponding symmetric entropic affinity P_X. 
    In the low-dimensional output space it computes the embedding Z whose doubly stochastic affinity Q_Z is closest to P_X w.r.t. the KL divergence. 
    It solves min_Z KL(P_X | Q_Z) with first order methods.

    Parameters
    ----------
    perp : int
        Perplexity parameter for the symmetric entropic affinity P_X. 
        Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and the number of samples. 
        Different values can result in significantly different results. The perplexity must be less than the number of samples.
    output_dim : int, optional
        Dimension of the embedded space (corresponds to the number of features of Z), by default 2.
    student_kernel : bool, optional
        Whether to use a normalized (symmetric + bistochastic) t-Student kernel instead of a Gaussian kernel in Z. 
        If True it computes tSNEkhorn instead of SNEkhorn (see [1]), by default False.
    optimizer : str, optional
        Which pytorch optimizer to use among ['SGD', 'Adam', 'NAdam'], by default 'Adam'.
    lr : float, optional
        Learning rate for the algorithm, usually in the range [1e-5, 10], by default 1e-1.
    tol : float, optional
        Precision threshold at which the algorithm stops, by default 1e-4.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm, by default 100.
    lr_sea : float, optional
        Learning for the computation of the symmetric entropic affinity, by default 1e-1.
    max_iter_sea : int, optional
         Number of maximum iterations for the computation of the symmetric entropic affinity, by default 500.
    tol_sea : _type_, optional
        Precision threshold at which the symmetric entropic affinity algorithm stops, by default 1e-3.
    square_parametrization : bool, optional
        Whether to optimize on the square of the dual variables for calculating the symmetric entropic affinity. 
        If True the algorithm is not convex anymore but is more stable in practice, by default True.
    eps : float, optional
        The strength of the regularization for the Sinkhorn algorithm that calculates the doubly stochastic affinity matrix. 
        It corresponds to the square root of the length scale of the Gaussian kernel when student_kernel = False, by default 1.0.
    init_sinkhorn : torch.Tensor of shape (n_samples), optional
        Initialization for the dual variable of the Sinkhorn algorithm, by default None.
    max_iter_sinkhorn : int, optional
         Number of maximum iterations for the Sinkhorn algorithm, by default 100.
    tol_sinkhorn : float, optional
         Precision threshold at which the Sinkhorn algorithm stops, by default 1e-5.
    verbose : bool, optional
        Verbosity, by default True.
    tolog : bool, optional
        Whether to store intermediate results in a dictionary, by default False.

    Attributes
    ----------
    log_ : dictionary
        Contains the log of affinity_in_Z, affinity_in_X and the loss at each iteration (if tolog is True).
    n_iter_: int
        Number of iterations run.
    embedding_ : torch.Tensor of shape (n_samples, output_dim)
        Stores the embedding vectors.
    PX_ :  torch.Tensor of shape (n_samples, n_samples)
        Fitted symmetric entropic affinity matrix in the input space.

    References
    ----------
    [1] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities, Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
    """    
    def __init__(self,
                 perp,
                 output_dim=2,
                 student_kernel=False,  #True for tSNEkhorn
                 optimizer='Adam',
                 lr=1e-1,
                 tol=1e-4,
                 max_iter=100,
                 lr_sea=1e-1,
                 max_iter_sea=500,
                 tol_sea=1e-3,
                 square_parametrization=True,
                 eps=1.0,  #Regularization for Sinkhorn
                 init_sinkhorn=None,
                 max_iter_sinkhorn=100,
                 tol_sinkhorn=1e-5,
                 verbose=True,
                 tolog=False):

        self.perp = perp
        symmetric_entropic_affinity = SymmetricEntropicAffinity(perp=perp,
                                                                lr=lr_sea,
                                                                max_iter=max_iter_sea,
                                                                tol=tol_sea,
                                                                tolog=tolog,
                                                                optimizer=optimizer,
                                                                verbose=verbose,
                                                                square_parametrization=square_parametrization)
        sinkhorn_affinity = BistochasticAffinity(eps=eps,
                                                 f=init_sinkhorn,
                                                 student=student_kernel,
                                                 tolog=tolog,
                                                 tol=tol_sinkhorn,
                                                 max_iter=max_iter_sinkhorn)

        super(SNEkhorn, self).__init__(affinity_in_Z=sinkhorn_affinity,
                                       affinity_in_X=symmetric_entropic_affinity,
                                       output_dim=output_dim,
                                       optimizer=optimizer,
                                       verbose=verbose,
                                       tol=tol,
                                       max_iter=max_iter,
                                       lr=lr,
                                       tolog=tolog)


class SNE(AffinityMatcher):
    """This class compute the standard SNE/t-SNE algorithm with our own implementation. 
    Results may differ from those of scikit-learn that implements different initialisation and optimization strategies.

    Parameters
    ----------
    perp : int
        Perplexity parameter for the entropic affinity P_X. 
        Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and the number of samples. 
        Different values can result in significantly different results. The perplexity must be less than the number of samples.
    output_dim : int, optional
        Dimension of the embedded space (corresponds to the number of features of Z), by default 2.
    student_kernel : bool, optional
        Whether to use a t-Student kernel instead of a Gaussian kernel in Z. 
        If True it computes tSNE instead of SNE, by default False.
    optimizer : str, optional
        Which pytorch optimizer to use among ['SGD', 'Adam', 'NAdam'], by default 'Adam'.
    lr : float, optional
        Learning rate for the algorithm, usually in the range [1e-5, 10], by default 1e-1.
    tol : float, optional
        Precision threshold at which the algorithm stops, by default 1e-4.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm, by default 100.
    tol_ea : _type_, optional
        Precision threshold at which the entropic affinity algorithm stops, by default 1e-5.
    verbose : bool, optional
        Verbosity, by default True.
    tolog : bool, optional
        Whether to store intermediate results in a dictionary, by default False.

    Attributes
    ----------
    log_ : dictionary
        Contains the log of affinity_in_Z, affinity_in_X and the loss at each iteration (if tolog is True).
    n_iter_: int
        Number of iterations run.
    embedding_ : torch.Tensor of shape (n_samples, output_dim)
        Stores the embedding vectors.
    PX_ :  torch.Tensor of shape (n_samples, n_samples)
        Fitted entropic affinity matrix in the input space.
    """    
    def __init__(self,
                 perp,
                 output_dim=2,
                 student_kernel=False,  #True for tSNE
                 optimizer='Adam',
                 lr=1e-1, 
                 tol=1e-4,
                 max_iter=100,
                 tol_ea=1e-5,
                 verbose=True,
                 tolog=False):
            
        self.perp = perp
        entropic_affinity = EntropicAffinity(perp=perp,
                                             tol=tol_ea,
                                             verbose=verbose,
                                             normalize_as_sne=True)
        affinity_in_Z = NormalizedGaussianAndStudentAffinity(
            student=student_kernel)

        super(SNE, self).__init__(affinity_in_Z=affinity_in_Z,
                                  affinity_in_X=entropic_affinity,
                                  output_dim=output_dim,
                                  optimizer=optimizer,
                                  verbose=verbose,
                                  tol=tol,
                                  max_iter=max_iter,
                                  lr=lr,
                                  tolog=tolog)
        

class SimpleSNEkhorn(AffinityMatcher):
 
    def __init__(self,
                 perp,
                 output_dim=2,
                 student_kernel=False,
                 optimizer='Adam',
                 lr=1e-1,
                 tol=1e-4,
                 max_iter=100,
                 lr_sea=1e-1,
                 max_iter_sea=500,
                 tol_sea=1e-3,
                 square_parametrization=True,
                 verbose=True,
                 tolog=False):

        self.perp = perp
        symmetric_entropic_affinity = SymmetricEntropicAffinity(perp=perp,
                                                                lr=lr_sea,
                                                                max_iter=max_iter_sea,
                                                                tol=tol_sea,
                                                                tolog=tolog,
                                                                optimizer=optimizer,
                                                                verbose=verbose,
                                                                square_parametrization=square_parametrization)
        affinity_in_Z = NormalizedGaussianAndStudentAffinity(
            student=student_kernel)

        super(SimpleSNEkhorn, self).__init__(affinity_in_Z=affinity_in_Z,
                                       affinity_in_X=symmetric_entropic_affinity,
                                       output_dim=output_dim,
                                       optimizer=optimizer,
                                       verbose=verbose,
                                       tol=tol,
                                       max_iter=max_iter,
                                       lr=lr,
                                       tolog=tolog)


# # ---------- Dimension reduction methods related to stochastic neighbor embedding, including ours ----------

# def log_kernel(C: torch.Tensor, kernel: str):
#     if kernel == 'student':
#         return - torch.log(1 + C)
#     else:  # Gaussian
#         return - 0.5 * C

# def SNE(X, Z, perp, **coupling_kwargs):
#     C = torch.cdist(X, X, p=2)**2
#     P = sne_affinity(C, perp)
#     return affinity_coupling(P, Z, **coupling_kwargs)

# def DSNE(X, Z, eps, **coupling_kwargs):
#     C = torch.cdist(X, X, p=2)**2
#     T0 = torch.exp(log_selfsink(C, eps=eps)[0])
#     return affinity_coupling(T0, Z, **coupling_kwargs)

# def SSNE(X, Z, perp, **coupling_kwargs):
#     C = torch.cdist(X, X, p=2)**2
#     P = symmetric_entropic_affinity(C, perp=perp)
#     return affinity_coupling(P, Z, **coupling_kwargs)

# def affinity_coupling(P0, Z, kernel=None, eps=1.0,  lr=1, max_iter=1000, optimizer='Adam', loss='KL', verbose=True, tol=1e-4, pz=2, exaggeration=False):

#     Z.requires_grad = True
#     f = None
#     optimizer = OPTIMIZERS[optimizer]([Z], lr=lr)
#     counter_cv = 0

#     log = {}
#     log['loss'] = []
#     # log['Z'] = []

#     pbar = tqdm(range(max_iter))
#     for k in pbar:
#         C = torch.cdist(Z, Z, p=pz)**2
#         # C.fill_diagonal_(0)
#         optimizer.zero_grad()

#         if exaggeration and k < 100:
#             P = 12*P0
#         else:
#             P = P0

#         if kernel == 'student' or kernel == 'gaussian':
#             log_k = log_kernel(C, kernel=kernel)
#             log_Q = log_k - torch.logsumexp(log_k, dim=(0, 1))
#         else:
#             assert eps is not None
#             student = (kernel == 'tsnekhorn')
#             log_Q, f = log_selfsink(C=C, eps=eps, f=f, student=student)
#         # else:
#         #     raise ValueError('Kernel not implemented')

#         loss = torch.nn.functional.kl_div(log_Q, P, reduction='sum')
#         if torch.isnan(loss):
#             raise Exception(f'NaN in loss at iteration {k}')

#         loss.backward()
#         optimizer.step()

#         log['loss'].append(loss.item())
#         # log['Z'].append(Z.clone().detach().cpu())

#         if k > 1:
#             delta = abs(log['loss'][-1] - log['loss'][-2]) / \
#                 abs(log['loss'][-2])
#             if delta < tol:
#                 counter_cv += 1
#                 if counter_cv > 10:  # Convergence criterion satisfied for more than 10 iterations in a row -> stop the algorithm
#                     if verbose:
#                         print('---------- delta loss convergence ----------')
#                     break
#             else:
#                 counter_cv = 0

#             if verbose:
#                 pbar.set_description(f'Loss : {float(loss.item()): .3e}, '
#                                      f'delta : {float(delta): .3e} '
#                                      )

#     return Z.detach(), log
