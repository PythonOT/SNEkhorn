import torch
from tqdm import tqdm
from snekhorn.affinities import SymmetricEntropicAffinity, BistochasticAffinity, BaseAffinity, NormalizedGaussianAndStudentAffinity, EntropicAffinity
import snekhorn.root_finding as root_finding


class NotImplemenedError(Exception):
    pass


class NotBaseAffinityError(Exception):
    pass


OPTIMIZERS = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD}


class AffinityMatcher():
    # match two kernels as min_Z KL(P_X | Q_Z)
    def __init__(self,
                 affinity_in_Z,  # both are two BaseAffinity objects that computes the affinities
                 affinity_in_X,  # both are two BaseAffinity objects that computes the affinities
                 output_dim=2,
                 optimizer='Adam',
                 verbose=True,
                 tol=1e-4,
                 max_iter=100,
                 lr=1e-1,
                 tolog=False,
                 init='random'):

        assert optimizer in ['Adam', 'SGD']
        self.optimizer = optimizer
        self.verbose = verbose
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        if not isinstance(affinity_in_X, BaseAffinity) or not isinstance(affinity_in_X, BaseAffinity):
            raise NotBaseAffinityError(
                'affinity_in_X and affinity_in_X must be BaseAffinity and implement a compute_log_affinity method')
        self.affinity_in_X = affinity_in_X
        self.affinity_in_Z = affinity_in_Z  # should be in log space
        self.output_dim = output_dim
        if init not in ['random']:
            raise NotImplementedError(
                '{} initialisation strategy is not valid'.format(init))
        self.init = init
        self.tolog = tolog
        if tolog:
            self.log = {}

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        n = X.shape[0]
        P_X = self.affinity_in_X.compute_affinity(X)
        losses = []
        if self.init == "random":
            embedding_ = torch.normal(0, 1, size=(
                n, self.output_dim), dtype=torch.double)
        embedding_.requires_grad = True
        optimizer = OPTIMIZERS[self.optimizer]([embedding_], lr=self.lr)

        pbar = tqdm(range(self.max_iter))
        for k in pbar:
            optimizer.zero_grad()
            log_Q = self.affinity_in_Z.compute_log_affinity(
                embedding_)

            # pytorch reverse the standard definition of the KL div and impose that the input is in log space to avoid overflow
            loss = torch.nn.functional.kl_div(log_Q, P_X, reduction='sum')
            if torch.isnan(loss):
                raise Exception(f'NaN in loss at iteration {k}')

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
        if self.tolog:
            self.log['loss'] = losses
        self.embedding = embedding_.clone().detach()
        return self.embedding


class SNEkhorn(AffinityMatcher):
    def __init__(self,
                 perp,
                 output_dim=2,
                 optimizer='Adam',
                 verbose=True,
                 tol=1e-4,
                 max_iter=100,
                 lr=1e-1,
                 learning_rate_sea=1e-1,
                 max_iter_sea=1000,
                 tol_sea=1e-3,
                 squared_parametrization=True,
                 eps=1.0,  # regularization for sinkhorn
                 init_sinkhorn=None,
                 student_kernel=False,  # True for tSNEkhorn
                 max_iter_sinkhorn=100,
                 tol_sinkhorn=1e-5,
                 tolog=False):

        self.perp = perp
        symmetric_entropic_affinity = SymmetricEntropicAffinity(perp=perp,
                                                                lr=learning_rate_sea,
                                                                max_iter=max_iter_sea,
                                                                tol=tol_sea,
                                                                tolog=tolog,
                                                                optimizer=optimizer,
                                                                verbose=verbose,
                                                                squared_parametrization=squared_parametrization)
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
    def __init__(self,
                 perp,
                 output_dim=2,
                 optimizer='Adam',
                 verbose=True,
                 tol=1e-4,
                 max_iter=100,
                 lr=1e-1,
                 tol_ea=1e-5,
                 student_kernel=False,  # True for tSNE
                 tolog=False):
        self.perp = perp
        entropic_affinity = EntropicAffinity(perp=perp,
                                             tol=tol_ea,
                                             verbose=verbose,
                                             normalize_as_sne=True)
        affinity_in_Z = NormalizedGaussianAndStudentAffinity(
            student=student_kernel)

        super(tSNE, self).__init__(affinity_in_Z=affinity_in_Z,
                                   affinity_in_X=entropic_affinity,
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
