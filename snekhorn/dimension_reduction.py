import torch
from tqdm import tqdm
from snekhorn.affinities import SymmetricEntropicAffinity, BistochasticAffinity, BaseAffinity, NormalizedGaussianAndStudentAffinity, EntropicAffinity, NanError
from snekhorn.utils import PCA


class NotBaseAffinityError(Exception):
    pass


class WrongInputFitError(Exception):
    pass


class WrongParameter(Exception):
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
    affinity_embedding : BaseAffinity
        The affinity in the embedding space (in Z, corresponds to Q_Z).
    affinity_data : "precomputed" or BaseAffinity
        The affinity in the input space (in X, corresponds to P_X). 
        If affinity_data is "precomputed" then a affinity matrix (instead of a BaseAffinity object) is needed as input for the fit method.
    output_dim : int, optional
        Dimension of the embedded space (corresponds to the number of features of Z), by default 2.
    optimizer : str, optional
        Which pytorch optimizer to use among ['SGD', 'Adam', 'NAdam'], by default 'Adam'.
    lr : float, optional
        Learning rate for the algorithm, usually in the range [1e-5, 10], by default 1e-1.
    init : str, optional
        Initialization of embedding Z among ['random', 'pca'], default 'pca'.
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
        When tolog=True it contains the log of affinity_embedding, affinity_data (if affinity_data is not precomputed) and the loss at each iteration.
    n_iter_: int
        Number of iterations run.
    embedding_ : torch.Tensor of shape (n_samples, output_dim)
        Stores the embedding vectors.
    PX_ :  torch.Tensor of shape (n_samples, n_samples)
        Fitted affinity matrix in the input space.
    """

    def __init__(self,
                 affinity_embedding,
                 affinity_data,
                 output_dim=2,
                 optimizer='Adam',
                 lr=1e-1,
                 init='pca',
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
        if not isinstance(affinity_data, BaseAffinity) and not affinity_data == "precomputed":
            raise NotBaseAffinityError(
                'affinity_data  must be BaseAffinity or "precomputed".')
        if not isinstance(affinity_embedding, BaseAffinity):
            raise NotBaseAffinityError(
                'affinity_embedding must be BaseAffinity and implement a compute_log_affinity method.')
        self.affinity_data = affinity_data
        self.affinity_embedding = affinity_embedding
        self.output_dim = output_dim
        if init not in ['random', 'pca']:
            raise NotImplementedError(
                '{} initialisation strategy is not valid'.format(init))
        self.init = init
        self.tolog = tolog
        self.log_ = {}

    def fit(self, X, y=None):
        """Fit X into an embedded space.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features) or torch.Tensor of shape (n_samples, n_samples)
            Data to embed or affinity matrix between samples if affinity_data="precomputed".
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
        """Fit X into an embedded space and returns the embedding.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features) or torch.Tensor of shape (n_samples, n_samples)
            Data to embed or affinity matrix between samples if affinity_embedding="precomputed".
        y : None
            Ignored.

        Returns
        -------
        Z : torch.Tensor of shape (n_samples, output_dim)
            Embedding of the training data in low-dimensional space.
        """
        n = X.shape[0]
        if isinstance(self.affinity_data, BaseAffinity):
            PX_ = self.affinity_data.compute_affinity(X)
        else:
            if X.shape[1] != n:
                raise WrongInputFitError(
                    'When affinity_data="precomputed" the input X in fit must be a torch.Tensor of shape (n_samples, n_samples)')
            if not torch.all(X >= 0):  # a bit quick and dirty
                raise WrongInputFitError(
                    'When affinity_data="precomputed" the input X in fit must be non-negative')
            PX_ = X

        self.PX_ = PX_
        losses = []
        if self.init == "random":  # To add different initialization strategies
            embedding = torch.normal(0, 1, size=(
                n, self.output_dim), dtype=torch.double)  # Z embedding
        elif self.init == "pca":
            pca = PCA(n_components=self.output_dim)
            embedding = pca.fit_transform(X)

        embedding.requires_grad = True
        optimizer = OPTIMIZERS[self.optimizer]([embedding], lr=self.lr)

        pbar = tqdm(range(self.max_iter))
        for k in pbar:
            optimizer.zero_grad()
            log_Q = self.affinity_embedding.compute_log_affinity(
                embedding)

            # pytorch reverse the standard definition of the KL div and impose that the input is in log space to avoid overflow
            loss = torch.nn.functional.kl_div(log_Q, PX_, reduction='sum')
            if torch.isnan(loss):
                raise NanError(
                    f'NaN in loss at iteration {k}, consider decreasing the learning rate.')

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
            if isinstance(self.affinity_data, BaseAffinity):
                self.log_['log_affinity_data'] = self.affinity_data.log_
            self.log_['log_affinity_embedding'] = self.affinity_embedding.log_
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
    init : str, optional
        Initialization of embedding Z among ['random', 'pca'], default 'pca'.
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
        If True the algorithm is not convex anymore but may be more stable in practice, by default False.
    eps : float, optional
        The strength of the regularization for the Sinkhorn algorithm that calculates the doubly stochastic affinity matrix. 
        It corresponds to the square root of the length scale of the Gaussian kernel when student_kernel = False, by default 1.0.
    init_sinkhorn : torch.Tensor of shape (n_samples), optional
        Initialization for the dual variable of the Sinkhorn algorithm, by default None.
    max_iter_sinkhorn : int, optional
         Number of maximum iterations for the Sinkhorn algorithm, by default 50.
    tol_sinkhorn : float, optional
         Precision threshold at which the Sinkhorn algorithm stops, by default 1e-5.
    verbose : bool, optional
        Verbosity, by default True.
    tolog : bool, optional
        Whether to store intermediate results in a dictionary, by default False.

    Attributes
    ----------
    log_ : dictionary
        Contains the log of affinity_embedding, affinity_data and the loss at each iteration (if tolog is True).
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
                 student_kernel=False,  # True for tSNEkhorn
                 optimizer='Adam',
                 lr=1e-1,
                 init='pca',
                 tol=1e-4,
                 max_iter=100,
                 lr_sea=1e-1,
                 max_iter_sea=500,
                 tol_sea=1e-3,
                 square_parametrization=False,
                 eps=1.0,  # Regularization for Sinkhorn
                 init_sinkhorn=None,
                 max_iter_sinkhorn=50,
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
                                                 max_iter=max_iter_sinkhorn,
                                                 verbose=False)

        super(SNEkhorn, self).__init__(affinity_embedding=sinkhorn_affinity,
                                       affinity_data=symmetric_entropic_affinity,
                                       output_dim=output_dim,
                                       optimizer=optimizer,
                                       verbose=verbose,
                                       tol=tol,
                                       max_iter=max_iter,
                                       lr=lr,
                                       init=init,
                                       tolog=tolog)


class SNE(AffinityMatcher):
    """This class compute the standard SNE/t-SNE algorithm with our own implementation. 
    Results may differ from those of scikit-learn that implements different initialisation and optimization strategies.
    In particular we do not use exaggeration and gradient approximations

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
    init : str, optional
        Initialization of embedding Z among ['random', 'pca'], default 'pca'.
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
        Contains the log of affinity_embedding, affinity_data and the loss at each iteration (if tolog is True).
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
                 student_kernel=False,  # True for tSNE
                 optimizer='Adam',
                 lr=1e-1,
                 init='pca',
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
        affinity_embedding = NormalizedGaussianAndStudentAffinity(
            student=student_kernel)

        super(SNE, self).__init__(affinity_embedding=affinity_embedding,
                                  affinity_data=entropic_affinity,
                                  output_dim=output_dim,
                                  optimizer=optimizer,
                                  verbose=verbose,
                                  tol=tol,
                                  max_iter=max_iter,
                                  lr=lr,
                                  init=init,
                                  tolog=tolog)


class DRWrapper(AffinityMatcher):
    """Wrapper for all the dimension reduction methods. You can choose the dimension reduction method directly by selecting the parameters affinity_data and affinity_embedding.

    Parameters
    ----------
    perp : int, optional
        Perplexity parameter for the entropic or symmetric entropic affinity. 
        Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and the number of samples. 
        Different values can result in significantly different results. The perplexity must be less than the number of samples.
        Ignored if affinity_data is not "entropic_affinity" or "symmetric_entropic_affinity", by default None.
    affinity_data : str, optional
         Which affinity to use on the data among ['precomputed', 'gaussian', 'gaussian_bistochastic', 'entropic_affinity', 'symmetric_entropic_affinity'].
         affinity_data = 'gaussian' corresponds to a NormalizedGaussianAndStudentAffinity(student=False) BaseAffinity.
         affinity_data = 'gaussian_bistochastic' corresponds to a BistochasticAffinity(student=False) BaseAffinity.
         affinity_data = 'entropic_affinity' corresponds to a EntropicAffinity() BaseAffinity.
         affinity_data = 'symmetric_entropic_affinity' corresponds to a SymmetricEntropicAffinity() BaseAffinity.
         If 'entropic_affinity' or 'symmetric_entropic_affinity' the perplexity parameter must be defined, by default 'gaussian'.
    affinity_embedding : str, optional
        Which affinity to use on the embedding among ['gaussian', 'student', 'gaussian_bistochastic', 'student_bistochastic']
        affinity_embedding = 'gaussian' corresponds to a NormalizedGaussianAndStudentAffinity(student=False) BaseAffinity.
        affinity_embedding = 'student' corresponds to a NormalizedGaussianAndStudentAffinity(student=True) BaseAffinity.
        affinity_embedding = 'gaussian_bistochastic' corresponds to a BistochasticAffinity(student=False) BaseAffinity.
        affinity_embedding = 'student_bistochastic' corresponds to a BistochasticAffinity(student=True) BaseAffinity.
        By default 'student'.
    params_affinity_data : dict, optional
        Contains the different parameters for computing the affinity on the data, see corresponding BaseAffinity class. 
        If not provided default parameters, by default {}.
    params_affinity_embedding : dict, optional
        Contains the different parameters for computing the affinity on the embedding, see corresponding BaseAffinity class. 
        If not provided default parameters, by default {}.
    output_dim : int, optional
        Dimension of the embedded space (corresponds to the number of features of Z), by default 2.
    optimizer : str, optional
        _description_, by default 'Adam'
    optimizer : str, optional
        Which pytorch optimizer to use among ['SGD', 'Adam', 'NAdam'], by default 'Adam'.
    lr : float, optional
        Learning rate for the algorithm, usually in the range [1e-5, 10], by default 1e-1.
    init : str, optional
        Initialization of embedding Z among ['random', 'pca'], default 'pca'.
    tol : float, optional
        Precision threshold at which the algorithm stops, by default 1e-4.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm, by default 100.
    tolog : bool, optional
        Whether to store intermediate results in a dictionary, by default False.

    Attributes
    ----------
    log_ : dictionary
        Contains the log of affinity_embedding, affinity_data and the loss at each iteration (if tolog is True).
    n_iter_: int
        Number of iterations run.
    embedding_ : torch.Tensor of shape (n_samples, output_dim)
        Stores the embedding vectors.
    PX_ :  torch.Tensor of shape (n_samples, n_samples)
        Fitted symmetric entropic affinity matrix in the input space.
    """

    def __init__(self,
                 perp=None,
                 affinity_data='gaussian',
                 affinity_embedding='student',
                 params_affinity_data={},
                 params_affinity_embedding={},
                 output_dim=2,
                 optimizer='Adam',
                 lr=1e-1,
                 init='pca',
                 verbose=True,
                 tol=1e-4,
                 max_iter=100,
                 tolog=False):

        self.perp = perp
        possible_data_affinities = [
            'precomputed', 'gaussian', 'gaussian_bistochastic', 'entropic_affinity', 'symmetric_entropic_affinity']
        possible_embedding_affinities = [
            'gaussian', 'student', 'gaussian_bistochastic', 'student_bistochastic']

        if affinity_data not in possible_data_affinities:
            raise WrongParameter(
                'affinity_data must be in {}'.format(possible_data_affinities))

        if affinity_embedding not in possible_embedding_affinities:
            raise WrongParameter('affinity_embedding must be in {}'.format(
                possible_embedding_affinities))

        if self.perp is None and affinity_data in ['entropic_affinity', 'symmetric_entropic_affinity']:
            raise WrongParameter(
                'When affinity_data is entropic_affinity or symmetric_entropic_affinity you must define the perplexity parameter perp.')

        # How to compute the affinity of the data
        if affinity_data == 'precomputed':
            affinity_in_X = 'precomputed'

        elif affinity_data == 'gaussian':
            affinity_in_X = NormalizedGaussianAndStudentAffinity(student=False)

        elif affinity_data == 'gaussian_bistochastic':
            if 'student' in params_affinity_data.keys() and params_affinity_data['student'] == True:
                raise WrongParameter(
                    'You have chosen a bistochastic RBF kernel (affinity_data == "gaussian_bistochastic") but params_affinity_data[student] = True; so both disagree.')

            affinity_in_X = BistochasticAffinity(
                student=False, tolog=tolog, **params_affinity_data)

        elif affinity_data == 'entropic_affinity':
            affinity_in_X = EntropicAffinity(
                perp=self.perp, verbose=verbose, **params_affinity_data)

        elif affinity_data == 'symmetric_entropic_affinity':
            affinity_in_X = SymmetricEntropicAffinity(
                perp=self.perp, tolog=tolog, verbose=verbose, **params_affinity_data)

        # How to compute the affinity of the embedding
        if affinity_embedding == 'gaussian':
            affinity_in_Z = NormalizedGaussianAndStudentAffinity(student=False)

        elif affinity_embedding == 'student':
            affinity_in_Z = NormalizedGaussianAndStudentAffinity(student=True)

        elif affinity_embedding == 'gaussian_bistochastic':
            if 'student' in params_affinity_embedding.keys() and params_affinity_embedding['student'] == True:
                raise WrongParameter(
                    'You have chosen a bistochastic RBF kernel (affinity_embedding == "gaussian_bistochastic") but params_affinity_embedding[student] = True; so both disagree.')

            affinity_in_Z = BistochasticAffinity(
                student=False, tolog=tolog, verbose=False, **params_affinity_embedding)

        elif affinity_embedding == 'student_bistochastic':
            if 'student' in params_affinity_embedding.keys() and params_affinity_embedding['student'] == False:
                raise WrongParameter(
                    'You have chosen a bistochastic student kernel (affinity_embedding == "student_bistochastic") but params_affinity_embedding[student] = False; so both disagree.')

            affinity_in_Z = BistochasticAffinity(
                student=True, tolog=tolog, verbose=False, **params_affinity_embedding)

        super(DRWrapper, self).__init__(affinity_embedding=affinity_in_Z,
                                        affinity_data=affinity_in_X,
                                        output_dim=output_dim,
                                        optimizer=optimizer,
                                        verbose=verbose,
                                        tol=tol,
                                        max_iter=max_iter,
                                        lr=lr,
                                        init=init,
                                        tolog=tolog)
