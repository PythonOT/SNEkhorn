import torch
from torchvision import transforms
import os
from PIL import Image
import torch.nn as nn

def COIL_dataset(dir=None):
    if dir is None:
        dir = '../data/coil-20-proc'
    n = 1440
    p = 16384
    X = torch.empty((n,p), dtype=torch.double)
    Y = torch.empty(n)
    imgs = []
    for i,filename in enumerate(os.listdir(dir)):
        img = Image.open(os.path.join(dir, filename))
        imgs.append(img)
        convert_tensor = transforms.ToTensor() 
        X[i] = convert_tensor(img)[0].view(-1).double()
        if filename[4]=='_':
            Y[i] = int(filename[3])
        else:
            Y[i] = int(filename[3:5])
    return X, Y


def entropy(P: torch.Tensor,
            log: bool = False,
            ax: int = -1):
    """
        Returns the entropy of P along axis ax, supports log domain input.

        Parameters
        ----------
        P: array (n,n)
            input data
        log: bool
            if True, assumes that P is in log domain
        ax: int
            axis on which entropy is computed
    """
    if log:
        return -(torch.exp(P)*(P-1)).sum(ax)
    else:
        return -(P*(torch.log(P)-1)).sum(ax)


def kl_div(P: torch.Tensor,
        K: torch.Tensor,
        log: bool=False):
    """
        Returns the Kullback-Leibler divergence between P and K, supports log domain input for both matrices.

        Parameters
        ----------
        P: array
            input data
        K: array
            input data
        log: bool
            if True, assumes that P and K are in log domain
    """
    if log:
        return (torch.exp(P) * (P - K - 1)).sum()
    else:
        return (P * (torch.log(P/K) - 1)).sum()
    

def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v

class PCA(nn.Module):
    # PCA implementation in torch that matches the scikit-learn implementation
    # see https://github.com/gngdb/pytorch-pca
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_