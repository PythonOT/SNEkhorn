import torch
from torchvision import transforms
import os
from PIL import Image

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