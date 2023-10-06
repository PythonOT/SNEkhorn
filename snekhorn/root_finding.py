import torch
from tqdm import tqdm

def init_bounds(f, n: int, begin=None, end=None):
    if begin is None:
        begin = torch.ones(n, dtype=torch.double)
    else:
        begin = begin * torch.ones(n, dtype=torch.double)

    if end is None:
        end = torch.ones(n, dtype=torch.double)
    else:
        end = end * torch.ones(n, dtype=torch.double)

    out_begin = f(begin)>0
    while out_begin.any():
        end[out_begin] = torch.min(end[out_begin], begin[out_begin])
        begin[out_begin] /= 2
        out_begin = f(begin)>0

    out_end = f(end)<0
    while out_end.any():
        begin[out_end] = torch.max(begin[out_end], end[out_end])
        end[out_end] *= 2
        out_end = f(end)<0
        
    return begin, end

def binary_search(f,
                n: int, 
                begin: torch.Tensor = None, 
                end: torch.Tensor = None, 
                max_iter: int = 1000, 
                tol: float = 1e-9,
                verbose: bool = False):
    """
        Performs binary search to find the root of an increasing function f.

        Parameters
        ----------
        f: function
            function which root should be computed
        n: int
            size of the input of f
        begin: array (n)
            initial lower bound of the root
        begin: array (n)
            initial upper bound of the root
        max_iter: int
            maximum iterations of search
        tol: float
            precision threshold at which the algorithm stops
        verbose: bool
            if True, prints the mean of current bounds
    """

    begin, end = init_bounds(f=f, n=n, begin=begin, end=end)
    m = (begin+end)/2
    fm = f(m)

    pbar = tqdm(range(max_iter), disable = not verbose)
    for _ in pbar:
        if torch.max(torch.abs(fm)) < tol:
            break
        sam = fm * f(begin) > 0
        begin = sam*m + (~sam)*begin
        end = (~sam)*m + sam*end
        m = (begin+end)/2
        fm = f(m)

        if verbose:
            mean_f = fm.mean().item()
            std_f = fm.std().item()
            pbar.set_description(f'f mean : {float(mean_f): .3e}, '
                                 f'f std : {float(std_f): .3e}, '
                                 f'begin mean : {float(begin.mean().item()): .6e}, '
                                 f'end mean : {float(end.mean().item()): .6e} ')
    return m, begin, end

def false_position(f,
                n: int, 
                begin: torch.Tensor = None, 
                end: torch.Tensor = None, 
                max_iter: int = 1000, 
                tol: float = 1e-9,
                verbose: bool = False):
    """
        Performs the false position method to find the root of an increasing function f.

        Parameters
        ----------
        f: function
            function which root should be computed
        n: int
            size of the input of f
        begin: array (n)
            initial lower bound of the root
        begin: array (n)
            initial upper bound of the root
        max_iter: int
            maximum iterations of search
        tol: float
            precision threshold at which the algorithm stops
        verbose: bool
            if True, prints the mean of current bounds
    """

    begin, end = init_bounds(f=f, n=n, begin=begin, end=end)
    f_begin, f_end = f(begin), f(end)
    m = begin - ((begin - end) / (f(begin) - f(end))) * f(begin)
    fm = f(m)

    pbar = tqdm(range(max_iter), disable = not verbose)
    for _ in pbar:
        if torch.max(torch.abs(fm)) < tol:
            break
        sam = fm * f_begin > 0
        begin = sam*m + (~sam)*begin
        f_begin = sam*fm + (~sam)*f_begin
        end = (~sam)*m + sam*end
        f_end = (~sam)*fm + sam*f_end
        m = begin - ((begin - end) / (f_begin - f_end)) * f_begin
        fm = f(m)

        if verbose:
            mean_f = fm.mean().item()
            std_f = fm.std().item()
            pbar.set_description(f'f mean : {float(mean_f): .3e}, '
                                 f'f std : {float(std_f): .3e}, '
                                 f'begin mean : {float(begin.mean().item()): .6e}, '
                                 f'end mean : {float(end.mean().item()): .6e} ')
    return m, begin, end
