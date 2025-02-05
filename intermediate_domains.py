# intermediate_domains.py
import numpy as np
from scipy.spatial.distance import cdist
import ot

def compute_intermediate_domains_linear(source, target, alphas=[0.25, 0.50, 0.75]):
    """
    Returns a list of intermediate domains obtained by linear interpolation:
      For each alpha, features = (1 - alpha)*source + alpha*target.
    """
    intermediates = []
    for alpha in alphas:
        inter = (1 - alpha) * source + alpha * target
        intermediates.append(inter)
    return intermediates

def wasserstein_barycenter(Y, b, n, weight, max_iter=200, reg=1e-7, init=None):
    """
    Computes the Wasserstein barycenter of distributions Y with the given weights.
    
    Parameters:
      - Y: list of distributions (each of shape (d, n_points))
      - b: list of mass vectors (length n_points) for each distribution
      - n: number of points in the barycenter
      - weight: tuple of coefficients (e.g., (1-r, r))
      - max_iter: number of iterations
      - reg: regularization parameter (here, set to a very low value)
      - init: initial guess (matrix of shape (n, d)). If None, initializes from the source.
      
    Returns:
      - The barycenter as a matrix of shape (n, d)
    """
    N = len(Y)
    n = int(n)
    if init is None:
        tmp_Y0 = Y[0].T.copy()  # (n_points, d)
        X = tmp_Y0[:n, :].T    # X of shape (d, n)
    else:
        X = init.T.copy()
    a = np.ones(n) / n

    for t in range(1, max_iter + 1):
        teta = 1 / (1 + 0.1 * t)
        cost_matrices = [cdist(X.T, Y[i].T, metric='sqeuclidean') for i in range(N)]
        T_list = [ot.sinkhorn(a, b[i], cost_matrices[i], reg=reg, numItermax=1000)
                  for i in range(N)]
        g = np.zeros_like(X)
        for i in range(N):
            g += weight[i] * np.dot(Y[i], T_list[i].T)
        X = (1 - teta) * X + teta * (g / a[None, :])
    return X.T

def compute_intermediate_domains_barycenter(source, target, alphas=[0.25, 0.50, 0.75], reg=1e-7):
    """
    Computes intermediate domains between 'source' and 'target' using the Wasserstein barycenter.
    For each alpha, it forces initialization by linear interpolation.
    
    Returns a list of intermediate domains.
    """
    Y = [source.T, target.T]  # each distribution is of shape (d, n_points)
    n = source.shape[0]
    b = [np.ones(n) / n, np.ones(n) / n]
    intermediates = []
    for alpha in alphas:
        weights = (1 - alpha, alpha)
        init_guess = (1 - alpha) * source + alpha * target
        barycenter = wasserstein_barycenter(Y, b, n, weight=weights, max_iter=200, reg=reg, init=init_guess)
        intermediates.append(barycenter)
    return intermediates