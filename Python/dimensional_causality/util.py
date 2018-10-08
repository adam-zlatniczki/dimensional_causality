import numpy as np


def embed(x, emb_dim, tau):
    """
    Applies Takens' time-delay embedding to the time series x.

    :param x: The time series.
    :type x: numpy.ndarray
    :param emb_dim: Embedding dimension
    :type emb_dim: int
    :param tau: Time-delay
    :type tau: int
    :return: 2D matrix
    :rtype: numpy.ndarray
    """
    n = x.shape[0] - (emb_dim - 1)*tau
    X = np.zeros((n,emb_dim))
    for i in range(1,emb_dim+1):
        X[:,i-1] = x[(emb_dim-i)*tau:(emb_dim-i)*tau+n]
    return X
