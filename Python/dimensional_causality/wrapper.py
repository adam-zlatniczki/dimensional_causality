import os
from ctypes import cdll, c_double, c_uint, cast, POINTER


""" Load OS specific shared library """
cpu_dll_path = os.path.join(os.path.dirname(__file__), "dimensional_causality_cpu.dll")
cpu_so_path = os.path.join(os.path.dirname(__file__), "dimensional_causality_cpu.so")

libc = None

if os.name == "posix":
    libc = cdll.LoadLibrary(cpu_so_path)
else:
    libc = cdll.LoadLibrary(cpu_dll_path)

if libc is None:
    raise Exception("Compiled shared library not found! Make sure you installed the package without errors!")

libc.infer_causality.restype = POINTER(c_double)
""" Load library End """


def infer_causality(x, y, emb_dim, tau, k_range, eps=0.05, c=3.0, bins=20.0, downsample_rate=1):
    """
    Returns the probability of the possible causal cases in the following order:
        P(X -> Y), P(X <-> Y), P(X <- Y), P(X <- Z -> Y), P(X | Y)
    This is the implementation of the Dimensional Causality method developed in
    Benko, Zlatniczki, Fabo, Solyom, Eross, Telcs & Somogyvari (2018) - Inference of causal relations via dimensions.

    :param x: The first time series
    :type x: Array-like
    :param y: The second time series
    :type y: Array-like
    :param emb_dim: Embedding dimension
    :type emb_dim: int
    :param tau: Time delay
    :type tau: int
    :param k_range: List of k values for neighborhood sizes
    :type k_range: Array-like
    :param eps: Trimming quantile (0,1), default 0.05, trims both tails
    :type eps: float
    :param c: Number of standard deviations to keep in the distributions, default 3.0
    :type c: float
    :param bins: Number of bins that the distribution will be split into
    :type bins: float
    :return: Final probabilities
    :rtype: list
    """
    x_arr = (c_double * len(x))()
    y_arr = (c_double * len(x))()

    for i in range(len(x)):
        x_arr[i] = c_double(x[i])
        y_arr[i] = c_double(y[i])

    n = c_uint(len(x))
    emb_dim = c_uint(emb_dim)
    tau = c_uint(tau)

    k_range_arr = (c_uint * len(k_range))()

    for i in range(len(k_range)):
        k_range_arr[i] = c_uint(k_range[i])

    len_range = c_uint(len(k_range))
    eps = c_double(eps)
    c = c_double(c)
    bins = c_double(bins)
    downsample_rate = c_uint(downsample_rate)

    probs = libc.infer_causality(
        cast(x_arr, POINTER(c_double)),
        cast(y_arr, POINTER(c_double)),
        n,
        emb_dim,
        tau,
        cast(k_range_arr, POINTER(c_uint)),
        len_range,
        eps,
        c,
        bins,
        downsample_rate
    )

    return [probs[i] for i in range(5)]
