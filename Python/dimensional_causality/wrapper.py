import os
from ctypes import cdll, c_double, c_uint, cast, POINTER, pointer
from plots import plot_k_range_dimensions, plot_probabilities


""" Load OS specific shared library """
openmp_dll_path = os.path.join(os.path.dirname(__file__), "dimensional_causality_openmp.dll")
openmp_so_path = os.path.join(os.path.dirname(__file__), "dimensional_causality_openmp.so")

libc = None

if os.name == "posix":
    libc = cdll.LoadLibrary(openmp_so_path)
else:
    libc = cdll.LoadLibrary(openmp_dll_path)

if libc is None:
    raise Exception("Compiled shared library not found! Make sure you installed the package without errors!")

libc.infer_causality.argtypes = [POINTER(c_double), POINTER(c_double), c_uint, c_uint, c_uint, POINTER(c_uint), c_uint, c_double, c_double, c_double, c_uint, POINTER(POINTER(c_double)), POINTER(POINTER(c_double))]
libc.infer_causality.restype = POINTER(c_double)
""" Load library End """


def infer_causality(x, y, emb_dim, tau, k_range, eps=0.05, c=3.0, bins=20.0, downsample_rate=1, export_data=False, plot=True):
    """
    Returns a tuple. The first element is a list with the probability of the possible causal cases in the following order:
        P(X -> Y), P(X <-> Y), P(X <- Y), P(X <- Z -> Y), P(X | Y)
    The second element is a 2D list containing the dimensions estimates of each manifold for each k in the k_range.
    The third element is a 2D list containing the standard deviations of the dimensions estimates of each manifold for
    each k in the k_range.
    The 2D lists come with dimension (len(k_range) x 4).
    **Note that if you set plot=True, then you must have LaTeX installed.**

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
    :param downsample_rate: Specifies that every 'downsample_rate'-th point in the embedded manifold must only be kept
    :type downsample_rate: int
    :param plot: indicates whether plots should be drawn
    :type plot: bool
    :return: (case probabilities, exported dimension estimates, exported standard deviations for dimension estimates)
    :rtype: tuple of lists
    """
    export_data = export_data or plot

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

    export_dims = POINTER(c_double)() # create null pointer
    export_stdevs = POINTER(c_double)()
    if export_data:
        export_dims = pointer(export_dims) # convert to pointer of pointer
        export_stdevs = pointer(export_stdevs)

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
        downsample_rate,
        export_dims,
        export_stdevs
    )

    if export_data:
        export_dims = [[export_dims[0][i*4+j] for j in range(4)] for i in range(len(k_range))]

    if export_data:
        export_stdevs = [[export_stdevs[0][i * 4 + j] for j in range(4)] for i in range(len(k_range))]

    if not export_data:
        export_dims = None
        export_stdevs = None

    final_probabilities = [probs[i] for i in range(5)]

    if plot:
        plot_k_range_dimensions(k_range, export_dims, export_stdevs)
        plot_probabilities(final_probabilities)

    return final_probabilities, export_dims, export_stdevs
