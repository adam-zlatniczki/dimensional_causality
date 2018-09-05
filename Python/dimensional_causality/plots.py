from matplotlib import pyplot as plt
from constants import *
import numpy as np


def _set_axis_xticks(ax, xvals):
    '''
    :param ax: axis
    :type ax:
    :param xvals: numpy array of x values
    :return:
    '''
    length_xvals = len(xvals)
    nth = length_xvals / NUM_TICKS

    # Check if the desired number of ticks is whether zero or not
    if nth == 0:
        nth = 1

    ax.set_xticks(xvals[::nth])
    ax.set_xticklabels([str(i) for i in xvals[::nth]])


def add_to_axis_k_range_dimensions(ax, k_range, exported_dims, exported_stdevs, show_std=1.0, title=None, x_label=None,
                       show_legend=True, matplotlib_rc=None):
    r"""For each value in the explored range the function plots onto the given axis the mean dimension of each manifold
    and shows the standard deviations (multiplied by *show_std*).

    :param ax: axis object to be drawn upon
    :type ax: matplotlib.axes._subplots.AxesSubplot
    :param k_range: A list containing the k-NN neighbourhood sizes
    :type k_range: list
    :param exported_dims: the 2D list containing the estimated dimensions
    :type exported_dims: list
    :param exported_stdevs: the 2D list containing the standard deviations of the estimated dimensions
    :type exported_stdevs: list
    :param show_std: How many standard deviations should be plotted.
    :type show_std: float
    :param title: The title of the plot
    :type title: str
    :param x_label: The label to overwrite the default of the X axis
    :type x_label: str
    :param show_legend: Show legend or not
    :type show_legend: bool
    :param matplotlib_rc: A matplotlib style configuration dictionary
    :type matplotlib_rc: dict
    :return: None
    """
    num_of_manifolds = 4

    # set figure parameters
    if matplotlib_rc is not None:
        plt.rcParams.update(matplotlib_rc)

    colors = [COLOR_X, COLOR_Y, COLOR_J, COLOR_Z]
    labels = [LABEL_X, LABEL_Y, LABEL_J, LABEL_Z]

    # set the range values as X axis
    x_values = k_range
    y_max = 0.0

    # for each manifold
    for i in range(num_of_manifolds):
        # update the Y axis scale
        manifold_range_means = np.asarray([exported_dims[j][i] for j in range(len(k_range))])
        manifold_range_stds = np.asarray([exported_stdevs[j][i] for j in range(len(k_range))])

        y_max = max(y_max, max(manifold_range_means))

        # plot manifold dimensions
        ax.plot(x_values, manifold_range_means, label=labels[i], color=colors[i])
        ax.fill_between(x_values,
                         manifold_range_means - show_std * manifold_range_stds,
                         manifold_range_means + show_std * manifold_range_stds,
                         color=colors[i],
                         alpha=ALPHA)

    if title is not None:
        ax.set_title(title)

    if x_label is None:
        ax.set_xlabel("k")
    else:
        ax.set_xlabel(x_label)

    ax.set_ylabel("Dimension")

    # ax.set_xticks(x_values)
    # ax.set_xticklabels([str(x) for x in x_values])
    _set_axis_xticks(ax, x_values)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_legend:
        ax.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left')


def plot_k_range_dimensions(k_range, exported_dims, exported_stdevs, show_std=1.0, title=None, x_label=None,
                       show_legend=True, matplotlib_rc=None):
    r"""For each value in the explored range the function plots the mean dimension of each manifold and shows the
    standard deviations (multiplied by *show_std*).

        :param k_range: A list containing the k-NN neighbourhood sizes
        :type k_range: list
        :param exported_dims: the 2D list containing the estimated dimensions
        :type exported_dims: list
        :param exported_stdevs: the 2D list containing the standard deviations of the estimated dimensions
        :type exported_stdevs: list
        :param show_std: How many standard deviations should be plotted.
        :type show_std: float
        :param title: The title of the plot
        :type title: str
        :param x_label: The label to overwrite the default of the X axis
        :type x_label: str
        :param show_legend: Show legend or not
        :type show_legend: bool
        :param matplotlib_rc: A matplotlib style configuration dictionary
        :type matplotlib_rc: dict
        :return: None
        """
    fig, ax = plt.subplots()
    add_to_axis_k_range_dimensions(ax, k_range, exported_dims, exported_stdevs, show_std, title, x_label,
                       show_legend, matplotlib_rc)
    plt.show(block=True)
