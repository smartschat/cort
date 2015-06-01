""" Plot error analysis statistics. """

from __future__ import division


from matplotlib import pyplot
from matplotlib import cm

import numpy

from pylab import rcParams


__author__ = 'martscsn'


def plot(data,
         title,
         xlabel,
         ylabel,
         filename=None):
    """ Plot error analysis statistics.

    In particular, plot a bar chart for the numbers described in ``data``.

    Args:
        data (list(str, list((str,int)))): The data to be plotted. The ith entry
            of this list contains the name which will appear in the legend,
            and a list of (category, count) pairs. These are the individual
            data points which will be plotted.
        title (str): Title of the plot.
        xlabel (str): Label of the x axis.
        ylabel (str): Label of the y axis.
        filename (str, optional): If set, write plot to ``filename``.

    Example::
        pair_errs = errors["pair"]["recall_errors"]["all"]
        tree_errs = errors["tree"]["recall_errors"]["all"]

        plot(
            [("pair", [(cat, len(pair_errs[cat])) for cat in pair_errs.keys()]),
            ("tree", [(cat, len(tree_errs[cat])) for cat in tree_errs.keys()])],
            "Recall Errors",
            "Type of anaphor",
            "Number of Errors")
    """

    rcParams['xtick.major.pad'] = '12'
    rcParams['ytick.major.pad'] = '12'

    fig, ax = pyplot.subplots()

    systems = []
    categories = []

    colors = cm.Accent(numpy.linspace(0, 1, len(data)))

    bars_for_legend = []

    for i, system_data in enumerate(data):
        system_name, categories_and_numbers = system_data
        systems.append(system_name)

        for j, cat_and_number in enumerate(categories_and_numbers):
            category, number = cat_and_number

            if category not in categories:
                categories.append(category)

            bar = ax.bar(2*j + i*(1/len(data)), number, color=colors[i],
                         width=1/len(data), label=system_name)

            if j == 0:
                bars_for_legend.append(bar)

    xticks = [2*k + 0.5 for k in range(0, len(categories))]

    pyplot.title(title, fontsize=28)
    pyplot.xlabel(xlabel, fontsize=24)
    pyplot.ylabel(ylabel, fontsize=24)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_xticklabels(categories)
    ax.set_xticks(xticks)

    pyplot.tick_params(axis='both', which='major', labelsize=20)

    if filename:
        legend = ax.legend(bars_for_legend, systems,
                           loc='upper right', bbox_to_anchor=(1.2, 1.2))

        fig.savefig(filename, bbox_extra_artists=(legend,), bbox_inches='tight')
    else:
        legend = ax.legend(bars_for_legend, systems, loc='upper right')
        legend.draggable()

        fig.show()
