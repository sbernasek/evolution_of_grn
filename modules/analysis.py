__author__ = 'sbernasek'

from modules.cells import Cell
from modules.plotting import *
import numpy as np
import glob as glob
import json as json
import matplotlib.colors as colors
import matplotlib.cm as cmx

def load_results_from_json(file_path):
    """
    Reads all json files within the specified path and returns a single results dictionary in which keys are generation
    numbers and values are nested dictionaries in which keys are selected cell objects and values are a list of scores.

    Parameters:
        file_path (str) - directory in which json files reside

    Returns:
        results (dict) - dictionary in which keys are generations, values are dictionaries with cell, scores pairs
    """

    # get all json files in results folder
    files = glob.glob(file_path + '*.json')

    # initialize results
    results = {}

    # iterate across each file in generational order
    get_generation = lambda x: int(x[len(file_path): -len('.json')])
    for i, file in enumerate(sorted(files, key=get_generation)):

        # open json and read contents
        with open(file, mode='r', encoding='utf-8') as f:
            js = json.load(f)

            # extract a bunch of dictionaries where keys are 'cell' and 'scores'
            selected_cells = {Cell.from_json(result['cell']): result['scores'] for result in js.values()}
            results[i] = selected_cells

    return results

def plot_pareto_front_size(results):
    ax = create_subplot_figure(dim=(1, 1), size=(8, 6))[0]
    num_selected_cells = [len(val) for val in results.values()]
    ax.plot(list(results.keys()), num_selected_cells, '-b', linewidth=3)
    ax.set_ylim(0, max(num_selected_cells)+2)
    ax.set_xlabel('Generation', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Cells Selected', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)
    ax.xaxis.grid(False), ax.yaxis.grid(False)
    return ax

def get_ordered_front(results):

    unordered_front = list(results.values())[-1]

    cells = unordered_front.keys() # list of unordered cells
    scores = unordered_front.values() # corresponding list of lists of scores

    # sort score and cell lists from left to right on first objective function axis

    scores, cells = zip(*sorted(zip(scores, cells), key=lambda x: x[0]))


    return cells, scores

def plot_pareto_objective_tradeoff(metrics, metric_names=['Objective 1', 'Objective 2'], plot_title=None):
    """
    Create bar plot of metric as a function of order along the pareto front.

    Parameters:
        metrics (list) - list of lists of metric values ordered by position on pareto front
    """

    # create axes
    ax1 = create_subplot_figure(dim=(1, 1), size=(8, 6))[0]
    ax2 = ax1.twinx()

    # plot metric values
    ranks = np.arange(1, len(metrics[0])+1)
    ax1.plot(ranks, metrics[0], '-b', linewidth=3)
    ax2.plot(ranks, metrics[1], '-r', linewidth=3)

    # add proxy artists
    ax1.plot([], '-b', linewidth=3, label=metric_names[0])
    ax1.plot([], '-r', linewidth=3, label=metric_names[1])

    # create bar plot, remove vertical grid and xtick marks
    ax1.xaxis.grid(False), ax1.yaxis.grid(False)
    plt.tick_params(axis='x', which='both', bottom='off')

    # set tick labels as range from most robust to most efficient
    ax1.set_xticks(ranks)
    tick_labels = ax1.get_xticks().tolist()
    for i in range(0, len(tick_labels)):
        tick_labels[i] = ''
    tick_labels[3] = 'Most Robust'
    tick_labels[-4] = 'Most Efficient'

    ax1.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off')         # ticks along the top edge are off

    ax1.set_xlim(0, len(tick_labels)+1)
    ax1.set_xticklabels(tick_labels, fontsize=16, fontweight='bold', ha='center')
    ax1.set_xlabel('Selected Cells', fontsize=20)

    # change ytick font size
    _ = [ytick.set_fontsize(16) for ytick in ax1.get_yticklabels()]
    _ = [ytick.set_fontsize(16) for ytick in ax2.get_yticklabels()]

    # label axes and figure
    ax1.set_ylabel(metric_names[0], fontsize=18, fontweight='bold')
    ax2.set_ylabel(metric_names[1], fontsize=18, fontweight='bold')
    if plot_title is not None:
        ax1.set_title(plot_title, fontsize=16)
    ax1.legend(loc=(0.4, 0.8))

    return ax1, ax2

def plot_metric(metric, metric_name='Metric', plot_title=None):
    """
    Create bar plot of metric as a function of order along the pareto front.

    Parameters:
        metric (list) - list of metric values ordered by position on pareto front
    """

    # create axes
    ax = create_subplot_figure(dim=(1, 1), size=(8, 6))[0]

    # generate rank numbers
    ranks = np.arange(1, len(metric)+1)

    # create bar plot, remove vertical grid and xtick marks
    ax.bar(ranks, metric, align='center', alpha=0.5)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    plt.tick_params(axis='x', which='both', bottom='off')

    # set tick labels as range from most robust to most efficient
    ax.set_xticks(ranks)
    tick_labels = ax.get_xticks().tolist()
    for i in range(0, len(tick_labels)):
        tick_labels[i] = ''
    tick_labels[3] = 'Most Robust'
    tick_labels[-4] = 'Most Efficient'
    ax.set_xlim(0, len(tick_labels)+1)

    ax.set_xticklabels(tick_labels, fontsize=16, fontweight='bold', ha='center')
    ax.set_xlabel('Selected Cells', fontsize=20)

    # change ytick font size
    _ = [ytick.set_fontsize(16) for ytick in ax.get_yticklabels()]

    # label axes and figure
    ax.set_ylabel(metric_name, fontsize=18, fontweight='bold')


    if plot_title is not None:
        ax.set_title(plot_title, fontsize=16)

    ax.set_ylim(0, 1.2*max(metric))

    return ax

def get_topology_from_front(results):
    """
    Returns dictionaries of topological metrics from the pareto front, ordered from highest robustness to highest efficiency.
    """

    # get ordered pareto front (cells ordered by first objective function score)
    cells, scores = get_ordered_front(results)

    # initialize lists for network size distribution
    network_size = {
        'robustness': [],
        'energy_usage': [],
        'edge_count': [],
        'node_count': [],
        'edges_per_gene': []
    }

    # initialize lists for node type distributions
    node_types = {
        'genes': [],
        'modified_proteins': [],
        'micro_rnas': []
    }

    # initialize lists for edge type distributions
    edge_types = {
        'TR': [],
        'TA': [],
        'PTR': [],
        'CD': [],
        'M': [],
        'CM': []
    }

    # # for the future...
    # coherent_ff_loops = []
    # incoherent_ff_loops = []
    # fb_loops = []

    # store topological features of each cell on the pareto front
    for cell, score in zip(cells, scores):

        # get cell scores
        network_size['robustness'].append(score[0])
        network_size['energy_usage'].append(score[1])

        # get cell topology
        edges, nodes, key = cell.get_topology()

        # append number of edges
        network_size['edge_count'].append(len(edges))
        network_size['node_count'].append(len(nodes))

        # add edge density
        network_size['edges_per_gene'].append(len(edges)/ len([node for node, node_type in nodes.items() if node_type != 'modified protein']))

        # get number of each type of node
        node_types['genes'].append(len([1 for node_type in nodes.values() if node_type in ['removable gene']]))
        node_types['modified_proteins'].append(len([1 for node_type in nodes.values() if node_type in ['modified protein']]))
        node_types['micro_rnas'].append(len([1 for node_type in nodes.values() if node_type in ['non-coding gene']]))

        # get number of each type of edge
        edge_types['TR'].append(len([1 for edge in edges if edge[2] in ['repression']]))
        edge_types['TA'].append(len([1 for edge in edges if edge[2] in ['activation']]))
        edge_types['PTR'].append(len([1 for edge in edges if edge[2] in ['miRNA_silencing']]))
        edge_types['CD'].append(len([1 for edge in edges if edge[2] in ['catalytic_degradation']]))
        edge_types['M'].append(len([1 for edge in edges if edge[2] in ['modification']]))
        edge_types['CM'].append(len([1 for edge in edges if edge[2] in ['catalytic_modification']]))

    return network_size, node_types, edge_types


def plot_1D_trajectory(results, obj=0):
    """
    Plots evolutionary trajectory of a specified objective function with generation.

    Parameters:
        results (dict) - dictionary of all cells selected throughout procedure. keys are generation numbers, while
        values are dictionaries of (cell: scores) entries.

    Returns:
        ax (axes object)
    """

    # convert to (gen: scores) form
    score_evolution = {gen: list(score_dict.values()) for gen, score_dict in results.items()}

    # create axes
    ax = create_subplot_figure(dim=(1, 1), size=(8, 6))[0]

    max_x = 1
    for gen, scores in score_evolution.items():

        # plot all coordinates in generation
        coordinates = [coordinate for coordinate in zip(*scores)]
        x = coordinates[obj]
        ax.plot([gen for _ in x], x, '.b', markersize=10)

        # update maximum for plot axis scaling
        max_x = max(x + (max_x,))

    # format plot
    ax.set_xlim(0, len(score_evolution)+1)
    ax.set_ylim(0, 1.2*max_x)
    ax.set_xlabel('Generation', fontsize=16)
    ax.set_ylabel('Objective %d' % obj, fontsize=16)
    ax.set_title('Evolutionary Trajectory', fontsize=16)
    return ax


def plot_2D_trajectory(results, obj=None, connect_front=False, labels=None):
    """
    Plots evolutionary trajectory of two specified objective functions in objective-space.

    Parameters:
        results (dict) - dictionary of all cells selected throughout procedure. keys are generation numbers, while
        values are dictionaries of (cell: scores) entries.
        obj (tup) - tuple of indices of objective-coordinates to be visualized
        connect_front (bool) - if True, draw line connecting pareto front
        labels (list) - list of string labels for objective function names

    Returns:
        ax (axes object)
    """

    y_scaling = 1e-6

    # convert to (gen: scores) form
    score_evolution = {gen: list(score_dict.values()) for gen, score_dict in results.items()}

    # check that objective space has at least two dimensions
    dim = len(score_evolution[0][0])
    if dim < 2:
        print('Error: Objective space only has %d dimension(s).' % dim)
        return

    # create axes
    ax = create_subplot_figure(dim=(1, 1), size=(8, 6))[0]

    # unpack generations into long list of data
    x_data = []
    y_data = []
    gen_number = []

    for gen, scores in score_evolution.items():

        # plot all points in generation
        coordinates = [coordinate for coordinate in zip(*scores)]
        x, y = coordinates[obj[0]], coordinates[obj[1]]

        # scale y values
        y = tuple([y_*y_scaling for y_ in y])

        # add to stored data
        y_data.extend(y)
        x_data.extend(x)
        gen_number.extend([gen for _ in x])

        # plot pareto front for last generation
        if connect_front is True and gen == list(results.keys())[-1]:
            y, x = zip(*sorted(zip(y, x), reverse=True))
            ax.plot(x, y, '-', linewidth=10, color='k', zorder=0)

    # plot points
    im = ax.scatter(x_data, y_data, s=250, c=gen_number, marker='o', cmap='copper_r', alpha=0.5, lw=0)
    color_bar = plt.colorbar(im)
    color_bar.set_label('Generation', fontsize=16, fontweight='bold')
    color_bar.set_alpha(1)
    color_bar.draw_all()
    color_bar.ax.tick_params(labelsize=16)

    # format plot
    ax.set_xlim(0*0.75*min(x_data), 1.25*max(x_data))
    ax.set_ylim(0*0.75*min(y_data), 1.25*max(y_data))

    ax.tick_params(axis='both', labelsize=16)

    if labels is None:
        ax.set_xlabel('Objective %d' % obj[0], fontsize=16, fontweight='bold')
        ax.set_ylabel('Objective %d (%d)' % (obj[1], y_scaling), fontsize=16, fontweight='bold')
    else:
        ax.set_xlabel(labels[0], fontsize=16, fontweight='bold')
        ax.set_ylabel(labels[1] + ' (%s)' % str(y_scaling), fontsize=16, fontweight='bold')

    # add custom legend
    # if connect_front is True:
    #     ax.plot([], '-k', linewidth=10, label='Pareto Front')
    #     ax.legend(loc=0, prop={'size': 16})

    # turn off grid lines
    ax.xaxis.grid(False), ax.yaxis.grid(False)

    return ax


def plot_multiple_metrics(ax, metric, metric_name='Metric', include_axis_label=False):
    """
    Create bar plot of metric as a function of order along the pareto front.

    Parameters:
        metric (list) - list of metric values ordered by position on pareto front
    """

    # generate rank numbers
    ranks = np.arange(1, len(metric)+1)

    # create bar plot, remove vertical grid and xtick marks
    ax.bar(ranks, metric, align='center', alpha=0.5)
    ax.xaxis.grid(False), ax.yaxis.grid(False)
    ax.set_xlim(0, len(metric)+1)
    plt.tick_params(axis='x', which='both', bottom='off', top='off')

    # set y tick labels
    if max(metric) > 3:
        ax.yaxis.set_ticks(np.arange(0, np.ceil(max(metric))+1, int(max(metric)/3)))
    else:
        ax.yaxis.set_ticks(np.arange(0, np.ceil(max(metric))+1))

    # set x tick labels as range from most robust to most efficient
    if include_axis_label is True:
        ax.set_xticks(ranks)
        tick_labels = ax.get_xticks().tolist()
        for i in range(0, len(tick_labels)):
            tick_labels[i] = ''
        tick_labels[3] = 'Most Robust'
        tick_labels[-4] = 'Most Efficient'
        ax.set_xticklabels(tick_labels, fontsize=16, fontweight='bold', ha='center')
        ax.set_xlabel('Selected Cells', fontsize=20)
    else:
        ax.set_xticks([])

    # edit y axis
    _ = [ytick.set_fontsize(16) for ytick in ax.get_yticklabels()]
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.2*max(metric))
