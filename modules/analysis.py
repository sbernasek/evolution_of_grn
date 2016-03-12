__author__ = 'sbernasek'

from modules.cells import Cell
from modules.plotting import *
import numpy as np
import glob as glob
import json as json

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

    # iterate across each generation
    for i, file in enumerate(sorted(files)):

        # open json and read contents
        with open(file, mode='r', encoding='utf-8') as f:
            js = json.load(f)

            # extract a bunch of dictionaries where keys are 'cell' and 'scores'
            selected_cells = {Cell.from_json(result['cell']): result['scores'] for result in js.values()}
            results[i] = selected_cells

    return results

def plot_pareto_front_size(results):
    ax = create_subplot_figure(dim=(1, 1), size=(8, 6))[0]
    ax.plot(list(results.keys()), [len(val) for val in results.values()], '-b', linewidth=3)
    ax.set_xlabel('Generation', fontsize=16)
    ax.set_ylabel('Number of Selected Cells', fontsize=16)
    return ax

def get_ordered_front(results):

    unordered_front = list(results.values())[-1]

    cells = unordered_front.keys() # list of unordered cells
    scores = unordered_front.values() # corresponding list of lists of scores

    # sort score and cell lists from left to right on first objective function axis

    scores, cells = zip(*sorted(zip(scores, cells), key = lambda x: x[0]))


    return cells, scores

def plot_pareto_objective_tradeoff(metrics, metric_names=['Objective 1', 'Objective 2'], plot_title=None):
    """
    Create bar plot of metric as a function of order along the pareto front.

    Parameters:
        metrics (list) - list of lists of metric values ordered by position on pareto front
    """

    # create axes
    ax = create_subplot_figure(dim=(1, 1), size=(8, 6))[0]
    ax2 = ax.twinx()

    # plot metric values
    ranks = np.arange(1, len(metrics[0])+1)
    ax.plot(ranks, metrics[0], '-b', linewidth=3)
    ax2.plot(ranks, metrics[1], '-r', linewidth=3)

    # add proxy artists
    ax.plot([], '-b', linewidth=3, label=metric_names[0])
    ax.plot([], '-r', linewidth=3, label=metric_names[1])

    # create bar plot, remove vertical grid and xtick marks
    ax.xaxis.grid(False)
    plt.tick_params(axis='x', which='both', bottom='off')

    # set tick labels as range from most robust to most efficient
    ax.set_xticks(ranks)
    tick_labels=ax.get_xticks().tolist()
    for i in range(0, len(tick_labels)):
        tick_labels[i] = ''
    tick_labels[2]='Most Robust'
    tick_labels[-3]='Most Efficient'
    ax.set_xlim(0, len(metrics[0])+1)
    ax.set_xticklabels(tick_labels, fontsize=16)
    ax.set_xlabel('Selected Cells', fontsize=16)

    # label axes and figure
    ax.set_ylabel(metric_names[0], fontsize=16)
    ax2.set_ylabel(metric_names[1], fontsize=16)
    if plot_title is not None:
        ax.set_title(plot_title, fontsize=16)
    ax.legend(loc=(0.4, 0.8))

    return ax

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
    plt.tick_params(axis='x', which='both', bottom='off')

    # set tick labels as range from most robust to most efficient
    ax.set_xticks(ranks)
    tick_labels=ax.get_xticks().tolist()
    for i in range(0, len(tick_labels)):
        tick_labels[i] = ''
    tick_labels[2]='Most Robust'
    tick_labels[-2]='Most Efficient'
    ax.set_xticklabels(tick_labels, fontsize=16)
    ax.set_xlabel('Selected Cells', fontsize=16)

    # label axes and figure
    ax.set_ylabel(metric_name, fontsize=16)
    if plot_title is not None:
        ax.set_title(plot_title, fontsize=16)

    ax.set_ylim(0, 1.2*max(metric))

    return ax

def get_topology_from_front(results):
    """
    Returns lists of topological metrics from the pareto front, ordered from highest robustness to highest efficiency.
    """

    # get ordered pareto front (cells ordered by first objective function score)
    cells, scores = get_ordered_front(results)

    # initialize lists
    robustness = []
    energy_usage = []
    edge_count = []
    node_count = []
    edges_per_node = []
    edges_per_gene = []

    # # for the future...
    # coherent_ff_loops = []
    # incoherent_ff_loops = []
    # fb_loops = []

    # store topological features of each cell on the pareto front
    for cell, score in zip(cells, scores):

        # get cell scores
        robustness.append(score[0])
        energy_usage.append(score[1])

        # get cell topology
        edges, nodes, key = cell.get_topology()

        # append number of edges
        edge_count.append(len(edges))
        node_count.append(len(nodes))

        # add edge density
        edges_per_node.append(len(edges)/len(nodes))
        edges_per_gene.append(len(edges)/ len([node for node, node_type in nodes.items() if node_type != 'modified protein']) )


    return robustness, energy_usage, edge_count, node_count, edges_per_node, edges_per_gene