__author__ = 'Sebi'

import time as time
from modules.cells import *
from modules.pareto import *
from modules.fitness import *


"""
TO DO:
    1. write robustness test (error free?)
    2. normalize by area in performance metric
    2. make graphics for paper and video
    3. write paper
"""


def evaluate(f, cells, input_node=None, output_node=None, ss_dict=None):
    """
    Score the performance of all cells.

    Parameters:
        f (function) - test function to be applied to each cell
        cells (list) - list of cell objects
        input_node (int) - index of node to which input signal is sent
        output_node (int) - index of node from which output is read
        ss_dict (dict) - dictionary in which keys are cell objects and values are steady state arrays

    Returns:
        scores (list) - list of objective-space coordinates for each cell in cells
    """

    # if no ss_dict provided, let scoring procedure calculate each steady state
    if ss_dict is None:
        steady_states = None

    scores = []
    for cell in cells:
        steady_states = ss_dict[cell]
        score = f(cell, input_node=input_node, output_node=output_node, steady_states=steady_states)
        scores.append(score)
    return scores


def filter_scores(raw, tol=1e10):
    """
    Removes scores with any None, inf, nan, or values greater than the specified tolerance.

    Parameters:
        raw (list) - list of objective-space coordinates

    Returns:
        filtered (list) - list of filtered objective-space coordinates

    """

    filtered = []
    for score in raw:
        if None in score or float('inf') in score:
            pass
        elif sum([np.isnan(x) for x in score if x is not None]) > 0:
            pass
        elif len([x for x in score if x > tol]) > 0:
            pass
        else:
            filtered.append(score)

    return filtered


def run_simulation(generations=10, population_size=20, mutations_per_division=2, test=adaptation_test):
    """
    Runs full simulation procedure.

    Parameters:
        generations (int) - number of growth/selection stages
        population_size (int) - number of cells per generation
        mutations_per_division (int) - number of mutations per cell cycle
        test (function) - test function used to score each cell

    Returns:
        populations (list) - dictionary of all cells selected throughout procedure. keys are generation numbers, while
        values are dictionaries of (cell: scores) entries.
    """

    # define input and output to be tested
    input_node = 2
    output_node = 1

    # simulation parameters
    cell_type = 'prokaryote'  # defines simulation type

    # initialize cell population as a single cell with 3 genes, 2 of which are permanent (corresponds to get_fitness_2)
    population = []
    populations, ss_dict, score_dict = {}, {}, {}

    # iterate through selection+growth cycles
    for gen in range(0, generations):
        start = time.time()

        # encourage extra mutations in initial population
        if gen == 0:
            mutations_used = 10
        else:
            mutations_used = mutations_per_division

        # grow remaining cells back to desired population size
        new_cells = []
        while len(population) + len(new_cells) < population_size:

            # if no cells exist, use a plain template cell
            if len(population) == 0:
                cell = Cell(name=1, removable_genes=0, permanent_genes=2, cell_type=cell_type)
            else:
                cell = np.random.choice(population)

            # divide cells
            _, mutant = cell.divide(num_mutations=mutations_used)

            # only accept new cells in which the output is dependent upon the input, and a stable nonzero steady state is achieved
            steady_states = mutant.get_steady_states(input_node=input_node, input_magnitude=1)
            connected = mutant.interaction_check_numerical(input_node=input_node, output_node=output_node, steady_states=steady_states)

            if connected is True and steady_states is not None and steady_states[mutant.key[output_node]] >= 1:
                new_cells.append(mutant)
                ss_dict[mutant] = steady_states

        # run dynamics and score each new cell
        new_scores = evaluate(test, new_cells, input_node, output_node, ss_dict=ss_dict)

        # filter any scores with None, inf, nan, or values >1e15
        new_scores_considered = filter_scores(new_scores, tol=1e15)

        # if valid scores remain, select pareto front. if no valid scores remain, re-seed population
        scores_considered = new_scores_considered + list(score_dict.values())
        if len(scores_considered) > 0:

            # specify whether maximizing or minimizing along each dimension (True means max), then get pareto front
            goal = [False for _ in scores_considered[0]]
            scores_selected = get_pareto_front(scores_considered, goal=goal)

            # merge selected old cells with selected new cells
            score_dict = {cell: score for cell, score in score_dict.items() if score in scores_selected}
            score_dict_addition = {new_cells[i]: score for i, score in enumerate(new_scores) if score in scores_selected}
            score_dict.update(score_dict_addition)
            population = [cell for cell in score_dict.keys()]

        else:
            # if no valid scores exist, end simulation
            print('All cells produced invalid results, simulation ended.')
            break

        # store population of selected cells along with corresponding scores
        populations[gen] = {cell: score for cell, score in score_dict.items()}

        # display time to complete current generation
        stop = time.time()
        print('Generation ', gen, 'took', stop-start, 'seconds')

        # get topology distribution for current generation's networks
        #edge_counts, node_counts = zip(*list(map(lambda x: (len(x[0]), len(x[1])), [cell.get_topology() for cell in population])))
        #print('Generation', gen, ': %g edges in average selected network' % np.mean(edge_counts))

    return populations

# some other stuff should we want it for results/analysis
    # # show topology of a random cell in the final population
    # random_cell = np.random.choice(population)
    # random_cell.show_topology(graph_layout='shell')
    #
    # # get some basic distribution statistics
    # age_distribution = [cell.name for cell in population]
    # edge_count_distribution, node_count_distribution = zip(*list(map(lambda x:(len(x[0]), len(x[1])), [cell.get_topology() for cell in population])))
    # network_gene_count_distribution = [len(cell.removable_genes+cell.non_coding_rnas) for cell in population]


def plot_1D_trajectory(populations, obj=0):
    """
    Plots evolutionary trajectory of a specified objective function with generation.

    Parameters:
        populations (dict) - dictionary of all cells selected throughout procedure. keys are generation numbers, while
        values are dictionaries of (cell: scores) entries.

    Returns:
        ax (axes object)
    """

    # convert to (gen: scores) form
    score_evolution = {gen: list(score_dict.values()) for gen, score_dict in populations.items()}

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


def plot_2D_trajectory(populations, obj=None, connect_front=False, labels=None):
    """
    Plots evolutionary trajectory of two specified objective functions in objective-space.

    Parameters:
        populations (dict) - dictionary of all cells selected throughout procedure. keys are generation numbers, while
        values are dictionaries of (cell: scores) entries.
        obj (tup) - tuple of indices of objective-coordinates to be visualized
        connect_front (bool) - if True, draw line connecting pareto front
        labels (list) - list of string labels for objective function names

    Returns:
        ax (axes object)
    """

    # convert to (gen: scores) form
    score_evolution = {gen: list(score_dict.values()) for gen, score_dict in populations.items()}

    # check that objective space has at least two dimensions
    dim = len(score_evolution[0][0])
    if dim < 2:
        print('Error: Objective space only has %d dimension(s).' % dim)
        return

    # create axes
    ax = create_subplot_figure(dim=(1, 1), size=(8, 6))[0]

    # if objective space is two dimensional, plot points on objective-space plane
    min_x, max_x = 1e3, 1e-3
    min_y, max_y = 1e10, 1e-10
    for gen, scores in score_evolution.items():

        # plot all points in generation
        coordinates = [coordinate for coordinate in zip(*scores)]
        x, y = coordinates[obj[0]], coordinates[obj[1]]

        color = [gen/len(score_evolution), 0/256, 0/256]
        ax.plot(x, y, '.', markersize=25, color=color)

        if connect_front is True and gen == list(populations.keys())[-1]:
            y, x = zip(*sorted(zip(y, x), reverse=True))
            ax.plot(x, y, '-', linewidth=5, color=color, zorder=0)

        # update minima/maxima for plot axis scaling
        min_x, min_y = min(x + (min_x,)), min(y + (min_y,))
        max_x, max_y = max(x + (max_x,)), max(y + (max_y,))

    # format plot
    ax.set_xlim(0.75*min_x, 1.25*max_x)
    ax.set_ylim(0.75*min_y, 1.25*max_y)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Evolutionary Trajectory', fontsize=16)

    if labels is None:
        ax.set_xlabel('Objective %d' % obj[0], fontsize=16)
        ax.set_ylabel('Objective %d' % obj[1], fontsize=16)
    else:
        ax.set_xlabel(labels[0], fontsize=16)
        ax.set_ylabel(labels[1], fontsize=16)

    # plot artists
    ax.plot([], '.k', markersize=25, label='First Generation')
    ax.plot([], '.', markersize=25, color=[0.5, 0, 0], label='Middle Generation')
    ax.plot([], '.r', markersize=25, label='Last Generation')
    if connect_front is True:
        ax.plot([], '-r', linewidth=5, label='Pareto Front')
    ax.legend(loc=0)

    return ax