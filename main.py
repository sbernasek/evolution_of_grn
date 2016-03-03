__author__ = 'Sebi'


from modules.cells import *
from modules.pareto import *
from modules.fitness import *


"""
TO DO (long term):


TO DO (near term):
    1. write robustness test
    2. write an abort procedure... when concentrations blow up just end simulation early to save time

"""


def evaluate(cells):
    """
    Score the performance of all cells.

    Parameters:
        cells (list) - list of cell objects

    Returns:
        scores (list) - list of objective-space coordinates for each cell in cells
    """

    scores = []
    for cell in cells:
        score = get_fitness_2(cell, mode='langevin')
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


def run_simulation(generations=10, population_size=20, mutations_per_division=2, retall=False):
    """
    Runs full simulation procedure.

    Parameters:
        generations (int) - number of growth/selection stages
        population_size (int) - number of cells per generation
        mutations_per_division (int) - number of mutations per cell cycle

    Returns:
        population (list) - list of cells taken from pareto front after specified number of generations
        score_evolution (dict) - if retall is True, return a dictionary in which keys are generations and values are lists
        of objective-space coordinates
    """

    # simulation parameters
    cell_type = 'prokaryote' # defines simulation type

    # initialize cell population as a single cell with 3 genes
    population = [Cell(name=1, removable_genes=0, permanent_genes=2, cell_type=cell_type)]

    # grow to desired population
    while len(population) < population_size:

        # clone a randomly selected cell
        cell = np.random.choice(population)
        cell, mutant = cell.divide(num_mutations=mutations_per_division)
        population.append(mutant)

    # store scores
    score_evolution = {}

    # iterate through selection+growth cycles
    for gen in range(0, generations):

        # get topology distribution for current generation's networks
        edge_counts, node_counts = zip(*list(map(lambda x: (len(x[0]), len(x[1])), [cell.get_topology() for cell in population])))
        print('Generation', gen, ': %g edges in average network' % np.mean(edge_counts))

        # run dynamics and score each cell
        scores = evaluate(population)

        # filter any scores with None, inf, nan, or values >1e10
        scores_considered = filter_scores(scores, tol=1e10)

        # if valid scores remain, select pareto front. if no valid scores remain, re-seed population
        if len(scores_considered) > 0:
            goal = [False, False]  # specify whether maximizing or minimizing along each dimension (True means max)
            scores_selected = get_pareto_front(scores_considered, goal=goal)
            population = [population[i] for i, score in enumerate(scores) if score in scores_selected]
        else:
            # if no valid scores were achieved, end simulation
            print('All cells produced invalid results, simulation ended.')
            break

        # store selected scores
        score_evolution[gen] = scores_selected

        # grow remaining cells back to desired population size
        while len(population) < population_size:
            cell = np.random.choice(population)
            _, mutant = cell.divide(num_mutations=mutations_per_division)
            population.append(mutant)

    if retall is True:
        return population, score_evolution
    else:
        return population


# some other stuff should we want it for results/analysis
    # # show topology of a random cell in the final population
    # random_cell = np.random.choice(population)
    # random_cell.show_topology(graph_layout='shell')
    #
    # # get some basic distribution statistics
    # age_distribution = [cell.name for cell in population]
    # edge_count_distribution, node_count_distribution = zip(*list(map(lambda x:(len(x[0]), len(x[1])), [cell.get_topology() for cell in population])))
    # network_gene_count_distribution = [len(cell.removable_genes+cell.non_coding_rnas) for cell in population]


def plot_1D_trajectory(score_evolution, obj=0):
    """
    Plots evolutionary trajectory of a specified objective function with generation.

    Parameters:
        score_evolution (dict) - dictionary in which keys are generation numbers and values are lists of selected scores
        obj (int) - index of objective function to be visualized

    Returns:
        ax (axes object)
    """

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


def plot_2D_trajectory(score_evolution, obj=None):
    """
    Plots evolutionary trajectory of two specified objective functions in objective-space.

    Parameters:
        score_evolution (dict) - dictionary in which keys are generation numbers and values are lists of selected scores
        obj (tup) - tuple of indices of objective-coordinates to be visualized

    Returns:
        ax (axes object)
    """

    # check that objective space has at least two dimensions
    dim = len(score_evolution[0][0])
    if dim < 2:
        print('Error: Objective space only has %d dimension(s).' % dim)
        return

    # create axes
    ax = create_subplot_figure(dim=(1, 1), size=(8, 6))[0]

    # if objective space is two dimensional, plot points on objective-space plane
    max_x = 1
    max_y = 1
    for gen, scores in score_evolution.items():

        # plot all points in generation
        coordinates = [coordinate for coordinate in zip(*scores)]
        x, y = coordinates[obj[0]], coordinates[obj[1]]


        color = [gen/len(score_evolution), 0/256, 0/256]
        ax.plot(x, y, '.k', markersize=25, color=color, label='Generation %d' % gen)

        # update maxima for plot axis scaling
        max_x, max_y = max(x + (max_x,)), max(y + (max_y,))

    # format plot
    ax.set_xlim(0, 1.2*max_x)
    ax.set_ylim(0, 1.2*max_y)
    ax.set_xlabel('Objective %d' % obj[0], fontsize=16)
    ax.set_ylabel('Objective %d' % obj[1], fontsize=16)
    ax.set_title('Evolutionary Trajectory', fontsize=16)

    if len(score_evolution) < 10:
        ax.legend(loc=0)

    return ax