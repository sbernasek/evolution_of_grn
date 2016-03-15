__author__ = 'Sebi'

from modules.pareto import *
from modules.fitness import *
from modules.analysis import *
import json
import os


"""

add equation used to calculate robustness score


TO DO:
    1. write outer wrapper for file saving, if sim fails delete file
    2. change to ATP consumption rate
    3. try using optimizer to sole for steady state and see whether bifurcation occurs.. maybe avoid running non-adaptive sims
    4. constrain mutations to connected graph
    5. stop running rate calcs for energy usage calculations
    6. clean up analysis code.. comments, tick marks, etc
    7. write master plot formatter in plotting.py
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

def run_simulation(directory=None, generations=10, population_size=20, mutations_per_division=2, test='adaptation'):
    """
    Runs full simulation procedure.

    Parameters:
        directory (str) - path to directory in which results are stored as a subdirectory of json files
        generations (int) - number of growth/selection stages
        population_size (int) - number of cells per generation
        mutations_per_division (int) - number of mutations per cell cycle
        test (str) - name of test function used to score each cell

    Returns:
        populations (list) - dictionary of all cells selected throughout procedure. keys are generation numbers, while
        values are dictionaries of (cell: scores) entries.
    """

    # if results are to be saved, initialize a new directory corresponding to this simulation
    if directory is not None:
        version = 0
        while True:
            new_path = directory + test + '_' + str(generations) + 'x' + str(population_size) + '_v' + str(version)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
                break
            version += 1
    else:
        print('No directory specified.')
        return

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
        if test == 'adaptation':
            scoring_function = adaptation_test
        elif test == 'robustness':
            scoring_function = robustness_test
        new_scores = evaluate(scoring_function, new_cells, input_node, output_node, ss_dict=ss_dict)

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
        current_generation = {i: {'cell': cell.to_json(), 'scores': score} for i, (cell, score) in enumerate(score_dict.items())}

        # if directory was specified, write current population with corresponding scores to json
        with open(new_path + '/' + str(gen) + '.json', mode='w', encoding='utf-8') as f:
            json.dump(current_generation, f)

