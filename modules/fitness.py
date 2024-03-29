__author__ = 'Sebi'

import numpy as np
import scipy.integrate
import copy as copy
from modules.plotting import *
from modules.parameters import plateau_count, plateau_duration

"""
TO DO:

    1. separate out WHY a test failed... need to figure out fraction that are failing and why... make smarter choices

"""


def adaptation_test(cell, input_node=None, output_node=None, steady_states=None, interaction_check=False, input_random=False, plot=False, solve_if_stiff=True, ax=None):
    """
    Runs dynamic test and evaluates cell's performance. This test assesses the cell's ability to reject an input
    signal. An input signal with several plateaus is sent to an input node, and the output node's cumulative
    deviation from steady state is taken to be an inverse measure of its adaptive ability.

    Parameters:
        cell (cell object) - cell to be tested
        input_node (int) - index of node to which disturbance signal is sent
        output_node (int) - index of node from which output signal is read
        steady_states (np array) - steady state values under unit input to input_node
        interaction_check (bool) - if True, perform an interaction check
        input_random (bool) - if True, input consists of a random sequence of plateaus. if False, pre-defined sequence is used.
        plot (bool) - if True, plot dynamic simulation
        solve_if_stiff (bool) - if False, discard stiff systems

    Returns:
        score (list) - list of objective-space coordinates, namely [cumulative_error, energy_usage]
    """

    # set default input/output if none specified
    if input_node is None:
        input_node = 0

    if output_node is None:
        output_node = 2

    # get steady state levels, if no nonzero stable steady states are found then return None and move on
    if steady_states is None:
        steady_states = cell.get_steady_states(input_node, input_magnitude=1, ic=None)
        if steady_states is None or steady_states[cell.key[output_node]] < 0:
            print('couldnt get steady state')
            return [None, None]

    # check to see whether input/output are connected, if not then skip this cell
    if interaction_check is True:
        connected = cell.interaction_check_numerical(input_node=input_node, output_node=output_node, steady_states=steady_states, plot=False)
        if connected is False or connected is None:

            if connected is False:
                print('not connected')
            elif connected is None:
                print('connection test failed')
            return [None, None]

    # if input_random is True, generate a sequence of random plateaus
    if input_random is True:

        # create random sequence of plateaus for disturbance signal
        input_level = 1
        input_signal = [(0, input_level)]
        for i in range(1, plateau_count+1):
            next_step = np.random.choice([num for num in range(1, 6) if num != input_level])
            input_signal.append((i*plateau_duration, next_step))
            input_level = next_step

    else:
        # use specific sequence of plateaus for input signal
        input_signal = [(0, 1), (100, 3), (200, 5), (300, 3), (400, 1), (500, 1)]

    # run simulation
    times, states, energy = cell.simulate(input_signal, input_node=input_node, ic=steady_states, solve_if_stiff=solve_if_stiff)

    # if simulation failed (returned None), return None
    if sum([item is None for item in [times, states, energy]]) > 0:
        print('simulation failed')
        return [None, None]

    # retrieve output dynamics
    output_steady_state = steady_states[cell.key[output_node]]
    output = states[cell.key[output_node], :]

    # integrate absolute difference between output and its steady state level, then normalize by total area under SS
    performance = scipy.integrate.trapz(abs(output/output_steady_state-1), x=times) / times[-1]

    # plot dynamics
    if plot is True:

        # create axes and set tick label size
        if ax is None:
            axes = create_subplot_figure(dim=(1, 3), size=(24, 6))
            ax0, ax1, ax2 = axes

            for ax in axes:
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(16)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(16)

        else:
            ax0, ax1, ax2 = ax

        # set fontsize for axis labels
        font_size = 22

        # plot driving disturbance signal (need to change this)
        t, levels = [i for i in zip(*input_signal)]
        ax0.step(t, levels, '-b', where='post', linewidth=3)
        ax0.set_ylim(0, max(levels)+1)
        ax0.set_ylabel('Input Signal', fontsize=font_size, fontweight='bold')
        ax0.set_xlabel('Time (min)', fontsize=font_size, fontweight='bold')

        # plot normalized output level

        ax1.plot(times, output/output_steady_state, '-b', label='Normalized Output', linewidth=3)
        ax1.fill_between(times, np.ones((len(times))), output/output_steady_state, color='blue', alpha=0.5)
        ax1.set_ylim(0.9*np.min(output/output_steady_state), 1.1*np.max(output/output_steady_state))
        ax1.set_xlim(0, times[-1])
        ax1.set_xlabel('Time (min)', fontsize=font_size, fontweight='bold')
        ax1.set_ylabel('Output / Steady State', fontsize=font_size, fontweight='bold')

        # get cumulative relative error at each point
        cumulative_errors = scipy.integrate.cumtrapz(abs(output/output_steady_state - 1), x=times) / times[1:]

        # plot cumulative error
        ax2.set_ylabel('Adaptation Score', fontsize=font_size, fontweight='bold')
        ax2.set_xlabel('Time (min)', fontsize=font_size, fontweight='bold')
        ax2.plot(times[1:], cumulative_errors, '-r', linewidth=3)
        ax2.plot(times[-1], cumulative_errors[-1], '.r', markersize=25)
        ax2.set_xlim(0, times[-1]+10)
        ax2.set_ylim(0, 1.2*max(cumulative_errors))

        # format plots
        ax0.tick_params(labelsize=18)
        ax1.tick_params(labelsize=18)
        ax2.tick_params(labelsize=18)

    score = [performance, energy]

    return score


def robustness_test(cell, num_mutants=5, input_node=None, output_node=None, steady_states=None, plot=False):
    """
    Runs dynamic test and evaluates cell's performance. This test assesses the cell's ability to reject an input
    signal when its gene regulatory network topology is compromised.

    Parameters:
        cell (cell object) - cell to be tested
        num_mutants (int) - number of mutants used to test robustness
        input_node (int) - index of node to which disturbance signal is sent
        output_node (int) - index of node from which output signal is read
        steady_states (np array) - steady state values under unit input to input_node

    Returns:
        score (list) - list of objective-space coordinates, namely [cumulative_error, energy_usage, robustness]
    """

    # get score for parent cell
    [performance, energy] = adaptation_test(cell, input_node, output_node, steady_states, interaction_check=True, input_random=False)

    # if high level simulation failed, return None
    if performance is None or energy is None:
        return [None, None]

    # initialize mutants
    mutants = []
    for i in range(0, num_mutants):
        mutant = get_null_mutant(cell)
        mutants.append(mutant)

    # create axes labels for mutant plots
    ax = None
    if plot is True:
        ax = create_subplot_figure(dim=(1, 3), size=(24, 6))

        # create proxy artists for legend
        ax0, ax1, ax2 = ax
        ax1.plot([], '-b', label="Output", linewidth=3)
        ax1.plot([], '--b', label="Output Steady State", linewidth=3)
        ax1.plot([], '-r', label="Deviation from Steady State", linewidth=3)
        ax2.plot([], '.r', label="Performance Score", markersize=25)
        ax1.legend(loc=0)
        ax2.legend(loc=2)

    # get scores for all mutants. if mutant is None (no node could be removed), assume scores are also None
    mutant_scores = []
    for mutant in mutants:
        if mutant is None:
            mutant_score = [None, None]
        else:
            mutant_score = adaptation_test(mutant, input_node, output_node, interaction_check=True, input_random=False, ax=ax, plot=plot)
        mutant_scores.append(mutant_score)

    # aggregate scores into a single robustness metric, e.g. robustness = mean(mutant_scores)*(1+num_mutant_fails)
    mutant_performances = [score[0] for score in mutant_scores if score[0] is not None]
    if len(mutant_performances) > 0:
        mean_score = np.mean(mutant_performances)
        num_fails = len([score for score in mutant_scores if score[0] is None])
        robustness = mean_score * (1+num_fails)
    else:
        robustness = None

    # return [performance, energy, robustness]
    return [robustness, energy]


def get_null_mutant(parent):
    """
    Receives a parent cell and returns a null mutant in which a randomly selected gene or modified protein is completely removed, along with all of its dependents.

    Parameters:
        parent (cell object) - parent cell which is to be copied

    Returns:
        child (cell object) - copy of parent cell with one gene or modified protein removed
    """

    # clone parent cell
    child = copy.deepcopy(parent)

    # compute weighted node removal probabilities
    num_coding_genes = (len(child.coding_rnas)-len(child.permanent_genes))
    num_non_coding_genes = len(child.non_coding_rnas)
    num_modified_proteins = len(child.modified_proteins)

    probabilities = [num_coding_genes, num_non_coding_genes, num_modified_proteins]
    if sum(probabilities) == 0:
        return None

    probabilities = [p/float(sum(probabilities)) for p in probabilities]

    # select node type to be removed
    node_type = np.random.choice(['coding', 'non-coding', 'modified'], p=probabilities)

    # if 'coding' was selected, remove a coding gene
    if node_type == 'coding':
        child.remove_coding_gene()

    # if 'non-coding' was selected, remove a non-coding gene
    if node_type == 'non-coding':
        child.remove_non_coding_gene()

    # if 'modified' was selected, remove a modified protein
    if node_type == 'modified':

        # select a reaction that produce a downstream modified protein
        rxns = []
        for rxn in child.reactions:
            if rxn.rxn_type in ['modification', 'catalytic_modification']:
                if rxn.products[0] in child.modified_proteins and rxn.reactants[0] < rxn.products[0]:
                    rxns.append(rxn)
        reaction_selected = np.random.choice(rxns)

        # remove the select reaction and all downstream dependencies
        child.remove_protein_modification(rxn_removed=reaction_selected)

    return child