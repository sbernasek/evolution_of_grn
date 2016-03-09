__author__ = 'Sebi'

import scipy.integrate
from modules.plotting import *
from modules.signals import *

"""
TO DO:
"""


def get_fitness(cell, input_node=None, output_node=None, steady_states=None, plot=False, solve_if_stiff=True):
    """
    VERSION: VARIES TRANSCRIPTION OF NON-REPORTER PERMANENT GENE AND TRACKS REPORTER DEVIATION FROM STEADY STATE

    Runs dynamic test and evaluates cell's performance.

    Parameters:
        cell (cell object) - cell to be tested
        input_node (int) - index of node to which disturbance signal is sent
        output_node (int) - index of node from which output signal is read
        steady_states (np array) - steady state values under unit input to input_node
        plot (bool) - if True, plot dynamic simulation
        solve_if_stiff (bool) - if False, discard stiff systems

    Returns:
        score (list) - list of objective-space coordinates, namely [cumulative_error, energy_usage]
    """

    # set plateau count and duration for input signal
    plateau_count = 3
    plateau_duration = 200

    # set default input/output if none specified
    if input_node is None:
        input_node = 2

    if output_node is None:
        output_node = 1

    # get steady state levels, if no nonzero stable steady states are found then return None and move on
    if steady_states is None:
        steady_states = cell.get_steady_states(input_node, input_magnitude=1, ic=None)
        if steady_states is None or steady_states[cell.key[output_node]] < 1:
            return [None, None]

    # # check to see whether input/output are connected, if not then skip this cell
    # connected = cell.interaction_check_numerical(input_node=input_node, output_node=output_node, steady_states=steady_states, plot=False)
    # if connected is False or connected is None:
    #     return [None, None]

    # # create random sequence of plateaus for disturbance signal
    # input_level = 1
    # input_signal = [(0, input_level)]
    # for i in range(1, plateau_count+1):
    #     next_step = np.random.choice([num for num in range(1, 5) if num != input_level])
    #     input_signal.append((i*plateau_duration, next_step))
    #     input_level = next_step

    # create specific sequence of plateaus for disturbance signal
    # input_signal = [(0, 1), (100, 2), (200, 5), (300, 2), (400, 1), (500, 1)]
    input_signal = [(0, 1), (100, 0.5), (200, 5), (300, 2), (400, 1), (500, 1)]

    # run simulation
    times, states, energy = cell.simulate(input_signal, input_node=input_node, ic=steady_states, solve_if_stiff=solve_if_stiff)

    # if simulation failed (returned None), return None
    if sum([item is None for item in [times, states, energy]]) > 0:
        return [None, None]

    # retrieve output dynamics
    output = states[cell.key[output_node], :]

    # integrate absolute difference between output and its steady state level
    cumulative_error = scipy.integrate.cumtrapz(abs(output-steady_states[cell.key[output_node]]), x=times)

    # plot dynamics
    if plot is True:
        ax0, ax1, ax2 = create_subplot_figure(dim=(1, 3), size=(24, 6))

        # plot driving disturbance signal (need to change this)
        t, levels = [i for i in zip(*input_signal)]
        ax0.step(t, levels, '-b', where='post', linewidth=3)
        ax0.set_ylim(0, max(levels)+1)
        ax0.set_ylabel('Input Signal', fontsize=16)
        ax0.set_xlabel('Time (min)', fontsize=16)

        # plot output level, output steady state, and output deviation from steady state
        ax1.plot(times, output, '-b', label='Output', linewidth=3)
        ax1.plot(times, [steady_states[cell.key[output_node]] for _ in times], '--b', label='Output Steady State')
        ax1.plot(times, abs(output-steady_states[cell.key[output_node]]), '-r', label='Deviation from Steady State')
        ax1.set_ylim(0, 1.1*max(output))
        ax1.set_xlabel('Time (min)', fontsize=16)
        ax1.set_ylabel('Output and Deviation', fontsize=16)
        ax1.legend(loc=0)

        # plot cumulative error
        ax2.set_ylabel('Cumulative Deviation', fontsize=16)
        ax2.set_xlabel('Time (min)', fontsize=16)
        ax2.plot(times[1:], cumulative_error, '-r', linewidth=3)
        ax2.plot(times[-1], cumulative_error[-1], '.r', markersize=25, label='Score')
        ax2.set_xlim(0, times[-1]+10)
        ax2.set_ylim(0, 1.2*max(cumulative_error))
        ax2.legend(loc=2)

    return [cumulative_error[-1], energy]
