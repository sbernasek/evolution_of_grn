__author__ = 'Sebi'

from modules.plotting import *
from modules.signals import *


def get_ss(cell, output=None, dt=0.1):
    """
    Returns steady state level of specified node given zero initial conditions.

    Parameters:
        cell (cell object) - gene regulatory network of interest
        output (int) - index of output node, if not None, output steady state is returned
        dt (float) - time step

    Returns:
        steady_states (np array) - N-dimesnional vector of steady state levels (ordered by re-indexing key)
        output_ss (float) - steady state level of output node
    """
    blank_signal = Signal(name='get_ss', duration=200, dt=dt, signal=None)
    states, _, key = cell.simulate(blank_signal, mode='langevin', retall=True)
    steady_states = states[:, -1]

    if output is None:
        return steady_states
    else:
        output_ss = states[key[output], -1]
        return steady_states, output_ss


def interaction_check(cell, output, input_, plot=False, dt=0.1):
    """
    Determines whether a specified output node is affected by a particular input node.

    Parameters:
        cell (cell object) - gene regulatory network of interest
        output (int) - index of output node
        input_ (int) - index of input node
        plot (bool) - if true, plot output response
        dt (float) - time step

    Returns:
        (bool) - true if cells are connected, false if not
    """

    # procedure computes steady state level for an input level of 1, then computes the cumulative output deviation for a
    # 10 minute interval with input level of 5. If the cumulative deviation exceeds 1e-10, nodes are connected.

    # create elevated input signal
    elevated = Signal(name='elevated', duration=10, dt=dt, signal=None, channels=1)
    elevated.step(magnitude=5)

    # get steady states
    steady_states, output_ss = get_ss(cell, output=output, dt=dt)

    # run simulation
    states, _, key = cell.simulate(elevated, input_node=input_, ic=steady_states, mode='langevin', retall=True)

    # check if output differs
    output_response = states[key[output], :]
    cumulative_output_deviation = sum(abs(output_response-output_ss)*dt)

    if plot is True:
        ax = create_subplot_figure(dim=(1, 1), size=(8, 6))[0]
        ax.plot(elevated.time, states[key[output], :], '-b', label='Response to 5x Step')
        ax.set_ylim(0, 2*max(states[key[output], :]))
        ax.legend(loc=0)
        ax.set_xlabel('Time (min)', fontsize=16)
        ax.set_ylabel('Output', fontsize=16)
        ax.set_title('Input/Output Connection Test', fontsize=16)

    # determine average change, if greater than 5% then assume network is connected
    print(cumulative_output_deviation)

    if cumulative_output_deviation > 1e-10:
        return True
    else:
        return False


def get_fitness_1(cell, mode='langevin', plot=False):
    """
    VERSION: VARIES REPORTER TRANSCRIPTION RATE AND TRACKS DEVIATION FROM STEADY STATE

    Runs dynamic test and evaluates cell's performance.

    Parameters:
        cell (cell object) - cell to be tested
        mode (str) - numerical method used to solve system dynamics, either 'langevin' or 'tau_leaping'
        plot (bool) - if true, plot dynamic simulation

    Returns:
        score (list) - list of objective-space coordinates, namely [cumulative_error, energy_usage]
    """
 
    dt = 0.1
    plateau_count = 3
    plateau_duration = 100
    input_node = 0
    output_node = 1

    # get steady state levels
    steady_states, output_ss = get_ss(cell, output=output_node, dt=dt)

    # BEGIN TEST
    # create sequence of plateaus for disturbance signal
    disturbance = Signal(name='driver', duration=plateau_duration, dt=dt, channels=1)
    disturbance.step(magnitude=1)
    for stage in range(0, plateau_count):
        new_level = np.random.randint(1, 5)
        next_plateau = Signal(name=stage, duration=plateau_duration, dt=dt, channels=1)
        next_plateau.step(magnitude=new_level)
        disturbance = disturbance.merge_signals(next_plateau, shift=True, gap=dt)

    # run simulation
    states, energy_usage, key = cell.simulate(disturbance, input_node=input_node, ic=steady_states, mode=mode, retall=True)
    output = states[key[output_node], :]

    # integrate absolute difference between sensor and its steady state level
    cumulative_error = sum(abs(output-output_ss)*dt)  # compute integrated area using square rule

    # plot dynamics
    if plot is True:
        ax0, ax1, ax2 = create_subplot_figure(dim=(1, 3), size=(24, 6))

        # plot driving disturbance signal
        ax0.plot(disturbance.time, disturbance.signal[0], '--b', linewidth=3)
        ax0.set_ylim(0, max(disturbance.signal[0])+1)
        ax0.set_ylabel('Disturbance Signal', fontsize=16)
        ax0.set_xlabel('Time (min)', fontsize=16)

        # plot output level, output steady state, and output deviation from steady state
        ax1.plot(disturbance.time, output, '-b', label='Output', linewidth=3)
        ax1.plot(disturbance.time, [output_ss for _ in disturbance.time], '--b', label='Output Steady State')
        ax1.plot(disturbance.time, abs(output-output_ss), '-r', label='Deviation from Steady State')
        ax1.set_ylim(0, 1.1*max(output))
        ax1.set_xlabel('Time (min)', fontsize=16)
        ax1.set_ylabel('Output and Deviation', fontsize=16)
        ax1.legend(loc=0)

        # plot cumulative error
        cumulative_deviation = [sum(abs(output[0:i]-output_ss)*dt)/1e3 for i in range(0, len(disturbance.time))]
        ax2.set_ylabel('Cumulative Deviation (thousands)', fontsize=16)
        ax2.set_xlabel('Time (min)', fontsize=16)
        ax2.plot(disturbance.time, cumulative_deviation, '-r', linewidth=3)
        ax2.plot(disturbance.time[-1], cumulative_deviation[-1], '.r', markersize=25, label='Score')
        ax2.set_xlim(0, disturbance.time[-1]+10)
        ax2.legend(loc=2)

    return [cumulative_error, energy_usage]


def get_fitness_2(cell, mode='langevin', plot=False):
    """
    VERSION: VARIES TRANSCRIPTION OF NON-REPORTER PERMANENT GENE AND TRACKS REPORTER DEVIATION FROM STEADY STATE

    Runs dynamic test and evaluates cell's performance.

    Parameters:
        cell (cell object) - cell to be tested
        mode (str) - numerical method used to solve system dynamics, either 'langevin' or 'tau_leaping'
        plot (bool) - if true, plot dynamic simulation

    Returns:
        score (list) - list of objective-space coordinates, namely [cumulative_error, energy_usage]
    """

    dt = 0.1
    plateau_count = 3
    plateau_duration = 100
    input_node = 2
    output_node = 1

    # check to see whether input/output are connected, if not then skip this cell (not necessary, but saves time)
    connected = interaction_check(cell, output=output_node, input_=input_node, dt=dt)
    if connected is False:
        return None, None

    # get steady state levels
    steady_states, output_ss = get_ss(cell, output=output_node, dt=dt)

    # create sequence of plateaus for disturbance signal
    disturbance = Signal(name='driver', duration=plateau_duration, dt=dt, channels=1)
    disturbance.step(magnitude=1)
    for stage in range(0, plateau_count):
        new_level = np.random.randint(1, 5)
        next_plateau = Signal(name=stage, duration=plateau_duration, dt=dt, channels=1)
        next_plateau.step(magnitude=new_level)
        disturbance = disturbance.merge_signals(next_plateau, shift=True, gap=dt)

    # run simulation
    states, energy_usage, key = cell.simulate(disturbance, input_node=input_node, ic=steady_states, mode=mode, retall=True)

    # retrieve output dynamics
    output = states[key[output_node], :]

    # integrate absolute difference between output and its steady state level
    cumulative_error = sum(abs(output-output_ss)*dt)  # integrated area using square rule

    # plot dynamics
    if plot is True:
        ax0, ax1, ax2 = create_subplot_figure(dim=(1, 3), size=(24, 6))

        # plot driving disturbance signal
        ax0.plot(disturbance.time, disturbance.signal[0], '--b', linewidth=3)
        ax0.set_ylim(0, max(disturbance.signal[0])+1)
        ax0.set_ylabel('Disturbance Signal', fontsize=16)
        ax0.set_xlabel('Time (min)', fontsize=16)

        # plot output level, output steady state, and output deviation from steady state
        ax1.plot(disturbance.time, output, '-b', label='Output', linewidth=3)
        ax1.plot(disturbance.time, [output_ss for _ in disturbance.time], '--b', label='Output Steady State')
        ax1.plot(disturbance.time, abs(output-output_ss), '-r', label='Deviation from Steady State')
        ax1.set_ylim(0, 1.1*max(output))
        ax1.set_xlabel('Time (min)', fontsize=16)
        ax1.set_ylabel('Output and Deviation', fontsize=16)
        ax1.legend(loc=0)

        # plot cumulative error
        cumulative_deviation = [sum(abs(output[0:i]-output_ss)*dt)/1e3 for i in range(0, len(disturbance.time))]
        ax2.set_ylabel('Cumulative Deviation (thousands)', fontsize=16)
        ax2.set_xlabel('Time (min)', fontsize=16)
        ax2.plot(disturbance.time, cumulative_deviation, '-r', linewidth=3)
        ax2.plot(disturbance.time[-1], cumulative_deviation[-1], '.r', markersize=25, label='Score')
        ax2.set_xlim(0, disturbance.time[-1]+10)
        ax2.set_ylim(0, 1.2*max(cumulative_deviation))
        ax2.legend(loc=2)

    return [cumulative_error, energy_usage]
