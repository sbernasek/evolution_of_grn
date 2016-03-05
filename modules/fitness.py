__author__ = 'Sebi'

from modules.plotting import *
from modules.signals import *

"""

"""


def check_stability(cell, output, input_=None, max_dt=1, min_dt=0.1):
    """
    Progressively decreases step size until stable (positive, finite, nonzero, and real for all values) response is
    achieved. If no stable response is achieved by the minimum step size, system is deemed unstable.

    Parameters:
        cell (cell object) - cell undergoing stability test
        output (int) - index of output node
        input_ (int) - index of input node
        max_dt (float) - starting timestep (maximum possible returned)
        min_dt (float) - minimum allowable timestep (minimum possible returned)

    Returns:
        stable (bool) - if True, system exhibited stable response at current time step
        dt (float) - minimum stable time step

    """

    acceptable = False
    dt = max_dt

    while acceptable is False:

        # break loop if more than three attempts to find a stable solution are made
        if dt < min_dt:
            return acceptable, None

        if input_ is None:
            # if no input node is specified, get steady state values in the absence of any driving signal
            input_signal = Signal(name='get_ss', duration=20, dt=dt, signal=None)
            states, _, key = cell.simulate(input_signal, mode='langevin', retall=True)

        else:
            # if input node is specified, get steady state values following unit step input to unit node
            input_signal = Signal(name='get_ss', duration=20, dt=dt, channels=1)
            input_signal.step(1)
            states, _, key = cell.simulate(input_signal, input_node=input_, mode='langevin', retall=True)

        # if states are numerically stable, accept them. if not, reduce timestep
        positive, finite, nonzero, real = False, False, False, False
        if sum([level < 0 for level in states[key[output], :]]) == 0:
            positive = True

        if max([abs(level) for level in states[key[output], :]]) < 1e10:
            finite = True

        if sum(states[key[output], :]) > 1e-100:
            nonzero = True

        if sum(np.isnan(states[key[output], :])) == 0:
            real = True

        if positive is True and finite is True and nonzero is True and real is True:
            acceptable = True
            return acceptable, dt
        else:
            dt /= 10


def get_steady_states(cell, output=None, input_=None, dt=0.1, plot=False):
    """
    Returns steady state level of specified node given zero initial conditions and no disturbance signal.

    Parameters:
        cell (cell object) - gene regulatory network of interest
        output (int) - index of output node, if not None, output steady state is returned
        input (int) - index of input node, if not None, unit step signal is sent to input
        dt (float) - time step used
        plot (bool) - if True, plot prcedure

    Returns:
        steady_states (np array) - N-dimesnional vector of steady state levels (ordered by re-indexing key)
        output_ss (float) - steady state level of output node
    """

    if input_ is None:
        # if no input node is specified, get steady state values in the absence of any driving signal
        input_signal = Signal(name='get_ss', duration=200, dt=dt, signal=None)
        states, _, key = cell.simulate(input_signal, mode='langevin', retall=True)

    else:
        # if input node is specified, get steady state values following unit step input to unit node
        input_signal = Signal(name='get_ss', duration=200, dt=dt, channels=1)
        input_signal.step(1)
        states, _, key = cell.simulate(input_signal, input_node=input_, mode='langevin', retall=True)

    # get steady states
    steady_states = states[:, -1]

    # plot input and output trajectories (optional)
    if plot is True:

        ax = create_subplot_figure(dim=(1, 1), size=(8, 6))[0]

        reverse_key = {new_index: old_index for old_index, new_index in key.items()}
        for state, trajectory in enumerate(states):

            # plot input
            if reverse_key[state] == input_:
                ax.plot(input_signal.time, trajectory, '-b', linewidth=5, label='Input')

            # plot output
            elif reverse_key[state] == output:
                ax.plot(input_signal.time, trajectory, '-r', linewidth=5, label='Output')
                ax.set_ylim(0, 1.5*np.max(trajectory))

            # plot everything else
            else:
                ax.plot(input_signal.time, trajectory, '-k', linewidth=2)

        ax.legend(loc=0)
        ax.set_xlabel('Time (min)', fontsize=16)
        ax.set_ylabel('Species Levels', fontsize=16)
        ax.set_title('Steady State Test', fontsize=16)

    if output is None:
        return steady_states
    else:
        output_ss = states[key[output], -1]
        return steady_states, output_ss


def interaction_check_topographical(cell, input_, output):
    """
        Determines whether input influences output.

        Parameters:
            input_ (int) - input node index
            output (int) - output node index
        """

    # if output is a coded protein, constitutive expression of its corresponding gene will enable downregulating edges
    # to influence output level
    if [output] in [rxn.products for rxn in cell.reactions if rxn.rxn_type == 'translation']:
        gene = [rxn.reactants[0] for rxn in cell.reactions if rxn.rxn_type == 'translation' and output in rxn.products][0]

    # if output is a gene, constitutive expression will enable downregulating edges
    # to influence output level
    if output in (cell.removable_genes + cell.permanent_genes):
        gene = output

    # initialize dependent node queue
    upregulated_dependents = [input_]

    # identify all upregulated downstream nodes, if output gene is amongst them then output is positively upregulated
    for dependent in upregulated_dependents:

        # downstream of output or its corresponding gene is irrelevant
        if dependent == gene or dependent == output:
            return True

        # find all nodes downstream of current dependent
        children = []
        [children.extend(item) for item in [rxn.products for rxn in cell.reactions if dependent in rxn.reactants]]
        children.extend([mod.target for mod in cell.rate_mods if dependent == mod.substrate and mod.mod_type == 'activation'])
        children = list(set(children))

        # if child is not currently included in dependents, add it to the queue
        for child in children:
            if child not in upregulated_dependents:
                upregulated_dependents.append(child)

    # initialize dependent node queue
    dependents = [input_]

    # toggle for whether or not output is activated by input or output is constitutively active
    in_graph, constitutively_active = False, False

    # check if gene is constitutively active
    gene_transcription_rxn = [rxn for rxn in cell.reactions if rxn.rxn_type == 'transcription' and gene in rxn.products][0]
    if gene_transcription_rxn.rate_constant > 0:
        constitutively_active = True

    # identify all downstream nodes by walking through the network
    for dependent in dependents:

        # downstream of output or its corresponding gene is irrelevant
        if dependent == gene or dependent == output:
            in_graph = True
            break

        # find all nodes downstream of current dependent
        children = []
        [children.extend(item) for item in [(rxn.products + rxn.consumed) for rxn in cell.reactions if dependent in rxn.reactants]]
        children.extend([mod.target for mod in cell.rate_mods if dependent == mod.substrate])
        children = list(set(children))

        # if child is not currently included in dependents, add it to the queue
        for child in children:
            if child not in dependents:
                dependents.append(child)

    if in_graph is True and constitutively_active is True:
        return True
    else:
        return False


def interaction_check_numerical(cell, input_, output, plot=False, dt=0.1):
    """
    Determines whether a specified output node is affected by a particular input node.

    Parameters:
        cell (cell object) - gene regulatory network of interest
        input_ (int) - index of input node
        output (int) - index of output node
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
    steady_states, output_ss = get_steady_states(cell, output=output, dt=dt)

    # run simulation
    states, _, key = cell.simulate(elevated, input_node=input_, ic=steady_states, mode='langevin', retall=True)

    # if simulation blows up, return None
    if None in [state for state in states[:, -1]]:
        return None

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

    if cumulative_output_deviation > 1:
        return True
    else:
        return False


def get_fitness_1(cell, input_node=0, output_node=1, mode='langevin', dt=0.1, plot=False):
    """
    VERSION: VARIES REPORTER TRANSCRIPTION RATE AND TRACKS DEVIATION FROM STEADY STATE

    Runs dynamic test and evaluates cell's performance.

    Parameters:
        cell (cell object) - cell to be tested
        input_node (int) - index of node to which disturbance signal is sent
        output_node (int) - index of node from which output signal is read
        mode (str) - numerical method used to solve system dynamics, either 'langevin' or 'tau_leaping'
        dt (float) - time step for numerical solver
        plot (bool) - if true, plot dynamic simulation

    Returns:
        score (list) - list of objective-space coordinates, namely [cumulative_error, energy_usage]
    """
 
    plateau_count = 3
    plateau_duration = 200

    # get steady state levels
    steady_states, output_ss = get_steady_states(cell, output=output_node, input_=input_node)

    # if no stable steady states are found, return None
    if None in steady_states:
        return None, None

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


def get_fitness_2(cell, input_node=None, output_node=None, mode='langevin', dt=None, plot=False):
    """
    VERSION: VARIES TRANSCRIPTION OF NON-REPORTER PERMANENT GENE AND TRACKS REPORTER DEVIATION FROM STEADY STATE

    Runs dynamic test and evaluates cell's performance.

    Parameters:
        cell (cell object) - cell to be tested
        input_node (int) - index of node to which disturbance signal is sent
        output_node (int) - index of node from which output signal is read
        mode (str) - numerical method used to solve system dynamics, either 'langevin' or 'tau_leaping'
        dt (float) - time step for numerical solver, if None then max stable value obtained from steady state simulation
        plot (bool) - if true, plot dynamic simulation

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

    # check to see whether input/output are connected, if not then skip this cell (not necessary, but saves time)
    # similarly, if connected is None (simulation blows up), skip this cell
    connected = interaction_check_topographical(cell, input_=input_node, output=output_node)
    if connected is False or connected is None:
        return None, None

    # check stability and get time step
    stable, dt = check_stability(cell, output_node, input_=input_node)
    if stable is False:
        return None, None

    # get steady state levels, if no stable steady states are found then return None and move on
    steady_states, output_ss = get_steady_states(cell, output=output_node, input_=input_node)
    if steady_states is None:
        return None, None

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
