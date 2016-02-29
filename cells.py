__author__ = 'Sebi'

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import copy
from reactions import *
from parameters import *

"""
TO DO:
    1. Could add irreversible dimerization
    2. Could add reversibility of phosphorylation, as well as binding kinetics
    3. could allow rate constants to mutate
    4. could add time delay for transcriptional regulators
"""


class Cell:
    """
    Class defines a network of interacting species described by a system of stochastic ODEs.
    """

    def __init__(self, name, initial_genes, permanent_genes=0, cell_type='prokaryote'):
        """
        Parameters:
            name (int) - cell index
            initial_genes (int) - number of gene-protein pairs in virgin cell
            permanent_genes (int) - number of gene-protein pairs that can never be removed by mutation
            cell_type (str) - 'prokaryote' or 'eukaryote' is used to define reaction ATP requirements
        """
        # assign cell index as name and get mutation probabilities
        self.name = name
        self.cell_type = cell_type

        # initialize cell species
        self.species_count = 0  # this is the total number of unique chemical species in this cell's genetic lineage
        self.network_dimension = 0
        self.coding_rnas = []
        self.proteins = []
        self.non_coding_rnas = []
        self.modified_proteins = []
        self.removable_genes = []
        self.permanent_genes = []

        # initialize cell reactions
        self.reactions = []
        self.rate_mods = []

        # initialize stoichiometric matrix
        self.stoichiometry = np.empty(())

        # add permanent genes to network
        for _ in range(0, permanent_genes):
            self.add_coding_gene(removable=False)

        # add initial non-permanent gene-protein pairs
        for _ in range(0, initial_genes):
            self.add_coding_gene(removable=True)

    def divide(self, num_mutations):
        """
        Returns original cell along with mutant sister cell.

        Parameters:
            num_mutations (int) - number of mutations to be performed
        Returns:
            original_cell (cell object) - unchanged copy of selected cell
            mutant_cell (cell object) - mutant sister of selected cell
        """
        # produce mutant offspring
        mutant = self.run_mutations(num_mutations)
        return self, mutant

    def run_mutations(self, num_mutations):
        """
        Create a clone then implement desired number of mutations upon it.

        Parameters:
            num_mutations (int) - number of mutations to be performed
        """

        # clone parent cell
        mutant = copy.deepcopy(self)
        mutant.name = self.name + 1

        # perform desired number of mutations
        mutation_count = 0
        while mutation_count < num_mutations:
            mutation_count += 1

            # perform single mutation
            mutant.mutate()

        return mutant

    def mutate(self):
        """
        Select and implement a single mutation. Procedure determines the probability of each type of mutation by
        constructing a list of (mutation, relative-probability) tuples, normalizing the probabilities, then selecting
        and implementing a single mutation. Mutation probabilities are weighted by the abundance of the relevant
        nodes/edges within the cell's regulatory network.
        """
        # determine probability of each type of mutation by construction list of (mutation, relative-probability) tuples

        # first select whether mutation involves a node or an edge (1:1 odds)
        mutation_type = np.random.choice(['node', 'edge'], size=1, p=[0.05, 0.95])

        if mutation_type == 'node':
            # add/remove a node from the network

            # get node counts for scaling mutation probabilities
            num_coding_genes = len(self.removable_genes)
            num_non_coding_genes = len(self.non_coding_rnas)

            # set target network size (i.e. approximate node count at which node additional/removal are equally likely)
            target_network_size = 3

            # define possible mutations and corresponding probabilities
            possible_mutations = [
                (self.add_coding_gene, target_network_size),
                (self.add_non_coding_gene, target_network_size),
                (self.remove_coding_gene, num_coding_genes-1),  # subtract one so we never go below 1 gene
                (self.remove_non_coding_gene, num_non_coding_genes)]

        elif mutation_type == 'edge':
            # add/remove a regulatory element

            # get node counts for scaling mutation probabilities
            num_proteins = len(self.proteins)
            num_rnas = len(self.coding_rnas + self.non_coding_rnas)
            num_non_coding_mrna = len(self.non_coding_rnas)

            # get edge counts for scaling mutation probabilities
            num_protein_mods = len(self.modified_proteins)
            num_transcriptional = len(self.rate_mods)
            num_post_transcriptional = len([rxn for rxn in self.reactions if rxn.rxn_type == 'miRNA_silencing'])
            num_catalytic_degradation = len([rxn for rxn in self.reactions if rxn.rxn_type == 'catalytic_degradation'])

            # define possible mutations and corresponding probabilities
            possible_mutations = [
                (self.add_catalytic_degradation, num_proteins),
                (self.add_protein_modification, num_proteins),
                (self.add_transcriptional_regulation, num_rnas),
                (self.add_post_transcriptional_regulation, num_non_coding_mrna),
                (self.remove_catalytic_degradation, num_catalytic_degradation),
                (self.remove_protein_modification, num_protein_mods),
                (self.remove_transcriptional_regulation, num_transcriptional),
                (self.remove_post_transcriptional_regulation, num_post_transcriptional)]

        else:
            print('Mutation type not recognized.')
            return

        # re-normalize probabilities
        mutations, relative_probabiltiies = zip(*possible_mutations)
        probabilities = list(map((lambda x: x/sum(relative_probabiltiies)), relative_probabiltiies))

        # select mutation
        mutation_selected = np.random.choice(mutations, size=1, p=probabilities)

        # apply mutations
        _ = [mutation() for mutation in mutation_selected]

    def add_coding_gene(self, removable=True):
        """
        Adds mRNA + protein pair to cell's internal network.

        Parameters:
            removable (bool) - if true, gene is not permanent and should be added to removable list
        """

        # get indices for new species
        mrna, protein = self.species_count, self.species_count + 1

        # add new mRNA and protein pair to network
        self.coding_rnas.append(mrna)  # all coding mrnas
        self.proteins.append(protein)  # all coded proteins

        # add to permanent or removable gene lists
        if removable is True:
            self.removable_genes.append(mrna)
        else:
            self.permanent_genes.append(mrna)

        # increment species count
        self.species_count += 2

        # create reaction objects for transcription, translation, mRNA decay, and protein decay
        transcription = Reaction(products=[mrna], rate_constant=basal_transcription_rate, rxn_type='transcription', cell_type=self.cell_type)
        translation = Reaction(reactants=[mrna], products=[protein], rate_constant=translation_constant, rxn_type='translation', cell_type=self.cell_type)
        mrna_decay = Reaction(reactants=[mrna], rate_constant=mrna_decay_constant, consumed=[mrna], rxn_type='mrna_decay', cell_type=self.cell_type)
        protein_decay = Reaction(reactants=[protein], rate_constant=protein_decay_constant, consumed=[protein], rxn_type='protein_decay', cell_type=self.cell_type)

        # add transcription, translation, mRNA and protein decay reactions to reaction dictionary
        self.reactions.append(transcription)
        self.reactions.append(mrna_decay)
        self.reactions.append(translation)
        self.reactions.append(protein_decay)

    def add_non_coding_gene(self):
        """
        Adds non-coding mRNA to cell's internal network. No protein is created.
        """
        # get indices for new species
        mrna = self.species_count

        # add new mRNA and protein pair to network
        self.non_coding_rnas.append(mrna)
        self.species_count += 1

        # create reaction objects for transcription, translation, mRNA decay, and protein decay
        transcription = Reaction(products=[mrna], rate_constant=basal_transcription_rate, rxn_type='transcription', cell_type=self.cell_type)
        mrna_decay = Reaction(reactants=[mrna], rate_constant=mrna_decay_constant, consumed=[mrna], rxn_type='mrna_decay', cell_type=self.cell_type)

        # add transcription, translation, mRNA and protein decay reactions to reaction list
        self.reactions.append(transcription)
        self.reactions.append(mrna_decay)

    def remove_isomers(self, protein):
        """
        Remove modified forms of a selected protein along with all corresponding reactions. Note that this method is
        circular, iterating through each isomer and its downstream isomers.

        Parameters:
            protein (int) - index of gene-coded protein whose isomers are to be removed
        """

        # get all downstream isomers
        isomers = [rxn.products[0] for rxn in self.reactions if rxn.rxn_type == 'modification' and protein in rxn.reactants]

        if len(isomers) == 0:
            # if no isomers are present, end routine
            return
        else:
            # if isomers are present, delete each one and all its dependents, then iteratively remove any more isomers
            for isomer in isomers:

                # remove isomer
                self.proteins.remove(isomer)
                self.modified_proteins.remove(isomer)

                # remove all reactions and rate-modifiers downstream of isomer, unless it's another isomerization
                self.reactions = [rxn for rxn in self.reactions if (isomer not in rxn.reactants or rxn.rxn_type == 'modification')]
                self.rate_mods = [mod for mod in self.rate_mods if isomer != mod.substrate]

                # remove any more isomers (iterative)
                self.remove_isomers(isomer)

                # remove modification reaction that produced this isomer
                self.reactions = [rxn for rxn in self.reactions if isomer not in rxn.products]

    def remove_coding_gene(self):
        """
        Removes mRNA + protein pair from cell's internal network, along with all edges attached to it.
        """

        # randomly select a removable coding mRNA and protein pair
        rna_index = np.random.choice(self.removable_genes)
        protein_index = [rxn.products[0] for rxn in self.reactions if rxn.rxn_type == 'translation' and rxn.reactants[0]
                         == rna_index][0]

        # remove mRNA and protein species from network
        self.removable_genes.remove(rna_index)
        self.coding_rnas.remove(rna_index)
        self.proteins.remove(protein_index)

        # remove all related reactions
        self.reactions = [rxn for rxn in self.reactions if rna_index not in rxn.reactants and rna_index not in rxn.products]  # remove all reactions involving mrna
        self.reactions = [rxn for rxn in self.reactions if protein_index not in rxn.products]  # remove translation
        self.reactions = [rxn for rxn in self.reactions if protein_index not in rxn.reactants or rxn.rxn_type == 'modification']  # remove all reactions in which protein is a reactant, except modifications

        # remove all protein isomers
        self.remove_isomers(protein_index)

        # remove all related transcriptional rate modifiers
        self.rate_mods = [mod for mod in self.rate_mods if rna_index != mod.target and protein_index != mod.substrate]

    def remove_non_coding_gene(self):
        """
        Removes non-coding mRNA from cell's internal network, along with all edges attached to it.
        """

        # randomly select a coding mRNA and protein pair
        rna = np.random.choice(self.non_coding_rnas)

        # remove mRNA and protein species from network
        self.non_coding_rnas.remove(rna)

        # remove all related reactions
        self.reactions = [rxn for rxn in self.reactions if rna not in rxn.reactants and rna not in rxn.products]

        # remove all related transcriptional rate modifiers
        self.rate_mods = [mod for mod in self.rate_mods if rna != mod.target]

    def add_transcriptional_regulation(self, tf=None, gene=None, mod_type=None):
        """
        Randomly select a transcription factor (protein) and an RNA, then create a transcription rate modifier.

        Parameters:
            tf (int) - index of a transcription factor, if None then a protein is randomly selected
            gene (int) - index of the mRNA undergoing regulation, if None then a random gene is selected
            mod_type (str) - type of transcriptional regulation, either 'activation' or 'repression'
        """

        if tf is None:
            tf = np.random.choice(self.proteins)
        if gene is None:
            gene = np.random.choice(self.coding_rnas + self.non_coding_rnas)
        if mod_type is None:
            mod_type = np.random.choice(['activation', 'repression'])

        # create rate modifier
        mod = TranscriptionModification(substrate=tf, target=gene, mod_type=mod_type,
                                        dissociation_constant=dna_binding_affinity,
                                        promotion_strength=promoter_strength,
                                        cooperativity=hill_coefficient)
        self.rate_mods.append(mod)

    def remove_transcriptional_regulation(self):
        """
        Randomly select and delete a transcriptional rate modifier.
        """
        selected_mod = np.random.choice(self.rate_mods)
        self.rate_mods.remove(selected_mod)

    def add_post_transcriptional_regulation(self):
        """
        Add a post transcriptional regulatory interaction between a non-coding mRNA and another randomly chosen mRNA.
        """

        # select a non-coding rna
        rna1 = np.random.choice(self.non_coding_rnas)

        # select any target rna
        rna2 = np.random.choice(self.non_coding_rnas + self.coding_rnas)

        # create post-transcriptional modification reaction
        rxn = Reaction(reactants=[rna1, rna2], products=None, rate_constant=PTR_RATE_CONSTANT, consumed=[rna1, rna2], rxn_type='miRNA_silencing', cell_type=self.cell_type)
        self.reactions.append(rxn)

    def remove_post_transcriptional_regulation(self):
        """
        Randomly select and remove a post-transcriptional regulatory interaction.
        """
        rxn_choices = [rxn for rxn in self.reactions if rxn.rxn_type == 'miRNA_silencing']
        rxn_selected = np.random.choice(rxn_choices)
        self.reactions.remove(rxn_selected)

    # def add_dimerization(self):
    #     """
    #     Randomly select two proteins and add a protein-protein dimerization (joint degradation) reaction.
    #     """
    #     # randomly choose two proteins
    #     if len(self.proteins) > 2:
    #         p1, p2 = np.random.choice(self.proteins, size=2, replace=False)
    #
    #         # create reaction
    #         rxn = Reaction(reactants=[p1, p2], products=None, rate_constant=DIMERIZATION_RATE_CONSTANT, consumed=[p1, p2], rxn_type='dimerization')
    #         self.reactions.append(rxn)
    #
    # def remove_dimerization(self):
    #     """
    #     Randomly select and remove a protein-protein dimerization reaction.
    #     """
    #     rxn_choices = [rxn for rxn in self.reactions if rxn.rxn_type == 'dimerization']
    #     rxn_selected = np.random.choice(rxn_choices)
    #     self.reactions.remove(rxn_selected)

    def add_catalytic_degradation(self, p1=None, p2=None):
        """
        Randomly select two proteins and add a protein-protein catalytic degradation (partial degradation) reaction.

        Parameters:
            p1 (int) - index of degraded protein
            p2 (int) - index of catalytic protein that promotes degradation of p1
        """
        # randomly choose two proteins, or use the two assigned

        if len(self.proteins) < 2:
            return

        if p1 is None and p2 is None:
            p1, p2 = np.random.choice(self.proteins, size=2, replace=False)

        elif p1 is None and p2 is not None:
            p1 = np.random.choice(self.proteins, size=1, replace=False)

        elif p1 is not None and p2 is None:
            p2 = np.random.choice(self.proteins, size=1, replace=False)

        # create reaction
        rxn = Reaction(reactants=[p1, p2], products=None, rate_constant=CATALYTIC_DEGRADATION_RATE_CONSTANT, consumed=[p1], rxn_type='catalytic_degradation', cell_type=self.cell_type)
        self.reactions.append(rxn)

    def remove_catalytic_degradation(self):
        """
        Randomly select and remove a protein-protein catalytic degradation reaction.
        """
        rxn_choices = [rxn for rxn in self.reactions if rxn.rxn_type == 'catalytic_degradation']
        rxn_selected = np.random.choice(rxn_choices)
        self.reactions.remove(rxn_selected)

    def add_protein_modification(self, substrate=None):
        """
        Randomly select a protein and add a reaction in which it is converted to a new protein species via first order
        kinetics.

        Parameters:
            substrate (int) - optional index of specific protein to be modified

        """
        # randomly choose one protein
        if substrate is None:
            protein1 = np.random.choice(self.proteins)
        else:
            protein1 = substrate

        # create new protein with decay reaction
        protein2 = self.species_count
        self.proteins.append(protein2)
        self.modified_proteins.append(protein2)
        self.species_count += 1
        protein2_decay = Reaction(reactants=[protein2], rate_constant=protein_decay_constant, consumed=[protein2], rxn_type='protein_decay', cell_type=self.cell_type)
        self.reactions.append(protein2_decay)

        # create modification reaction
        rxn = Reaction(reactants=[protein1], products=[protein2], rate_constant=MODIFICATION_RATE_CONSTANT, consumed=[protein1], rxn_type='modification', cell_type=self.cell_type)
        self.reactions.append(rxn)

    def remove_protein_modification(self):
        """
        Randomly select and remove a protein modification reaction, along with the corresponding modified protein and
        all its dependencies.
        """
        # remove modification reaction
        rxn_choices = [rxn for rxn in self.reactions if rxn.rxn_type == 'modification']
        rxn_selected = np.random.choice(rxn_choices)
        self.reactions.remove(rxn_selected)

        # remove modified form of protein
        modified_protein = rxn_selected.products[0]
        self.proteins.remove(modified_protein)
        self.modified_proteins.remove(modified_protein)

        # remove all modified-protein dependent reactions.
        self.reactions = [rxn for rxn in self.reactions if modified_protein not in rxn.reactants or rxn.rxn_type == 'modification']  # remove all reactions in which protein is a reactant, except modifications
        self.rate_mods = [mod for mod in self.rate_mods if modified_protein != mod.substrate]

        # remove all protein isomers and their dependencies
        self.remove_isomers(modified_protein)

    def compile_stoichiometry(self, key):
        """
        Iterates through list of M reactions involving N species and constructs NxM stoichiometry matrix.

        Parameters:
            key (dict) - dictionary in which keys are the legacy indices and values are sequential indices
        """
        # initialize stoichiometric matrix as zeros
        self.stoichiometry = np.zeros((self.network_dimension, len(self.reactions)))

        # insert coefficients for reactants and products
        for rxn_num, rxn in enumerate(self.reactions):
            # consumed reactants receive negative coefficient
            for reactant in rxn.consumed:
                self.stoichiometry[key[reactant], rxn_num] = -1

            # products receive positive coefficient
            for product in rxn.products:
                self.stoichiometry[key[product], rxn_num] = 1

    def get_rate_vector(self, concentrations, key):
        """
        Computes rate of each reaction.

        Parameters:
            concentrations (np array) - Nx1 vector of current species concentrations
            key (dict) - dictionary in which keys are the legacy indices and values are sequential indices
        Returns:
            rxn_rates (list) - list of M current reaction rates
        """
        rxn_rates = []
        for rxn in self.reactions:

            # compute transcription rates
            if len(rxn.reactants) == 0:
                rate = rxn.get_rate(rxn.reactants)

                # check for and apply any transcriptional rate modifiers
                if rxn.rxn_type == 'transcription':

                    activation_strength = [1]
                    for mod in self.rate_mods:
                        if key[mod.target] == key[rxn.products[0]]:

                            # get transcription factor concentration
                            tf = key[mod.substrate]

                            # compute total transcription rate as max(activation_strengths)*product(repression_strengths)
                            tf_impact = mod.get_rate_modifier(concentrations[tf])
                            if mod.mod_type == 'activation':
                                activation_strength.append(tf_impact)
                            else:
                                rate *= tf_impact

                    # apply largest promoter effect
                    if len(activation_strength) > 0:
                        rate *= max(activation_strength)

            else:
                # compute rates for other reaction types
                reactants = np.array([key[reactant] for reactant in rxn.reactants])
                rate = rxn.get_rate(concentrations[reactants])

            # add reaction rate to rxn rate vector
            rxn_rates.append(rate)

        return rxn_rates

    def reindex_nodes(self):
        """
        Compile dictionary for reindexing all surviving nodes into sequential order.

        Returns:
            reindexing_key (dict) - dictionary in which keys are old indices and values are the new sequential indices
        """
        active_species = self.coding_rnas + self.non_coding_rnas + self.proteins
        self.network_dimension = len(active_species)
        reindexing_key = {old_index: new_index for new_index, old_index in enumerate(active_species)}
        return reindexing_key

    def simulate(self, disturbances, input_node=0, ic=None, mode='tau_leaping', retall=False):
        """
        Simulates dynamic system.

        Parameters:
            disturbances (signal object) - signal describing external perturbation of system states
            input_node (int) - index of mRNA whose transcription rate is impacted by disturbance
            ic (np array) - vector of initial conditions for each species, assumed zero if omitted
            mode (str) - indicates whether 'tau_leaping' or 'langevin' method is used to solve system ODEs
            retall (bool) - if true, return the energy usage and reindexing key

        Returns:
            states (np array) - N by t matrix of controlled variable states at each time point
            energy_usage (float) - total energy used
            reindexing_key (dict) - dictionary in which keys are old indices and values are the new sequential indices
        """

        # compile dictionary for reindexing nodes
        reindexing_key = self.reindex_nodes()

        # compile stoichiometric matrix and heat of reaction vector
        self.compile_stoichiometry(reindexing_key)
        atp_usage_vector = [rxn.atp_usage for rxn in self.reactions]

        # initialize states array and total energy usage
        states = np.empty((self.network_dimension, len(disturbances.time)))
        energy_usage = 0

        # if no initial condition is provided, assume all states are initially zero
        states[:, 0] = ic
        if ic is None:
            states[:, 0] = np.zeros(self.network_dimension)

        # begin dynamic simulation
        dt = disturbances.dt
        for i, t in enumerate(disturbances.time[:-1]):

            # determine reaction rates
            rxn_rates = self.get_rate_vector(states[:, i], reindexing_key)

            # apply disturbances as rate-multiplier of target gene's transcription
            if disturbances.signal is not None:
                disturbance_impact = disturbances.signal[0][i]
                disturbed_rxn = [j for j, rxn in enumerate(self.reactions)
                                 if rxn.rxn_type == 'transcription' and rxn.products[0] == input_node]
                rxn_rates[disturbed_rxn[0]] *= disturbance_impact

            # determine reaction extents
            if mode == 'tau_leaping':
                rxn_extents = [np.random.poisson(rate*dt) for rate in rxn_rates]
            elif mode == 'langevin':
                rxn_extents = [rate*dt for rate in rxn_rates]
            else:
                print('Error: Solution method not recognized')
                break

            # determine species extents
            species_extents = np.dot(self.stoichiometry, rxn_extents)

            # if concentration is going to be negative, adjust extent such that level drops to zero
            for species, extent in enumerate(species_extents):
                if extent < 0 and abs(extent) > states[species, i]:
                    species_extents[species] = states[species, i]

            # update energy usage
            energy_usage += np.dot(rxn_extents, atp_usage_vector)

            # update species concentrations
            states[:, i+1] = states[:, i] + species_extents

        if retall is True:
            return states, energy_usage, reindexing_key
        else:
            return states

    def get_topology(self):
        """
        Aggregates and returns gene regulatory network topology as an edge list.

        Returns:
            edge_list (list) -  gene regulatory network topology composed of (regulatory_gene, target_gene, regulation_type) tuples
            node_labels (dict) - dictionary of node labels in which keys are node indices and values are node types
        """

        # generate (node_from, node_to, edge_type) tuples
        edge_list = []

        # reindex species used as nodes in gene-regulatory topology
        node_key, node_labels = {}, {}

        # reindex permanent genes
        for h, gene in enumerate(self.permanent_genes):
            node_key[gene] = h
            node_labels[h] = 'permanent gene'
            # find and add corresponding protein to same index
            for rxn in self.reactions:
                if rxn.rxn_type == 'translation' and rxn.reactants[0] == gene:
                    node_key[rxn.products[0]] = h

        # reindex removable genes
        for i, gene in enumerate(self.removable_genes):
            node_key[gene] = len(self.permanent_genes) + i
            node_labels[len(self.permanent_genes) + i] = 'removable gene'
            # find and add corresponding protein to same index
            for rxn in self.reactions:
                if rxn.rxn_type == 'translation' and rxn.reactants[0] == gene:
                    node_key[rxn.products[0]] = i

        # reindex unique non-coding-rnas
        for j, gene in enumerate(self.non_coding_rnas):
            node_key[gene] = len(self.permanent_genes) + len(self.removable_genes) + j
            node_labels[node_key[gene]] = 'non-coding gene'

        # reindex modified proteins
        for k, protein in enumerate(self.modified_proteins):
            node_key[protein] = len(self.permanent_genes) + len(self.removable_genes) + len(self.non_coding_rnas) + k
            node_labels[node_key[protein]] = 'modified protein'

        # get transcriptional regulation edges
        for mod in self.rate_mods:
            edge = (node_key[mod.substrate], node_key[mod.target], mod.mod_type)
            edge_list.append(edge)

        # get other regulatory edges
        for rxn in self.reactions:
            if rxn.rxn_type == 'miRNA_silencing':
                edge = (node_key[rxn.reactants[0]], node_key[rxn.reactants[1]], rxn.rxn_type)
                edge_list.append(edge)
            if rxn.rxn_type == 'catalytic_degradation':
                edge = (node_key[rxn.reactants[1]], node_key[rxn.reactants[0]], rxn.rxn_type)
                edge_list.append(edge)
            if rxn.rxn_type == 'modification':
                edge = (node_key[rxn.reactants[0]], node_key[rxn.products[0]], rxn.rxn_type)
                edge_list.append(edge)

        return edge_list, node_labels

    def show_topology(self, graph_layout='shell'):
        """
        Generates networkx visualization of network topology.

        Parameters:
            graph_layout (string) - method used to arrange network nodes in space
        """

        # get network topology
        edge_list, node_labels = self.get_topology()

        if len(edge_list) == 0:
            print('Network has no edges.')
            return

        # display options
        node_size = 10000
        node_alpha = 1.0
        node_text_size = 10
        edge_alpha = 1
        edge_text_pos = 0.4

        # create networkx graph
        plt.figure()
        g = nx.DiGraph()

        # add edges
        for edge in edge_list:
            g.add_edge(edge[0], edge[1])

        # select graph layout scheme
        if graph_layout == 'spring':
            pos = nx.spring_layout(g)
        elif graph_layout == 'spectral':
            pos = nx.spectral_layout(g)
        elif graph_layout == 'random':
            pos = nx.random_layout(g)
        else:
            pos = nx.shell_layout(g)

        # TEMP - this is a little pointless but will keep it for now
        # retain only nodes that appear within the regulatory network (i.e. have connections)
        node_labels = {node: node_type for node, node_type in node_labels.items()
                       if node in ([edge[0] for edge in edge_list] + [edge[1] for edge in edge_list])}

        # sort nodes into four types (different colors on graph)
        permanent_genes = [node for node, node_type in node_labels.items() if node_type == 'permanent gene']
        removable_genes = [node for node, node_type in node_labels.items() if node_type == 'removable gene']
        non_coding_genes = [node for node, node_type in node_labels.items() if node_type == 'non-coding gene']
        modified_proteins = [node for node, node_type in node_labels.items() if node_type == 'modified protein']

        # draw green permanent genes
        if len(permanent_genes) > 0:
            nx.draw_networkx_nodes(g, pos, nodelist=permanent_genes, node_color='g', node_size=node_size, alpha=node_alpha)

        # draw black coding genes
        if len(removable_genes) > 0:
            nx.draw_networkx_nodes(g, pos, nodelist=removable_genes, node_color='k', node_size=node_size, alpha=node_alpha)

        # draw blue non-coding genes (miRNAs)
        if len(non_coding_genes) > 0:
            nx.draw_networkx_nodes(g, pos, nodelist=non_coding_genes, node_color='b', node_size=node_size, alpha=node_alpha)

        # draw magenta modified proteins
        if len(modified_proteins) > 0:
            nx.draw_networkx_nodes(g, pos, nodelist=modified_proteins, node_color='m', node_size=node_size, alpha=node_alpha)

        # sort edge_list into dictionaries of up-regulating and down-regulating interactions, where (from, to) tuples
        # are keys and values are edge_type strings
        up_regulating_edges, down_regulating_edges = {}, {}
        for edge in edge_list:
            if edge[2] in ['activation', 'modification']:
                up_regulating_edges[(edge[0], edge[1])] = edge[2]
            else:
                down_regulating_edges[(edge[0], edge[1])] = edge[2]

        # draw green up regulating edges
        nx.draw_networkx_edges(g, pos, edgelist=up_regulating_edges.keys(), width=3, alpha=edge_alpha, edge_color='g')
        nx.draw_networkx_edge_labels(g, pos, edge_labels=up_regulating_edges, label_pos=edge_text_pos, font_size=10, fontweight='bold')

        # draw red down regulating edges
        nx.draw_networkx_edges(g, pos, edgelist=down_regulating_edges.keys(), width=3, alpha=edge_alpha, edge_color='r')
        nx.draw_networkx_edge_labels(g, pos, edge_labels=down_regulating_edges, label_pos=edge_text_pos, font_size=10, fontweight='bold')

        # # draw node labels
        # nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=node_text_size, fontweight='bold', color='k')

        fig = plt.gcf()
        fig.set_size_inches(15, 15)
        _ = plt.axis('off')