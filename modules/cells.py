__author__ = 'Sebi'

import numpy as np
import networkx as nx
import copy
import scipy.integrate
import scipy.optimize
import warnings
from tabulate import tabulate
from modules.reactions import *
from modules.parameters import *
from modules.plotting import *

warnings.filterwarnings('error')

"""
TO DO:
    1. add new types of input (i.e. direct protein level)
    2. speed up ode solver... maybe cython?
    3. add enzyme column to show_reactions()

Long term:
    1. dimerization + reverse
    2. time delays for regulation (nuclear translocation)
    3. split cell class into network structure and network dynamics classes
    4. compile linear reaction stoichiometries in advance so the get_rate function can be vectorized
    5. why are we selecting such low protein levels?
"""


class Cell:
    """
    Class defines a network of interacting species described by a system of stochastic ODEs.
    """

    def __init__(self, name, removable_genes=0, permanent_genes=0, cell_type='prokaryote'):
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
        self.key = None

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
        self.stoichiometry = None

        # add permanent genes to network
        for _ in range(0, permanent_genes):
            self.add_coding_gene(removable=False)

        # add initial non-permanent gene-protein pairs
        for _ in range(0, removable_genes):
            self.add_coding_gene(removable=True)

        # compile stoiciometric matrix
        self.compile_stoichiometry()

    @staticmethod
    def from_json(js):

        # create instance
        cell = Cell(name=None)

        # get each attribute from json dictionary
        cell.name = js['name']
        cell.cell_type = js['cell_type']
        cell.key = js['key']
        cell.species_count = js['species_count']
        cell.network_dimension = js['network_dimension']
        cell.coding_rnas = js['coding_rnas']
        cell.proteins = js['proteins']
        cell.non_coding_rnas = js['non_coding_rnas']
        cell.modified_proteins = js['modified_proteins']
        cell.removable_genes = js['removable_genes']
        cell.permanent_genes = js['permanent_genes']
        cell.stoichiometry = np.array(js['stoichiometry'])

        # get attributes containing nested classes
        cell.reactions = [Reaction.from_json(x) for x in js['reactions']]
        cell.rate_mods = [TranscriptionModification.from_json(x) for x in js['rate_mods']]

        return cell

    def to_json(self):
        return {
            # return each attribute
            'name': self.name,
            'cell_type': self.cell_type,
            'key': self.key,
            'species_count': self.species_count,
            'network_dimension': self.network_dimension,
            'coding_rnas': self.coding_rnas,
            'proteins': self.proteins,
            'non_coding_rnas': self.non_coding_rnas,
            'modified_proteins': self.modified_proteins,
            'removable_genes': self.removable_genes,
            'permanent_genes': self.permanent_genes,
            'stoichiometry': self.stoichiometry.tolist(),

            # return attributes containing nested classes
            'reactions': [rxn.to_json() for rxn in self.reactions],
            'rate_mods': [mod.to_json() for mod in self.rate_mods]}

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
        mutant.compile_stoichiometry()
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

        # first select whether mutation involves a node, an edge, or a rate constant
        mutation_type = np.random.choice(['node', 'edge', 'constant'], p=mutation_type_probabilities)

        if mutation_type == 'node':
            # add/remove a node from the network

            # get node counts for scaling mutation probabilities
            num_removable_genes = len(self.removable_genes)
            num_non_coding_genes = len(self.non_coding_rnas)

            # set target network size (i.e. approximate node count at which node additional/removal are equally likely)
            target_network_size = 3

            # define possible mutations and corresponding probabilities
            possible_mutations = [
                (self.add_coding_gene, target_network_size),
                (self.add_non_coding_gene, target_network_size),
                (self.remove_coding_gene, max(num_removable_genes-1, 0)),  # subtract one so we never go below 1 gene
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
                (self.add_transcriptional_regulation, num_rnas*2),
                (self.add_post_transcriptional_regulation, num_non_coding_mrna),
                (self.remove_catalytic_degradation, num_catalytic_degradation),
                (self.remove_protein_modification, num_protein_mods),
                (self.remove_transcriptional_regulation, num_transcriptional),
                (self.remove_post_transcriptional_regulation, num_post_transcriptional)]

        elif mutation_type == 'constant':

            # define possible mutations and corresponding probabilities
            possible_mutations = [
                (self.change_rate_constant, len(self.reactions)),
                (self.change_modifier_constant, len(self.rate_mods))]

        else:
            print('Mutation type not recognized.')
            return

        # re-normalize probabilities
        mutations, relative_probabiltiies = zip(*possible_mutations)
        probabilities = list(map((lambda x: x/float(sum(relative_probabiltiies))), relative_probabiltiies))

        # select mutation
        mutation_selected = np.random.choice(mutations, size=1, p=probabilities)

        # apply mutations
        _ = [mutation() for mutation in mutation_selected]

    def change_rate_constant(self):
        """
        Changes a randomly selected reaction rate constant. This includes both rate constants and dissociation/cooperativity
        constants.
        """
        index = np.random.randint(0, len(self.reactions))
        rxn = self.reactions[index]
        rxn.rate_constant *= (np.random.random()*1.5 + 0.5)
        self.reactions[index] = rxn

    def change_modifier_constant(self):
        """
        Changes a randomly selected transcriptional modification constant. This includes promotion strengths,
        dissociation constants, and cooperativity constants.
        """

        # get a random transcriptional modifier
        index = np.random.randint(0, len(self.rate_mods))
        mod = self.rate_mods[index]

        # select the type of constant to be updated
        if mod.mod_type == 'activation':
            constant = np.random.choice(['promotion_strength', 'dissociation_constant', 'cooperativity'], p=[0.4, 0.4, 0.2])
        else:
            constant = np.random.choice(['dissociation_constant', 'cooperativity'], p=[0.5, 0.5])

        # update the selected type of constant by
        if constant == 'promotion_strength':
            mod.promotion_strength *= (np.random.random()*1.5 + 0.5)

        elif constant == 'dissociation_constant':
            mod.dissociation_constant *= (np.random.random()*1.5 + 0.5)

        elif constant == 'cooperativity':
            mod.cooperativity *= (np.random.random()*1.5 + 0.5)

        # add modifier back to network
        self.rate_mods[index] = mod

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

    def delete_dependencies(self, root_proteins, genes_removable=False):
        """
        Removes a reaction from the network along with all protein dependencies.

        Parameters:
            root_proteins (list) - list of proteins to be removed from the network
            genes_removable (bool) - determines whether gene-encoded proteins are eligible for deletion
        """

        # initialize lists of retained, candidate, and deleted nodes
        retained, candidates, deleted = [], [], []

        # define list of root nodes to be coded proteins
        if genes_removable is True:
            retained = [rxn.products[0] for rxn in self.reactions if rxn.rxn_type == 'translation' and rxn.products[0] not in [protein for protein in root_proteins]]
        else:
            retained = [rxn.products[0] for rxn in self.reactions if rxn.rxn_type == 'translation']

        # add those that are not retained to the candidate list
        candidates.extend([protein for protein in root_proteins if protein not in retained])

        # identify all possible candidate nodes for deletion
        for candidate in candidates:

            # add all children of the candidate to the candidate list
            for children in [rxn.products for rxn in self.reactions if candidate in rxn.reactants]:
                for child in children:
                    if child not in candidates and child not in retained:
                        candidates.append(child)

        count = 0
        candidates_old = None
        while len(candidates) != 0:

            # break loop if candidate list has not changed
            if candidates == candidates_old:
                break

            # store current candidate list for comparison on next iteration
            candidates_old = copy.deepcopy(candidates)

            # break loop if in excess of 1000 iterations
            count += 1
            if count > 1e3:
                print('exceeded 1000 iterations')
                break

            # iterate through each candidate
            for candidate in candidates_old:

                # get all of candidate's parents
                parent_sets = [rxn.reactants for rxn in self.reactions if candidate in rxn.products]

                # if candidate does not have any parents in the candidate or retained lists, move to removed
                orphan = True
                for parents in parent_sets:
                    if False not in [parent in retained or parent in candidates for parent in parents]:
                        orphan = False
                        break

                if orphan is True:
                    candidates.remove(candidate)
                    deleted.append(candidate)

                # if any of the candidate's parents are in the retained list, candidate is retained
                secured = False
                for parents in parent_sets:
                    if False not in [parent in retained for parent in parents]:
                        secured = True
                        break

                if secured is True:
                        # move candidate from candidates to retained
                        candidates.remove(candidate)
                        retained.append(candidate)

        # add any remaining candidates to the deleted list
        deleted += candidates

        # delete all dependent nodes and reactions in which they participate
        for protein in deleted:
            self.proteins.remove(protein)
            if protein in self.modified_proteins:
                self.modified_proteins.remove(protein)
            self.reactions = [rxn for rxn in self.reactions if protein not in rxn.reactants and protein not in rxn.products]
            self.rate_mods = [mod for mod in self.rate_mods if protein != mod.substrate]

    def remove_coding_gene(self):
        """
        Removes mRNA + protein pair from cell's internal network, along with all edges attached to it.
        """

        # randomly select a removable coding mRNA and protein pair
        mrna = np.random.choice(self.removable_genes)
        protein = [rxn.products[0] for rxn in self.reactions if rxn.rxn_type == 'translation' and rxn.reactants[0] == mrna][0]

        # remove gene and mRNA from network
        self.removable_genes.remove(mrna)
        self.coding_rnas.remove(mrna)

        # remove all reactions amd transcriptional rate modifiers involving mrna
        self.rate_mods = [mod for mod in self.rate_mods if mrna != mod.target]
        self.reactions = [rxn for rxn in self.reactions if mrna not in rxn.reactants and mrna not in rxn.products]

        # remove protein and all dependencies
        self.delete_dependencies([protein], genes_removable=True)

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
            tf = int(tf)
        if gene is None:
            gene = np.random.choice(self.coding_rnas + self.non_coding_rnas)
            gene = int(gene)
        if mod_type is None:
            mod_type = np.random.choice(['activation', 'repression'])
            mod_type = str(mod_type)

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

    def add_post_transcriptional_regulation(self, target=None):
        """
        Add a post transcriptional regulatory interaction between a non-coding mRNA and another randomly chosen mRNA.

        Parameter:
            targer (int) - index of target transcript
        """

        # select a non-coding rna
        micro_rna = np.random.choice(self.non_coding_rnas)
        micro_rna = int(micro_rna)

        # select any target rna if none were provided
        if target is None:
            target = np.random.choice(self.non_coding_rnas + self.coding_rnas)
            target = int(target)

        # create post-transcriptional modification reaction
        rxn = Reaction(reactants=[micro_rna, target], products=None, rate_constant=PTR_RATE_CONSTANT, consumed=[micro_rna, target], rxn_type='miRNA_silencing', cell_type=self.cell_type)
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

    def add_catalytic_degradation(self, degraded=None, enzyme=None):
        """
        Randomly select two proteins and add a protein-protein catalytic degradation (partial degradation) reaction.

        Parameters:
            degraded (int) - index of degraded protein
            enzyme (int) - index of catalytic protein that promotes degradation of p1
        """
        # randomly choose two proteins, or use the two assigned

        if len(self.proteins) < 2:
            return

        if degraded is None and enzyme is None:
            degraded, enzyme = np.random.choice(self.proteins, size=2, replace=False)
            degraded = int(degraded)
            enzyme = int(enzyme)

        elif degraded is None and enzyme is not None:
            degraded = np.random.choice(self.proteins, size=1, replace=False)
            degraded = int(degraded)
        elif degraded is not None and enzyme is None:
            enzyme = np.random.choice(self.proteins, size=1, replace=False)
            enzyme = int(enzyme)

        # create reaction
        rxn = Reaction(reactants=[degraded, enzyme], products=None, rate_constant=CATALYTIC_DEGRADATION_RATE_CONSTANT, consumed=[degraded], rxn_type='catalytic_degradation', cell_type=self.cell_type)
        self.reactions.append(rxn)

    def remove_catalytic_degradation(self):
        """
        Randomly select and remove a protein-protein catalytic degradation reaction.
        """
        rxn_choices = [rxn for rxn in self.reactions if rxn.rxn_type == 'catalytic_degradation']
        rxn_selected = np.random.choice(rxn_choices)
        self.reactions.remove(rxn_selected)

    def add_protein_modification(self, substrate=None, enzyme=None, product_specified=None, reversibility_bias=1):
        """
        Add a reaction in which a protein is converted to a new protein species. The reaction may or may not be
        assisted by another protein.

        Parameters:
            substrate (int) - index of specific protein to be modified
            enzyme (int) - index of enzyme that catalyzes modification
            product_specified (int) - index of modified protein product
            reversibility_bias (float) - probability that reverse modification is selected vs further modification. by
            default, multiple sequential modifications are not accessible.
        """

        # while loop is used to ensure that a unique reaction is added (no duplicate reactant/product pairs allowed)
        duplicate_rxn = True

        count = 0
        while duplicate_rxn is True:
            count += 1
            if count > 1e2:
                break

            # randomly choose one protein substrate to be modified
            if substrate is None:
                substrate = np.random.choice(self.proteins)
                substrate = int(substrate)

            # if target product is not specified, first decide whether substrate is itself a modified protein. if it is,
            # randomly decide whether to re-form its parent or create new modified form. otherwise, create new modified form.
            product = product_specified
            if product is None:

                if substrate in self.modified_proteins and np.random.random() < reversibility_bias:

                    # set product to one of modified protein's parent forms
                    parents = [rxn.reactants[0] for rxn in self.reactions if substrate in rxn.products]
                    product = np.random.choice(parents)
                    product = int(product)

                else:

                    # create new modified protein
                    product = self.species_count
                    self.proteins.append(product)
                    self.modified_proteins.append(product)
                    self.species_count += 1

                    # add new protein's decay reaction
                    decay_rxn = Reaction(reactants=[product], rate_constant=protein_decay_constant, consumed=[product], rxn_type='protein_decay', cell_type=self.cell_type)
                    self.reactions.append(decay_rxn)

            # decide whether protein modification is enzyme assisted (requires at least 2 proteins total)

            # if enzyme is specified, assume catalytic mechanism
            if enzyme is not None:
                probabilities = [0, 1]

            # if at least two protein species exist, randomly choose a mechanism
            elif len(self.proteins) > 3:
                probabilities = [0.5, 0.5]

            # if only one protein exists, assume non-catalytic mechanism
            else:
                probabilities = [1, 0]

            # select mechanism
            rxn_type = str(np.random.choice(['modification', 'catalytic_modification'], p=probabilities))

            if rxn_type == 'modification':

                # get rate constant and define reactants
                rate_constant = MODIFICATION_RATE_CONSTANT
                reactants = [substrate]

            else:

                # randomly choose one protein as an enzyme to catalyze modification of the substrate (exclude autocatalysis)
                if enzyme is None:
                    enzyme = np.random.choice([protein for protein in self.proteins if protein != substrate and protein != product])
                    enzyme = int(enzyme)

                # get rate constant and define reactants
                rate_constant = CATALYTIC_DEGRADATION_RATE_CONSTANT
                reactants = [substrate, enzyme]

            # if selected substrate, enzyme, and product lead to a unique reaction, break loop. otherwise, repeat it
            existing_reactions = [rxn for rxn in self.reactions if rxn.rxn_type in ['modification', 'catalytic_modification']]
            if len(existing_reactions) == 0:
                duplicate_rxn = False
            else:
                duplicate_rxn = False
                for rxn in existing_reactions:
                    if rxn.reactants == reactants and rxn.products == [product]:
                        duplicate_rxn = True
                        break

        # add modification reaction
        rxn = Reaction(reactants=reactants, products=[product], rate_constant=rate_constant, consumed=[substrate], rxn_type=rxn_type, cell_type=self.cell_type)
        self.reactions.append(rxn)

    def remove_protein_modification(self, rxn_removed=None):
        """
        Randomly select and remove a protein modification reaction. If the modified protein product is not formed
        by any other reactions, remove it and all its dependencies.

        Parameters:
            rxn_removed (reaction object) - specific reaction to be removed
        """

        # randomly select a protein modification reaction for removal
        rxn_choices = [rxn for rxn in self.reactions if rxn.rxn_type in ['modification', 'catalytic_modification']]

        # if no modifications exist, skip this mutation
        if len(rxn_choices) == 0:
            return

        # if no reaction is specified, select one at random
        if rxn_removed is None:
            rxn_removed = np.random.choice(rxn_choices)

        # remove reaction
        self.reactions.remove(rxn_removed)

        # get list of products from deleted reaction
        reaction_products = [product for product in rxn_removed.products]

        # remove products and all dependent proteins
        self.delete_dependencies(reaction_products, genes_removable=False)

    def compile_stoichiometry(self):
        """
        Iterates through list of M reactions involving N species and constructs NxM stoichiometric matrix.
        """

        # reindex nodes to sequential order
        self.reindex_nodes()

        # initialize stoichiometric matrix as zeros
        self.stoichiometry = np.zeros((self.network_dimension, len(self.reactions)))

        # insert coefficients for reactants and products
        for rxn_num, rxn in enumerate(self.reactions):
            # consumed reactants receive negative coefficient
            for reactant in rxn.consumed:
                self.stoichiometry[self.key[reactant], rxn_num] = -1

            # products receive positive coefficient
            for product in rxn.products:
                self.stoichiometry[self.key[product], rxn_num] = 1

    def get_rate_vector(self, concentrations):
        """
        Computes rate of each reaction.

        TODO: if this can be vectorized we would save a ton of time

        Parameters:
            concentrations (np array) - Nx1 vector of current species concentrations
        Returns:
            rxn_rates (np array) - array of M current reaction rates
        """

        # iterate through all rate mods, storing their rate enhancement/repression terms for each gene
        mods = {target: {'activation': [], 'repression': []} for target in (self.non_coding_rnas + self.coding_rnas)}
        for mod in self.rate_mods:
            tf = self.key[mod.substrate]
            tf_impact = mod.get_rate_modifier(concentrations[tf])
            mods[mod.target][mod.mod_type].append(tf_impact)

        # initialize rate vector
        rxn_rates = np.empty((len(self.reactions)))

        # compute rate of each reaction
        for i, rxn in enumerate(self.reactions):

            # compute transcription rates
            if rxn.rxn_type == 'transcription':
                mrna = rxn.products[0]
                rate = 1
                rate *= max([rxn.rate_constant] + mods[mrna]['activation'])
                if len(mods[mrna]['repression']) > 0:
                   rate *= functools.reduce(mul, mods[mrna]['repression'])

            # compute mass action rates
            elif rxn.reactants != []:
                reactant_concentrations = [concentrations[self.key[reactant]] for reactant in rxn.reactants]
                rate = rxn.get_rate(reactant_concentrations)

            # compute degradation rates
            else:
                rate = rxn.rate_constant

            # add reaction rate to rate vector
            rxn_rates[i] = rate

        return rxn_rates

    def reindex_nodes(self):
        """
        Compile dictionary for reindexing all nodes into sequential order (genes, proteins, modified proteins) used in
        states array.

        """
        active_species = self.coding_rnas + self.non_coding_rnas + self.proteins
        self.network_dimension = len(active_species)
        reindexing_key = {old_index: new_index for new_index, old_index in enumerate(active_species)}
        self.key = reindexing_key

    def get_topology(self):
        """
        Aggregates and returns gene regulatory network topology as an edge list.

        Returns:
            edge_list (list) -  gene regulatory network topology composed of (regulatory_gene, target_gene, regulation_type) tuples
            node_labels (dict) - dictionary of node labels in which keys are new indices and values are node types
            node_key (dict) - dictionary mapping old to new indices in which keys are old indices and values are new
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
                    node_key[rxn.products[0]] = len(self.permanent_genes) + i

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
            if rxn.rxn_type == 'catalytic_modification':
                edge = (node_key[rxn.reactants[0]], node_key[rxn.products[0]], rxn.rxn_type) # ignore enzyme for visualization
                edge_list.append(edge)

        return edge_list, node_labels, node_key

    def show_topology(self, graph_layout='shell', input_node=None, output_node=None, retall=False):
        """
        Generates networkx visualization of network topology.

        Parameters:
            graph_layout (string) - method used to arrange network nodes in space
            input_node (int) - index of node to which input signal is sent
            output_node (int) - index of node from which output is retrieved
            retall (bool) - if True, return axes
        """

        # get network topology
        edge_list, node_labels, node_key = self.get_topology()
        node_key[None] = None

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

        # add nodes
        for node, node_type in node_labels.items():
            g.add_node(node, attr_dict={'node_type': node_type})

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

        # sort nodes into four types (different colors on graph)
        permanent_genes = [node for node, node_type in node_labels.items() if node_type == 'permanent gene']
        removable_genes = [node for node, node_type in node_labels.items() if node_type == 'removable gene']
        non_coding_genes = [node for node, node_type in node_labels.items() if node_type == 'non-coding gene']
        modified_proteins = [node for node, node_type in node_labels.items() if node_type == 'modified protein']

        # draw green permanent genes
        if len(permanent_genes) > 0:
            nx.draw_networkx_nodes(g, pos, nodelist=permanent_genes, node_color='g', node_size=node_size, alpha=node_alpha)

        # draw cyan coding genes
        if len(removable_genes) > 0:
            nx.draw_networkx_nodes(g, pos, nodelist=removable_genes, node_color='c', node_size=node_size, alpha=node_alpha)

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
            if edge[2] in ['activation', 'modification', 'catalytic_modification']:
                up_regulating_edges[(edge[0], edge[1])] = edge[2]
            else:
                down_regulating_edges[(edge[0], edge[1])] = edge[2]

        # draw green up regulating edges
        nx.draw_networkx_edges(g, pos, edgelist=up_regulating_edges.keys(), width=3, alpha=edge_alpha, edge_color='g')
        nx.draw_networkx_edge_labels(g, pos, edge_labels=up_regulating_edges, label_pos=edge_text_pos, font_size=10, fontweight='bold')

        # draw red down regulating edges
        nx.draw_networkx_edges(g, pos, edgelist=down_regulating_edges.keys(), width=3, alpha=edge_alpha, edge_color='r')
        nx.draw_networkx_edge_labels(g, pos, edge_labels=down_regulating_edges, label_pos=edge_text_pos, font_size=10, fontweight='bold')

        # draw node labels with gene numbers
        for node, node_type in node_labels.items():

            if node == node_key[input_node] and input_node is not None:
                node_labels[node] = 'INPUT' + '\n' + str(node)
            elif node == node_key[output_node] and output_node is not None:
                node_labels[node] = 'OUTPUT' + '\n' + str(node)
            else:
                node_labels[node] = node_type + '\n' + str(node)

        nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=node_text_size, fontweight='bold', color='k', ha='center')

        # get current figure and axes
        fig = plt.gcf()
        fig.set_size_inches(15, 15)
        _ = plt.axis('off')
        ax = plt.gca()

        # add all catalytic interactions
        for rxn in self.reactions:
            if rxn.rxn_type == 'catalytic_modification':

                x_substrate, y_substrate = pos[node_key[rxn.reactants[0]]]
                x_enzyme, y_enzyme = pos[node_key[rxn.reactants[1]]]
                x_product, y_product = pos[node_key[rxn.products[0]]]

                x = x_enzyme
                y = y_enzyme
                U = (x_product - x_substrate)/2 + x_substrate - x_enzyme
                V = (y_product - y_substrate)/2 + y_substrate - y_enzyme
                ax.arrow(x, y, U, V, length_includes_head=True, head_length=0.05, head_width=0.025, fc='g', ec='g', linewidth=3)

        if retall is True:
            return ax

    def simulate(self, input_signal, input_node=0, ic=None, solve_if_stiff=True):
        """
        Simulates dynamic system using Scipy's ode object.

        Parameters:
            input_signal (list) - values for input signal, supplied as a list of (start_time, input_magnitude) tuples
            input_node (int) - index of mRNA whose transcription rate is impacted by input signal
            ic (np array) - vector of initial conditions for each species, assumed zero if omitted
            solve_if_stiff (bool) - if True, when dopri5 solver fails, revert to VODE (slow, but handles stiff ODEs)

        Returns:
            times (np array) - time points corresponding to state values (1 by t)
            states (np array) - matrix of state values at each time point (N by t)
            energy_usage (float) - total energy used
        """

        # compile reindexing key, stoichiometric matrix, and atp requirements
        self.compile_stoichiometry()
        atp_per_rxn = [rxn.atp_usage for rxn in self.reactions]

        # initialize solution list
        solution = []

        # if no initial condition is provided, assume all states are initially zero
        initial_states = ic
        if ic is None:
            initial_states = np.zeros(self.network_dimension)

        # initialize ODE solver
        integration_length = input_signal[-1][0]

        # first try dopri5 for non-stiff systems:
        try:
            solver = scipy.integrate.ode(self.get_species_rates).set_integrator('dopri5', method='bdf', nsteps=1000)
            solout = lambda t, y: solution.append([t] + [y_i for y_i in y])
            solver.set_solout(solout)
            solver.set_initial_value(initial_states, 0).set_f_params(input_signal, input_node)
            solver.integrate(integration_length)

        # if dopri5 fails, use vode (slow, but more likely to work for stiff systems). if vode fails, return None
        except UserWarning:

            if solve_if_stiff is True:
                try:
                    solver = scipy.integrate.ode(self.get_species_rates).set_integrator('vode', method='bdf')
                    solver.set_initial_value(initial_states, 0).set_f_params(input_signal, input_node)
                    while solver.successful() and solver.t < integration_length:
                        solver.integrate(integration_length, step=True)
                        solution.append([solver.t] + [y_i for y_i in solver.y])
                except:
                    print('stiff equation solver failed')
                    return None, None, None
            else:
                return None, None, None

        # get solution and sort by time (sometimes solver produces erroneous time points)
        solution = np.array(solution).T
        sort_indices = np.argsort(solution[0, :])
        solution = solution[:, sort_indices]
        times = solution[0, :]
        states = solution[1:, :]

        # update energy usage
        energy_usage = self.get_energy_usage(times, states, input_signal, input_node, atp_per_rxn)

        return times, states, energy_usage

    def get_species_rates(self, t, state, input_signal=None, input_node=None):
        """
        Computes net rate of change of each chemical species. This serves as the "derivative" function for the ODE solver.

        Parameters:
            t (float) - current time
            state (np array) - array of current state values
            input_signal (list) - values for input signal, supplied as a list of (start_time, input_magnitude) tuples
            input_node (int) - index of gene to which activation signal is sent

        Returns:
            species_rates (np array) - net rate of change of each species (N x 1)
        """

        rxn_rates = self.get_rxn_rates(t, state, input_signal, input_node)
        species_rates = np.dot(self.stoichiometry, rxn_rates)

        return species_rates

    def get_rxn_rates(self, t, state, input_signal, input_node):
        """
        Computes rate of each reaction. This serves as the basis for the energy usage calculations.

        Parameters:
            t (float) - current time
            state (np array) - array of current state values
            input_signal (list) - values for input signal, supplied as a list of (start_time, input_magnitude) tuples
            input_node (int) - index of mRNA whose transcription rate is impacted by input signal

        Returns:
            rxn_rates (np array) - rate of each reaction (M x 1)
        """

        # compute current reactions rates
        rxn_rates = self.get_rate_vector(state)

        # determine current input value
        if input_signal is not None:
            for time, magnitude in input_signal:
                if t >= time:
                    current_input = magnitude
                else:
                    break

            # apply input as activation of target gene
            disturbed_rxn = [j for j, rxn in enumerate(self.reactions)
                             if rxn.rxn_type == 'transcription' and rxn.products[0] == input_node]
            rxn_rates[disturbed_rxn[0]] = current_input

        return rxn_rates

    def get_energy_usage(self, times, states, input_signal, input_node, atp_per_rxn):
        """

        TODO: This is horrendous! calling reaction rates at every timestep is a terrible idea

        Computes total atp usage for the current simulation.

        Parameters:
            times (np array) - time points corresponding to state values (1 by t)
            states (np array) - matrix of state values at each time point (N by t)
            input_signal (list) - values for input signal, supplied as a list of (start_time, input_magnitude) tuples
            input_node (int) - index of mRNA whose transcription rate is impacted by input signal
            atp_per_rxn (list) - list of atp requirements per unit flux through each reaction pathway (M x 1)

        Returns:
            energy_usage (float) - total number of ATPs required
        """

        # compute reaction rates
        rxn_rates = np.empty((len(self.reactions), len(times)))
        for i, t in enumerate(times):
            rxn_rates[:, i] = self.get_rxn_rates(t, states[:, i], input_signal, input_node)

        # compute reaction extents
        rxn_extents = scipy.integrate.trapz(rxn_rates, x=times)

        # compute total atp usage
        energy_usage = np.dot(rxn_extents, atp_per_rxn)

        return energy_usage

    def get_steady_states(self, input_node=None, input_magnitude=1, ic=None):
        """
        Computes steady state of system from specified initial condition.

        Parameters:
            input_node (int) - index of input node, if None then zero input assumed
            input_magnitude (float) - magnitude of input signal, if None then zero input assumed
            ic (np array) - initial conditions from which steady state is found
        """

        # compile stoichiometric matrix
        self.compile_stoichiometry()

        # if no initial condition is provided, assume all states are initially zero
        initial_states = ic
        if ic is None:
            initial_states = np.zeros(self.network_dimension)

        # use fsolve to find roots of ODE system:
        while True:
            if initial_states[0] > 150:
                return None
            try:
                steady_states = scipy.optimize.fsolve(self.get_ss_species_rates, initial_states, args=(input_node, input_magnitude))
                break
            except RuntimeWarning:
                initial_states += 50

        return steady_states

    def plot_steady_states(self, input_node=None, input_magnitude=1, output_node=None, ic=None):
        """
        Plots steady state trajectory of system from specified initial condition.

        Parameters:
            input_node (int) - index of input node, if None then zero input assumed
            input_magnitude (float) - magnitude of input signal, if None then zero input assumed
            output_node (int) - index of output node to be plotted
            ic (np array) - initial conditions from which steady state is found
        """

        # if no initial condition is provided, assume all states are initially zero
        initial_states = ic
        if ic is None:
            initial_states = np.zeros(self.network_dimension)

        # if plot is True and output_node is specified, plot steady state procedure
        step_input = [(0, input_magnitude), (100, input_magnitude)]
        times, states, _ = self.simulate(step_input, input_node=input_node, ic=initial_states)

        if times is not None and states is not None:
            output_states = states[self.key[output_node], :]
            ax = create_subplot_figure(dim=(1, 1), size=(8, 6))[0]
            ax.plot(times, output_states, linewidth=5)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1.5*max(output_states))
            ax.set_xlabel('Time (min)', fontsize=16)
            ax.set_ylabel('Output Level', fontsize=16)
            ax.set_title('Steady State Test', fontsize=16)

    def get_ss_species_rates(self, states, input_node, input_magnitude):
        """
        Computes net rate of change of each chemical species. This serves as the "derivative" function for the steady
         state solver.

        Parameters:
            states (np array) - array of current state values
            input_node (int) - index of node to which input signal is sent
            input_signal (float) - magnitude of input signal

        Returns:
            species_rates (np array) - net rate of change of each species (N x 1)
        """

        # compute current reactions rates
        rxn_rates = self.get_rate_vector(states)

        # apply input signal as activation of target gene
        if input_node is not None and input_magnitude is not None:
            disturbed_rxn = [j for j, rxn in enumerate(self.reactions)
                             if rxn.rxn_type == 'transcription' and rxn.products[0] == input_node]
            rxn_rates[disturbed_rxn[0]] = input_magnitude

        # compute species rates
        species_rates = np.dot(self.stoichiometry, rxn_rates)

        return species_rates

    def interaction_check_numerical(self, input_node, output_node, steady_states=None, plot=False):
        """
        Determines whether input influences output by checking whether output deviates from steady state upon step
        change to input.

        Parameters:
            input_node (int) - input node index
            output_node (int) - output node index
            steady_states (np array) - array of steady state values to be used as initial conditions
            plot (bool) - if True, plot procedure

        Returns:
            connected (bool) - if True, output depends upon input
        """

        # if no steady states were provided, get them
        if steady_states is None:
            steady_states = self.get_steady_states(input_node=input_node, input_magnitude=1, ic=None)
            if steady_states is None:
                return None

        # get baseline output level
        output_ss_baseline = steady_states[self.key[output_node]]

        # if output steady state is zero under constant input, return False
        if output_ss_baseline == 0:
            return False

        # apply new step change to input and simulate response
        input_signal = [(0, 2), (10, 2)]
        times, states, _ = self.simulate(input_signal, input_node=input_node, ic=steady_states)

        # if solver failed, return None so simulation is thrown out
        if states is None:
            return None

        # get peak output deviation from initial steady state value
        output_states = states[self.key[output_node], :]
        max_deviation = max([abs(op - output_ss_baseline) for op in output_states]) / output_ss_baseline

        # plot input and output trajectories (optional)
        if plot is True:

            ax = create_subplot_figure(dim=(1, 1), size=(8, 6))[0]
            ax.plot(times, output_states, '-r', linewidth=5, label='Output')
            ax.legend(loc=0)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 1.5*max(output_states))
            ax.set_xlabel('Time (min)', fontsize=16)
            ax.set_ylabel('Output Level', fontsize=16)
            ax.set_title('Interaction Test', fontsize=16)

        if max_deviation > 0.01:
            return True
        else:
            return False

    def show_reactions(self, interactions_only=True, grn_indices=False):
        """
        Pretty-print table of all reactions within the cell.

        Parameters:
            interactions_only (bool) - if True, only include interactions between different genes and proteins
            grn_indices (bool) - if True, display reactants and products in terms of their gene numbers
        """

        # if grn_indices is True, get model to grn key
        if grn_indices is True:
            _, _, key = self.get_topology()

        # create table of reactions
        rxn_table = []
        for rxn in self.reactions:

            # if interactions_only is True, skip basic reactions
            if interactions_only is True and rxn.rxn_type in ['transcription', 'mrna_decay', 'translation', 'protein_decay']:
                continue

            # sort species into reactants, enzymes, and products
            if rxn.rxn_type in ['catalytic_degradation', 'catalytic_modification']:
                reactants = [rxn.reactants[0]]
                enzymes = [rxn.reactants[1]]
            else:
                reactants = rxn.reactants
                enzymes = []
            products = rxn.products

            # if grn_indices is True, store reactant/product list in terms of gene numbers, otherwise leave as is
            if grn_indices is True:
                reactants = [key[reactant] for reactant in reactants]
                enzymes = [key[enzyme] for enzyme in enzymes]
                products = [key[product] for product in products]

            # append reaction to table
            rxn_table.append([rxn.rxn_type, reactants, enzymes, products])

        # create table of transcriptional regulators
        mod_table = []
        for mod in self.rate_mods:
            tf = mod.substrate
            target = mod.target

            # if grn_indices is True, store reactant/product list in terms of gene numbers, otherwise leave as is
            if grn_indices is True:
                tf = key[tf]
                target = key[target]

            # append reaction to table
            mod_table.append([mod.mod_type, target, tf])

        # print tables
        print(tabulate(rxn_table, headers=["Reaction Type", "Reactants", "Enzymes", "Products"]))
        print('\n')
        print(tabulate(mod_table, headers=["Regulation Type", "Target Gene", "Transcription Factor"]))