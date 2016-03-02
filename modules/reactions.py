__author__ = 'Sebi'

import functools
from modules.parameters import *


class Reaction:

    def __init__(self, reactants=None, products=None, rate_constant=1, consumed=None, rxn_type=None, atp_usage=None, cell_type='prokaryote'):
        """
        Class describes a single kinetic pathway.

        Parameters:
            reactants (list) - list of reactant indices
            products (list) - list of product indices
            rate_constant (float) - mass-action reaction rate constant
            consumed (list) - lists which reactants are consumed by reaction
            rxn_type (str) - identifies reaction type
            atp_usage (float) - energetic cost of reaction in activated phosphorous equivalents (e.g. ~P), if none use defaults for specified rxn_type
            cell_type (str) - 'prokaryote' or 'eukaryote', changes rate library
        """
        self.reactants = reactants
        if reactants is None:
            self.reactants = []

        self.products = products
        if products is None:
            self.products = []

        self.consumed = consumed
        if consumed is None:
            self.consumed = []

        self.rate_constant = rate_constant
        self.rxn_type = rxn_type

        self.atp_usage = atp_usage
        if self.atp_usage is None:
            self.atp_usage = sum(atp_requirements[cell_type][rxn_type])

    def get_rate(self, concentrations):
        """
        Compute and return current rate of a given pathway.

        Parameters:
            concentrations (list) - list of reactant species concentrations
        Returns:
            rate (float) - rate of reaction
        """

        activity = 1
        if len(self.reactants) > 0:
            activity = functools.reduce(lambda x, y: x*y, concentrations)
        rate = self.rate_constant * activity
        return rate


class TranscriptionModification:

    def __init__(self, substrate, target, mod_type, dissociation_constant, promotion_strength=1, cooperativity=1):
        """
        Class defines reactions that modify the rates of other reactions.

        Parameters:
            substrate (int) - index of transcription factor substrate
            target (int) - index of mRNA whose transcription is repressed
            mod_type (str) - indicates whether activation or repression kinetics are used
            dissociation_constant (float) - gene binding affinity
            rate_increase (float) - factor by which a promoter increases rate
            cooperativity (float) - hill coefficient for transcription factor binding
        """

        self.substrate = substrate
        self.target = target
        self.dissociation_constant = dissociation_constant
        self.promotion_strength = promotion_strength
        self.cooperativity = cooperativity

        if mod_type not in ['activation', 'repression']:
            print('Error: Transcription factor type not recognized.')
        else:
            self.mod_type = mod_type

    def get_rate_modifier(self, tf):
        """
        Returns transcription rate modification.

        Parameters:
            tf (float) - current level of transcription factor
        Returns:
            modifier (float) - current occupancy for transcription promotion or vacancy for repression
        """

        # get dissociation constant (dna binding affinity) and cooperativity (hill coefficient)
        kd = self.dissociation_constant
        n = self.cooperativity

        if self.mod_type == 'activation':
            sites_occupied = (tf**n) / ((kd**n) + (tf**n))
            return self.promotion_strength * sites_occupied

        elif self.mod_type == 'repression':
            sites_unoccupied = (kd**n) / ((kd**n) + (tf**n))
            return sites_unoccupied
