__author__ = 'Sebi'

import functools
from operator import mul
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
            if rxn_type is not None:
                self.atp_usage = sum(atp_requirements[cell_type][rxn_type])
            else:
                self.atp_usage = None

    @staticmethod
    def from_json(js):

        # create instance
        rxn = Reaction()

        # get each attribute from json dictionary
        rxn.reactants = js['reactants']
        rxn.products = js['products']
        rxn.consumed = js['consumed']
        rxn.rate_constant = js['rate_constant']
        rxn.rxn_type = js['rxn_type']
        rxn.atp_usage = js['atp_usage']
        return rxn

    def to_json(self):
        return {
            # return each attribute
            'reactants': self.reactants,
            'products': self.products,
            'consumed': self.consumed,
            'rate_constant': self.rate_constant,
            'rxn_type': self.rxn_type,
            'atp_usage': self.atp_usage}

    def get_rate(self, concentrations):
        """
        Compute and return current rate of a given pathway.

        Parameters:
            concentrations (list) - list of reactant species concentrations
        Returns:
            rate (float) - rate of reaction
        """
        activity = functools.reduce(mul, concentrations)
        return self.rate_constant * activity


class TranscriptionModification:

    def __init__(self, substrate=None, target=None, mod_type=None, dissociation_constant=dna_binding_affinity, promotion_strength=promoter_strength, cooperativity=hill_coefficient):
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
        self.mod_type = mod_type

    @staticmethod
    def from_json(js):

        # create instance
        mod = TranscriptionModification()

        # get each attribute from json dictionary
        mod.substrate = js['substrate']
        mod.target = js['target']
        mod.dissociation_constant = js['dissociation_constant']
        mod.promotion_strength = js['promotion_strength']
        mod.cooperativity = js['cooperativity']
        mod.mod_type = js['mod_type']

        return mod

    def to_json(self):
        return {
            # return each attribute
            'substrate': self.substrate,
            'target': self.target,
            'dissociation_constant': self.dissociation_constant,
            'promotion_strength': self.promotion_strength,
            'cooperativity': self.cooperativity,
            'mod_type': self.mod_type}

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

            if tf <= 0:
                sites_occupied = 0
            else:
                sites_occupied = (tf**n) / ((kd**n) + (tf**n))
            return self.promotion_strength * sites_occupied

        elif self.mod_type == 'repression':

            if tf <= 0:
                sites_unoccupied = 1
            else:
                sites_unoccupied = (kd**n) / ((kd**n) + (tf**n))
            return sites_unoccupied
