__author__ = 'Sebi'


# species parameters:
mrna_length = 1500  # nucleotides, ~1000 for e.coli, 1500 for yeast, 3k for drosophila
protein_length = 400  # amino acids, 300 for e.coli, 385 for yeast, ~600 for mammals
n_introns = 4  # roughly 2-8 for eukaryotes, maybe 2 for e.coli and 8 for humans

# no clue
PTR_RATE_CONSTANT = 0.1
CATALYTIC_DEGRADATION_RATE_CONSTANT = 0.1
MODIFICATION_RATE_CONSTANT = 0.1
CATALYTIC_MODIFICATION_RATE_CONSTANT = 0.1

# iffy
basal_transcription_rate = 0.1
dna_binding_affinity = 1e2
hill_coefficient = 1

# settled rates
promoter_strength = 30 * 60 / mrna_length  # mrna/min, based on 30 nt/s
translation_constant = 10 * 60 / protein_length  # proteins/min, based on 10 amino acids per second
mrna_decay_constant = 0.05
protein_decay_constant = 0.025

atp_requirements = {

    'prokaryote': {

        # initiation, elongation, termination
        'transcription': [1, 2*mrna_length, 100],

        # initiation, elongation, termination
        'translation': [3, 4*protein_length, 1],

        # activation/repression (could add translocation cost here)
        'trancriptional_regulation': [5],

        # energy required is re-used for elongation
        'mrna_decay': [0],

        # degradation (range ~0.25-1)
        'protein_decay': [1*protein_length],

        # phosphorylation
        'modification': [1],

        # enzyme-assisted phosphorylation
        'catalytic_modification': [1],

        # miRNA silencing (could address dicer/ago synthesis, as well as decreasing miRNA synthesis cost)
        'miRNA_silencing': [0],

        # assumed similar to protein degradation... I have no idea
        'catalytic_degradation': [1]},


    'eukaryote': {
        # initiation, elongation, termination, nucleosome barrier, splicing
        'transcription': [50, 2*mrna_length, 900, 0.17*mrna_length, 10*n_introns],

        # initiation, elongation, termination
        'translation': [10, 4*protein_length, 1],

        # activation/repression (could add translocation cost here)
        'trancriptional_regulation': [10],

        # energy required is re-used for elongation
        'mrna_decay': [0],

        # degradation (range ~0.25-1)
        'protein_decay': [1*protein_length],

        # phosphorylation
        'modification': [1],

        # enzyme-assisted phosphorylation
        'catalytic_modification': [1],

        # miRNA silencing (could address dicer/ago synthesis, as well as decreasing miRNA synthesis cost)
        'miRNA_silencing': [0],

        # assumed similar to degradation... I have no idea
        'catalytic_degradation': [1]}
}
