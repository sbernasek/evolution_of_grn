__author__ = 'Sebi'

from main import *
import pickle

generations = 200
population_size = 75

# assign name to results file
results_file_name = 'robustness_' + str(generations) + 'gen' + str(population_size) + 'cells.p'

# run simulation
results = run_simulation(generations=generations, population_size=population_size, mutations_per_division=2, test=robustness_test)

# write results to serialized file
with open(results_file_name, 'wb') as f:
    pickle.dump(results, f)
