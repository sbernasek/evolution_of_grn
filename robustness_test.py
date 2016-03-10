from main import *
import pickle

generations = 70
population_size = 25
num_mutants = 5  # it's a default argument in fitness.py, not used here

print(generations*population_size*num_mutants*1.2/3600, ' hours')

# run simulation
populations = run_simulation(generations=generations, population_size=population_size, mutations_per_division=2, test=robustness_test)

# write results to serialized file
filename = ('adaptation_test_results_Ngens_%d_Ncells_%d.p' % (generations, population_size))
with open('robustness_test_results.p', 'wb') as f:
    pickle.dump(populations, f)