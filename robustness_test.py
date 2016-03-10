from main import *
import pickle

generations = 80
population_size = 40
num_mutants = 5  # it's a default argument in fitness.py, not used here

timetorun = generations*population_size*num_mutants*1.2/3600
print('Program will take ~%4.2f hours to run!' % timetorun)

# run simulation
populations = run_simulation(generations=generations, population_size=population_size, mutations_per_division=2, test=robustness_test)

# write results to serialized file
filename = ('adaptation_test_results_Ngens_%d_Ncells_%d.p' % (generations, population_size))
with open(filename, 'wb') as f:
    pickle.dump(populations, f)