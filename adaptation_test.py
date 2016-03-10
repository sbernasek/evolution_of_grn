import tabulate
from main import *
import pickle

"""
Run the largest simulation you can... ideally with a number of generations ~2-3x the number of cells. As a 
crude estimate: 

        total time = generations * population_size * 8s 

So 100 generations of 50 cells should take ~11 hours.

Running this cell should create two files:   adaptation_test_results.p   and   adaptation_test.prof 

"""

# define simulation size here
generations = 200
population_size = 100

# run simulation
populations = run_simulation(generations=generations, population_size=population_size, mutations_per_division=2)

# write results to serialized file
filename = ('adaptation_test_results_Ngens_%d_Ncells_%d.p' % (generations, population_size))
with open(filename, 'wb') as f:
    pickle.dump(populations, f)