__author__ = 'sbernasek'

# import package
from main import *

# define simulation type
test = "robustness"

# define simulation size
generations = 200
population_size = 50

# write data to specified directory
results_directory = 'data/'

# run simulation
run_simulation(directory=results_directory, generations=generations, population_size=population_size, mutations_per_division=2, test=test)

# to generate profile - must run at command line
# nohup python -m cProfile -o data/simulation.prof run.py &
