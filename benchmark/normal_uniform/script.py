## Imports
import numpy as np
import json
import itertools
from tqdm.auto import tqdm
import sys

sys.path.append('C:/Users/meri2/Documents/3A_MVA/Flowtree')
from flowtree import Distribution, Quadtree, reference_w1_distance


## Parameters
NUMBER_OF_POINTS = [1000, 2500, 5000, 7500, 10000, 15000, 20000, 25000, 30000, 50000]
SUPPORT_SIZE = [100, 250, 500, 750, 1000, 1500, 2000, 5000]
DIMENSION = [1, 2, 5, 10, 20, 50, 100, 200]
OUTFILE = 'C:/Users/meri2/Documents/3A_MVA/Flowtree/benchmark/normal_uniform_0.json'


## Script

# Prepare the loop
estimates, references = {}, {}
iterator = itertools.product(NUMBER_OF_POINTS, SUPPORT_SIZE, DIMENSION)
total_iters = len(NUMBER_OF_POINTS) * len(SUPPORT_SIZE) * len(DIMENSION)

# Loop
for number_of_points, support_size, dimension in tqdm(iterator, total=total_iters):
    # Pass to the next iteration if the support is too big
    if number_of_points < support_size:
        continue
    # Generate points
    points = np.random.normal(size=(number_of_points, dimension)) 
    # Generate distributions
    mu = Distribution.uniform_distribution(number_of_points, support_size)
    nu = Distribution.uniform_distribution(number_of_points, support_size)
    # Set key
    key = str((number_of_points, support_size, dimension))
    # Estimate W1 distance
    estimates[key] = Quadtree(points).compute_w1_distance(mu, nu)
    # Get real W1 distance
    references[key] = reference_w1_distance(points, mu, nu)

# Save
with open(OUTFILE, 'w') as file:
    json.dump([estimates, references], file)