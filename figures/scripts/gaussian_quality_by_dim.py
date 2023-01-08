## Imports
from numpy.random import RandomState

import matplotlib.pyplot as plt
from tqdm.auto import trange
from time import perf_counter
import json

from core.flowtree import Quadtree, reference_w1_distance
from figures.utils import gaussian_distributions


## Params
Nb_points =    [100, 500, 1000] + [1000, 5000, 10000] + [10000]
Support_size = [10,  10,  10  ] + [100,  100,  100  ] + [400  ]
Dims = [1, 2, 3, 5, 7] + [10, 20, 30] + [50, 75, 100]
Iters = 250
FigureName = 'gaussian_quality_by_dim'
RS = RandomState(0)

## Script
RATIO = {}
TIMES = {}

for N, s in zip(Nb_points, Support_size):
    key = f"{N = } & {s = }"
    print(key)
    RATIO[key] = []
    TIMES[key] = [[], []]

    for dim in Dims:
        acc_ratio = 0
        acc_time = [0, 0]
        for _ in trange(Iters, desc=f"{dim = }"):

            X, mus = gaussian_distributions(N, dim, [s, s], RS)
            t0 = perf_counter()
            estimate = Quadtree(X).compute_w1_distance(*mus, 2)
            t1 = perf_counter()
            reference = reference_w1_distance(X, *mus, 2)
            t2 = perf_counter()

            acc_ratio += estimate / reference
            acc_time[0] += t1 - t0
            acc_time[1] += t2 - t1

        RATIO[key].append(acc_ratio / Iters)
        TIMES[key][0].append(acc_time[0]/Iters)
        TIMES[key][1].append(acc_time[1]/Iters)

    with open(f"figures/data/{FigureName}.json", 'w') as file:
        json.dump({
            'RATIO' : RATIO, 
            'TIMES' : TIMES
        }, file)


## Plot
fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=200)

for N, s in zip(Nb_points, Support_size):
    key = f"{N = } & {s = }"
    ax.plot(Dims, RATIO[key], label=f"{s = } & {N = }")

ax.set_ylabel('Ratio of estimate to real W1')
ax.set_xlabel('Dimension')
ax.set_xticks(Dims)
ax.set_xticklabels(Dims)
ax.set_title('Dependance of estimate\'s quality on dimension (gaussian model)')
plt.legend()
plt.savefig(f"figures/imgs/{FigureName}.png")
plt.show()