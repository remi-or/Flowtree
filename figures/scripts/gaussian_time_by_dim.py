## Imports
from numpy.random import RandomState
from numpy import ndarray as Array

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
from tqdm.auto import trange
from time import perf_counter
import json
import ot

from core.flowtree import Quadtree, reference_w1_distance
from figures.utils import gaussian_distributions

## Functions
def find_min_dist(points: Array) -> float:
    return np.min(
        ot.dist(points, metric='euclidean', p=2)
    )

## Params
Nb_points =    [100, + 1000, 10000]
Support_size = [10, 100, 400]
Dims = [1, 2, 3, 5, 7] + [10, 20, 30] + [50, 75, 100]
Iters = 250
FigureName = 'gaussian_time_by_dim'
RS = RandomState(0)

## Script
TIMES = {}
MIN_DIST = {}

for N, s in zip(Nb_points, Support_size):
    key = f"{N = } & {s = }"
    print(key)
    TIMES[key] = [[], []]
    MIN_DIST[key] = []

    for dim in Dims:
        acc_time = [0, 0]
        acc_min_dist = 0
        for _ in trange(Iters, desc=f"{dim = }"):

            X, mus = gaussian_distributions(N, dim, [s, s], RS)
            t0 = perf_counter()
            Quadtree(X).compute_w1_distance(*mus, 2)
            t1 = perf_counter()
            reference_w1_distance(X, *mus, 2, False)
            t2 = perf_counter()
            dist_matrix = ot.dist(X, metric='euclidean')
            dist_matrix += 10**9 * np.eye(dist_matrix.shape[0])

            acc_time[0] += t1 - t0
            acc_time[1] += t2 - t1
            acc_min_dist += np.min(dist_matrix)

        TIMES[key][0].append(acc_time[0]/Iters)
        TIMES[key][1].append(acc_time[1]/Iters)
        MIN_DIST[key].append(acc_min_dist/Iters)

        with open(f"figures/data/{FigureName}.json", 'w') as file:
            json.dump({
                'TIMES' : TIMES
            }, file)

## Plot
fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=200)

some_key = list(TIMES.keys())[0]
provenance = [
    axs[0].plot(Dims, TIMES[some_key][0], label='Flowtree', color='gray')[0],
    axs[0].plot(Dims, TIMES[some_key][1], '--', label='POT', color='gray')[0],
]

experience = []
for i, N, s in zip(range(len(Nb_points)), Nb_points, Support_size):
    key = f"{N = } & {s = }"
    color = get_cmap('tab10')(i)
    experience.append(axs[0].plot(Dims, TIMES[key][0], label=f"{i+1}", color=color)[0])
    axs[0].plot(Dims, TIMES[key][1], '--', color=color)
    axs[1].plot(Dims, MIN_DIST[key], label=f"{i+1} : {key}")

provenance_legend = axs[0].legend(handles=provenance, loc='upper center')
axs[0].add_artist(provenance_legend)
axs[0].legend(handles=experience, loc='upper right')
axs[1].legend()

for i in range(2):
    axs[i].set_xlabel('Dimension')
    axs[i].set_xticks(Dims)
    axs[i].set_xticklabels(Dims)
    axs[i].set_yscale('log')


axs[0].set_title('Average time to compute W1')
axs[0].set_ylabel('Time (s)')

axs[1].set_title('Minimal distance between points')

plt.savefig(f"figures/imgs/{FigureName}.png")
plt.show()