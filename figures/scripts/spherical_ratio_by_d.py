## Imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange

from core.flowtree import Distribution, Quadtree, reference_w1_distance
from figures.utils import spherical_sample



## Params
N = 10000
s = 400
noise = 0.04
RS = np.random.RandomState(0)
iters = 1000
p = 2
ds = [2, 3, 5] + [i * 10 for i in range(1, 11)]
figure_name = 'spherical_ratio_by_d'


## Script 
DATA = {}
for i_d, d in enumerate(ds):
    print(f"Gathering data for {d = }; step {i_d+1}/{len(ds)}")
    DATUM = [[],  []]
    
    X = spherical_sample(N, d, 0.04, rs=RS)
    Q_X = Quadtree(X)

    for _ in trange(iters):
        mu = Distribution.uniform_distribution(N, s, RS)
        nu = Distribution.uniform_distribution(N, s, RS)
        DATUM[0].append(Q_X.compute_w1_distance(mu, nu, p))
        DATUM[1].append(reference_w1_distance(X, mu, nu, p))

    DATA[d] = DATUM + []


## Plot
mean_ratios = []
for d in ds:
    mean_ratios.append(
        sum((x[0]/x[1] for x in zip(*DATA[d]))) / iters)
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(ds, mean_ratios)
ax.set_xlabel('Dimension')
ax.set_ylabel('Ratio of estimate to actual W1')
ax.set_title('Dependance of estimate\'s quality on dimension')
plt.savefig(f"figures/imgs/{figure_name}.png")
plt.show()