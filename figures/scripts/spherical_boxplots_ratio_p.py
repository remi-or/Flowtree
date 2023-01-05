## Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from tqdm.auto import trange

from core.flowtree import Distribution, Quadtree, reference_w1_distance
from figures.utils import spherical_sample


## Script

# Params
N = 10000
s = 400
noise = 0.04
iters = 1000
ps = [0.5, 1, 1.5, 2, 3, 4, 5, 6]
RS = np.random.RandomState(0)

# Generate data and compute flowtree
X = spherical_sample(N, 2, 0.04, rs=RS)
Q_X = Quadtree(X)

# Gather measure
DATA = {p : [[], []] for p in ps}
for _ in trange(iters):
    # Draw distributions
    mu = Distribution.uniform_distribution(N, s, RS)
    nu = Distribution.uniform_distribution(N, s, RS)
    # Compute W1 estimates
    for p, estimate in zip(ps, Q_X.compute_w1_distance(mu, nu, ps)):
        DATA[p][0].append(estimate)
    # Compute W1 references
    for p in ps:
        DATA[p][1].append(reference_w1_distance(X, mu, nu, p))

# Prepare figure
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
outfile = 'figures/imgs/spherical_boxplots_ratio_p.png'

# Plot points and boxplots
for i, p in enumerate(ps):
    r = [x[0]/x[1] for x in zip(*DATA[p])]
    ax.scatter(x=r, y=RS.normal(i/4, 0.02, size=len(r)), 
               s=1, color=get_cmap('tab20')(2*i+1))
    medianprops=dict(linewidth=3, color=get_cmap('tab20')(2*i))
    ax.boxplot(x=r, positions=[i/4], vert=False, 
               medianprops=medianprops, showfliers=False)

# Figre details
ax.set_title('Noisy spherical distribution with N = 10000, s = 400')
ax.set_xlabel('Ratio of estimated W1 to actual W1')
ax.set_yticklabels(ps)
ax.set_ylim(-0.25, 2.0)
ax.set_ylabel('p')
plt.savefig(outfile)
plt.show()