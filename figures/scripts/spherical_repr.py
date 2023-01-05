## Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from core.flowtree import Distribution
from figures.utils import spherical_sample


## Params
N = 10000
s = 400
noise = 0.04
RS = np.random.RandomState(0)
outfile = 'figures/imgs/spherical_repr.png'

## Script

# Generate data
X = spherical_sample(N, 2, noise, rs=RS)
mu = Distribution.uniform_distribution(N, s, RS)
nu = Distribution.uniform_distribution(N, s, RS)

# Plot figure
ax = plt.subplots(1, 1, figsize=(6, 6))[1]
for i, mask, label in zip(
    range(3), 
    [range(X.shape[0]), list(mu.support()), list(nu.support())],
    ['Points', 'Distribution 1', 'Distribution 2'],
):
    ax.scatter(*X[mask].T, s=1, color=get_cmap('tab10')(i))
    ax.scatter(0, 0, color=get_cmap('tab10')(i), label=label)
    ax.scatter(0, 0, s=100, color='white')
ax.set_title('Two uniform distribution with s = 400 and N = 10000')
plt.legend()
plt.savefig(outfile)
plt.plot()