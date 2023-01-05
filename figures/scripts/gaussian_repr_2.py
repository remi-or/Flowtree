## Imports
from numpy.random import RandomState

import matplotlib.pyplot as plt

from figures.utils import plot_2d_distributions, gaussian_distributions

## Params
N = 10000
s = 800
dim = 2
K = 2
RS = RandomState(3)
ncols = 4
figure_name = 'gaussian_repr_2'

## Script
fig, axs = plt.subplots(1, ncols, figsize=(4*ncols, 0.95*ncols))
fig.suptitle('Gaussian distributions with N = 10000 and s = 800')

for j in range(ncols):
    X, mus = gaussian_distributions(N, dim, [s, s], RS)
    plot_2d_distributions(X, mus, ax=axs[j], alpha=0.8)

plt.savefig(f"figures/imgs/{figure_name}.png")
plt.show()