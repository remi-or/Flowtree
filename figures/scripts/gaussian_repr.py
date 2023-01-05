## Imports
from numpy.random import RandomState

import matplotlib.pyplot as plt

from figures.utils import plot_2d_distribution, gaussian_distribution

## Params
N = 10000
s = 800
dim = 2
nrows = 2
ncols = 4
RS = RandomState(0)
figure_name = 'gaussian_repr'

## Script
fig, axs = plt.subplots(2, 4, figsize=(4*ncols, 0.95*nrows*ncols))
fig.suptitle('Gaussian distributions with N = 10000 and s = 800')

for j in range(ncols):
    for i in range(nrows):
        support, distribution = gaussian_distribution(N, dim, s, RS)
        plot_2d_distribution(support, distribution, ax=axs[i, j])

plt.savefig(f"figures/imgs/{figure_name}.png")
plt.show()