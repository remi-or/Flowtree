from typing import Tuple, Optional, List
from numpy.random import RandomState
from numpy import ndarray as Array
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

import numpy as np

from core.flowtree import Distribution


def spherical_sample(
    nb_points: int,
    dimension: int,
    noise: float = 0.0,
    radius: float = 1.0,
    rs: RandomState = None,
) -> Array:
    assert rs is not None, 'Pass a random state to ensure reproducibility.'
    sample = 2*rs.rand(nb_points, dimension) - 1
    norms = np.linalg.norm(sample, axis=1, keepdims=True)
    if noise:
        norms *= rs.normal(1, noise, size=(nb_points, 1))
    sample = radius * (sample / norms)
    return sample

def plot_2d_distribution(
    points: Array, # N, 2
    distribution: Distribution, # s
    ax: Optional[plt.Axes] = None,
    title: str = '',
    cmap: str = 'jet',
    size: int = 3,
    alpha: float = 1.0,
    plot_unsupported: bool = True,
) -> None:
    """
    Plots a 2D distribution (distribution) defined on (points).
    The plot is on the given (ax) but an ax is created if none is given.
    Points have a size of (size) and can be made transparent according to 
    (alpha). Default size is 3 and default alpha is 1.0, i.e. opaque.
    Colors are determined by weights and (cmap) which defaults to 'jet'.
    """
    ax = plt.subplots(1, 1, figsize=(6, 6))[1] if (ax is None) else ax
    # Eventually plot points outisde the support
    if plot_unsupported:
        ax.scatter(*points.T, s=1, color='#dfdfdf')
    # Plot points inside the support
    support = distribution.support('array')
    weights = distribution.weights('array')
    ax.scatter(*points[support].T, s=size, c=weights, cmap=cmap, alpha=alpha)
    ax.axis('equal')
    ax.set_title(title)

def plot_2d_distributions(
    points: Array, 
    distributions: List[Distribution], 
    ax: Optional[plt.Axes] = None,
    title: str = '',
    cmaps: Optional[List[str]] = None,
    size: int = 3,
    alpha: float = 1.0,
    plot_unsupported: bool = True,
) -> None:
    """
    Plots a 2D distribution (distribution) defined on (points).
    The plot is on the given (ax) but an ax is created if none is given.
    Points have a size of (size) and can be made transparent according to 
    (alpha). Default size is 3 and default alpha is 1.0, i.e. opaque.
    Colors are determined by weights and (cmap) which defaults to 'jet'.
    """
    cmaps = ['Blues', 'Oranges', 'Greens', 'Reds'] if (cmaps is None) else cmaps
    ax = plt.subplots(1, 1, figsize=(6, 6))[1] if (ax is None) else ax
    # Agglomerate supports and colors
    supports, colors = np.zeros((0,)),  np.zeros((0, 4))
    for i, distrib in enumerate(distributions):
        supports = np.hstack((supports, distrib.support('array')))
        weights = distrib.weights('array')
        weights -= np.min(weights)
        weights /= np.max(weights)
        new_colors = get_cmap(cmaps[i])(weights)
        new_colors[:, 3] = alpha
        colors = np.vstack((colors, new_colors))
    # Eventually plot points outside off supports
    if plot_unsupported:
        unsupported = set(range(points.shape[0])).difference(set(supports))
        ax.scatter(*points[list(unsupported)].T, s=1, color='#dfdfdf')
    # Shuffle them
    shuffle = np.random.permutation(colors.shape[0])
    supports = supports[shuffle]
    colors = colors[shuffle]
    # Plot points inside the support
    ax.scatter(*points[supports.astype(int)].T, s=size, c=colors, alpha=alpha)
    ax.axis('equal')
    ax.set_title(title)

def gaussian_distribution(
    nb_points: int,
    dimension: int,
    support_size: int,
    rs: Optional[RandomState] = None,
) -> Tuple[Array, Distribution]:
    assert rs is not None, 'Please pass a random state to ensure reproducibility.'
    # Compute backbone, points and weights
    variance = 2*rs.rand(dimension, dimension) - 1
    backbone = rs.normal(size=(nb_points, dimension))
    points = backbone @ variance
    all_weights = 1 - np.linalg.norm(backbone, axis=1) 
    # Compute distribution
    support = rs.choice(nb_points, (support_size,), False)
    weights = all_weights[support]
    weights -= np.min(weights)
    weights /= np.sum(weights)
    return points, Distribution(support, weights)

def gaussian_distributions(
    nb_points: int,
    dimension: int,
    support_sizes: List[int],
    rs: Optional[RandomState] = None,
) -> Tuple[Array, List[Distribution]]:
    subset_size = nb_points // len(support_sizes)
    points, distributions = np.zeros((0, dimension)), []
    for support_size in support_sizes:
        subset, distrib = gaussian_distribution(subset_size, dimension, support_size, rs)
        support, offset = distrib.support(), points.shape[0]
        for key in support:
            distrib.core[key+offset] = distrib.core.pop(key)
        points = np.vstack((points, subset))
        distributions.append(distrib)
    return points, distributions
