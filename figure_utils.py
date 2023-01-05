from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from numpy.random import RandomState

Array = np.ndarray


def plot_2d_distribution(
    supp: Array, # N, 2
    mu: Array, # N
    ax: Optional[plt.Axes] = None,
    title: str = '',
    cmap: str = 'jet',
    size: int = 1,
    alpha: float = 1.0,
) -> None:
    """
    Plots a 2D distribution (distrib) defined on with support (supp).
    The plot is on the given (ax) but an ax is created if none is given.
    Points have a size of (size) and can be made transparent according to 
    (alpha). Default size is 1 and default alpha is 1.0, i.e. opaque.
    Colors are determined by weights and (cmap) which defaults to 'jet'.
    """
    # Check shapes
    N = supp.shape[0]
    assert supp.shape == (N, 2)
    assert mu.shape == (N, )
    # Compute colors
    normalized = mu - np.min(mu)
    normalized /= np.max(normalized)
    colors = get_cmap(cmap)(normalized)
    # Compute permutation for increasing weight
    perm = np.argsort(mu)
    # Eventually generate the ax
    if ax is None:
        ax = plt.subplots(1, 1, figsize=(6, 6))[1]
    # Plot points
    ax.scatter(*supp[perm].T, s=size, c=colors[perm], alpha=alpha)
    ax.axis('equal')
    ax.set_title(title)

def spherical_distribution(
    nb_points: int,
    support_size: int,
    dimension: int,
    noise: float = 0.0,
    radius: float = 1.0,
    rs: Optional[RandomState] = None,
) -> Tuple[Array, Array]:
    assert support_size <= nb_points
    rs = np.random if (rs is None) else rs
    # Draw points and normalize them
    support = 2*rs.rand(nb_points, dimension) - 1
    norms = np.linalg.norm(support, axis=1, keepdims=True)
    if noise:
        norms *= rs.normal(1, noise, size=(nb_points, 1))
    support = radius * (support / norms)
    # Compute distribution
    distribution = np.zeros((nb_points,))
    if support_size:
        distribution[rs.choice(nb_points, support_size, False)] = 1/support_size
    return support, distribution

def gaussian_distribution(
    nb_points: int,
    dimension: int,
    rs: Optional[RandomState] = None,
) -> Tuple[Array, Array]:
    # Eventually define a default random state
    rs = np.random if (rs is None) else rs
    # Draw variance
    variance = 2*rs.rand(dimension, dimension) - 1
    # Compute support
    backbone = rs.normal(size=(nb_points, dimension))
    support = backbone @ variance
    # Compute distribution
    distribution = np.linalg.norm(backbone, axis=1)
    distribution = 1 - (distribution / np.max(distribution))
    distribution /= np.sum(distribution)
    return support, distribution