from __future__ import annotations
import numpy as np
from typing import Any
from tqdm.auto import trange
from treeswift import Tree, Node
from numpy import ndarray as Array
import matplotlib.pyplot as plt
import random as rd

from hypercube import Hypercube

def l1_normalize(points) -> None:
    N = points.shape[0]
    min_dist = np.inf
    for i in trange(N, desc='l1 normalization'):
        for j in range(i+1, N):
            dist = np.max(np.abs(points[i] - points[j]))
            min_dist = min(min_dist, dist)
    points /= np.sqrt(min_dist)
    points -= np.min(points)
    return points

class Quadtree(Tree):

    def __init__(self, points):
        super().__init__()
        self.original_data = points
        self.X = np.hstack((
            np.arange(points.shape[0]).reshape((-1, 1)),
            # l1_normalize(points + 0),
            points - np.min(points),
        ))
        self.phi = np.max(self.X[:, 1:])
        bounds = np.zeros((points.shape[1], 2))
        bounds[:, 1] = self.phi
        self.root.add_child(QuadNode(Hypercube(bounds, self.X), np.log(self.phi)))
        self.root.demand = ({}, {})

class QuadLeaf(Node):

    def __init__(self, x: np.ndarray, level: int) -> None:
        super().__init__(label=int(x[0]), edge_length=2**level)
        self.x = x[1:]
        self.level = level
        self.demand = ({}, {})
    
    def __repr__(self) -> str:
        return str(self.label)

class QuadNode(Node):

    def __init__(self, hypercube: Hypercube, level: int) -> None:
        super().__init__(label=level, edge_length=2**level)
        self.level = level
        self.demand = ({}, {})
        for child in hypercube.divide():
            if isinstance(child, Hypercube):
                self.add_child(QuadNode(child, level-1))
            else:
                self.add_child(QuadLeaf(child, level-1))

class Distribution:

    def __init__(self, support: Array, weights: Array) -> None:
        mass = np.sum(weights)
        if mass != 1:
            print(f"Normalizing distribution's mass from {mass} to 1")
            weights /= mass
        self.core = {point : weight for point, weight in zip(support, weights)}

    def __call__(self, index: int) -> float:
        if index in self.core:
            return self.core[index]
        return 0.0

    def __repr__(self) -> str:
        return '\t'.join(list(self.core.keys())[:6]) + '...\n' + '\t'.join(list(self.core.values())[:6])
    
    def __contains__(self, key: Any) -> bool:
        return (key in self.core)

    @classmethod
    def uniform_distribution(cls, N: int, s: int) -> Distribution:
        support = np.arange(0, N, 1)
        np.random.shuffle(support)
        support = support[:s]
        weights = np.random.rand(s)
        return Distribution(support, weights)

    def plot(self, N: int):
        support = list(self.core.keys())
        weights = [self.core[k] for k in support]
        plt.bar(support, weights)
        plt.xlim(0, N)
        plt.show()    