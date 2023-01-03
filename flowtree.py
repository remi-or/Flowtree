from __future__ import annotations
from typing import Tuple, Set
from collections import defaultdict
import numpy as np
from typing import Any
from treeswift import Tree, Node
from numpy import ndarray as Array
import matplotlib.pyplot as plt
import ot


EPS = 1e-12


class Hypercube:

    """
    A class to group d-dimensionnal vectors in hypercubes.
    """

    def __init__(self,
        bounds, # d, 2
        points, # k, 1+d
    ) -> None:
        """
        Initializes the d dimensionnal hypercube with its (bounds) and the (points) it contains.
        """
        d, two = bounds.shape
        if (two != 2) or (points.shape[1] != d+1):
            print(f"bounds : {bounds.shape}")
            print(f"points : {points.shape}")
            raise ValueError('Shape mismatch.')
        self.bounds = bounds
        self.points = points
    
    def divide(self):
        """
        Divides the hypericube in 2 alongs all dimensions and keeps only the 
        ones with at least one element. The ones with exactly one element are
        no longer hypercubes, just a d-dimensionnal vector and its index.
        """
        # Compute the signtures of each point
        middle = np.sum(self.bounds, 1) / 2 # d
        signatures = self.points[:, 1:] <= middle
        # Assign each cube its points
        children_descriptions = defaultdict(lambda : [])
        for i, signature in enumerate(signatures):
            children_descriptions[bytes(signature)].append(i)
        # Create the new hypercubes
        children = []
        for points in children_descriptions.values():
            # If the child only has one point, discard the hypercube structure
            if len(points) == 1:
                children.append(self.points[points[0]])
                continue
            # Otherwise, compute the child's new bounds
            signature = signatures[points[0]].astype('int')
            bounds = self.bounds + 0
            for i, bool in enumerate(signature):
                bounds[i, bool] = middle[i]
            points = self.points[points]
            children.append(Hypercube(bounds, points))
        return children

    def __repr__(self) -> str:
        repr = ''
        for lower, upper in self.bounds:
            repr += f"[{lower}, {upper}]\n"
        return repr[:-1]


class Quadtree(Tree):

    "A class to implement the quadtree."

    def __init__(self, points: Array):
        """Initializes and builds the quadtree from the given (points)"""
        super().__init__()
        self.format_points(points)
        self.format_root()
        bounds = np.zeros((self.d, 2))
        bounds[:, 1] = self.phi
        self.root.add_child(
            QuadNode.from_hypercube(Hypercube(bounds, self.X), np.log(self.phi))
            # if done with sigma, should be np.log(self.phi)+1
        )

    def format_points(self, points: Array) -> None:
        # Offset
        self.offset = np.min(points)
        points -= self.offset
        # Phi
        self.phi = np.max(points)
        # Indices
        indices = np.arange(points.shape[0]).reshape((-1, 1))
        points = np.hstack((indices, points))
        # Store
        self.X = points
        self.d = points.shape[1]-1
        self.k = points.shape[0]

    def format_root(self):
        self.root = QuadNode(-2, np.log(self.phi)+1)

    def fill_in_demands(self, ds: Tuple[Distribution, Distribution]) -> None:
        global_support = set(ds[0].support()).union(ds[1].support())
        for node in self.traverse_postorder():
            node.demand = [{}, {}] # for safety in case of multiple runs
            if (index := node.label) in global_support:
                for i in range(2):
                    if index in ds[i]:
                        node.demand[i] = {index : ds[i](index)}

    def compute_optimal_flow(self, d0: Distribution, d1: Distribution):
        self.fill_in_demands((d0, d1))
        optimal_flow = defaultdict(lambda : 0)
        for node in self.traverse_postorder():
            optimal_flow = node.resolve_demand(optimal_flow)
        return dict(optimal_flow)

    def compute_w1_distance(self, d0: Distribution, d1: Distribution):
        optimal_flow = self.compute_optimal_flow(d0, d1)
        w1_distance = 0
        for (i, j), coeff in optimal_flow.items():
            w1_distance += coeff * np.linalg.norm(self.X[i, 1:] - self.X[j, 1:], 1)
        return w1_distance

    def compute_tree_distance(self, i: int, j: int) -> float:
        found = []
        for leaf in self.traverse_leaves():
            if leaf.label == i or leaf.label == j:
                found.append(leaf)
                if len(found) == 2:
                    return self.distance_between(*found)
        raise ValueError(f"Couldn't find labels {(i, j)}")

class QuadNode(Node):

    def __init__(self, 
        label: int, 
        level: float,
    ):
        self.level = level
        self.demand = [{}, {}]
        super().__init__(label, 2**self.level)

    @classmethod
    def from_hypercube(cls, hypercube: Hypercube, level: float):
        node = QuadNode(-1, level)
        for child in hypercube.divide():
            if isinstance(child, Hypercube):
                node.add_child(QuadNode.from_hypercube(child, level-1))
            else:
                node.add_child(QuadNode.from_index(child, level-1))
        return node

    @classmethod
    def from_index(cls, index_and_vector: Array, level: float):
        index = int(index_and_vector[0])
        node = QuadNode(index, level)
        return node

    def find_transaction(self):
        argmins, mins = [-1, -1], [2, 2]
        for i in range(2):
            for index, amount in self.demand[i].items():
                if amount < mins[i]:
                    argmins[i] = index
                    mins[i] = amount
        return (min(mins), argmins)

    def resolve_demand(self, optimal_flow):
        # While we have demand on both sides
        while self.demand[0] and self.demand[1]:
            # Find the minimal transaction
            amount, argmins = self.find_transaction()
            # Substract the amount from the demand on each side
            for i in range(2):
                self.demand[i][argmins[i]] -= amount
                # If the demand falls behind a given threshold, delete it
                if self.demand[i][argmins[i]] < EPS:
                    self.demand[i].pop(argmins[i])
            # Keep track of the satisfied demand in the optimal flow
            optimal_flow[(argmins[0], argmins[1])] += amount
        # Now one side's demand is null
        # Root case
        if self.label == -2 and (self.demand[0] or self.demand[1]):
            print(f"{self.demand[0]}\n{self.demand[1]}")
            raise ValueError('Unresolved demand at root.')
        # Bump up the demand to the parent
        for i in range(2):
            for index, amount in self.demand[i].items():
                # Distinguish cases if the parent already has a demand
                if index in self.parent.demand[i]:
                    self.parent.demand[i][index] += amount
                else:
                    self.parent.demand[i][index] = amount
            self.demand[i] = {}
        return optimal_flow


class Distribution:

    def __init__(self, support: Array, weights: Array) -> None:
        mass = np.sum(weights)
        if abs(mass - 1) > EPS:
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

    def support(self) -> Set[int]:
        return set(self.core.keys())

    @classmethod
    def uniform_distribution(cls, N: int, s: int) -> Distribution:
        support = np.arange(0, N, 1)
        np.random.shuffle(support)
        support = support[:s]
        weights = np.random.rand(s)
        weights /= np.sum(weights)
        return Distribution(support, weights)

    def plot(self, N: int):
        support = list(self.core.keys())
        weights = [self.core[k] for k in support]
        plt.bar(support, weights)
        plt.xlim(0, N)
        plt.show()   

def compute_marginals(flow):
    marginals = [{}, {}]
    for i in range(2):
        for k, amount in flow.items():
            if k[i] in marginals[i]:
                marginals[i][k[i]] += amount
            else:
                marginals[i][k[i]] = amount
    return marginals


def reference_w1_distance(
    points: Array, 
    mu: Distribution, 
    nu: Distribution,
) -> float:
    mu_support, nu_support = list(mu.support()), list(nu.support())
    points_mu, points_nu = points[mu_support], points[nu_support]
    M = ot.dist(points_mu, points_nu, metric='cityblock')
    weights_mu, weights_nu = [mu(i) for i in mu_support], [nu(i) for i in nu_support]
    return ot.emd2(weights_mu, weights_nu, M)

def compare_estimate_and_reference(
    dimension: int, 
    number_of_points: int, 
    support_size: int,
) -> Tuple[float, float]:
    points = np.random.normal(size=(number_of_points, dimension)) 
    mu = Distribution.uniform_distribution(number_of_points, support_size)
    nu = Distribution.uniform_distribution(number_of_points, support_size)
    w1_dist = Quadtree(points).compute_w1_distance(mu, nu)
    w1_ref = reference_w1_distance(points, mu, nu)
    return w1_dist, w1_ref