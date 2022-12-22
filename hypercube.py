from collections import defaultdict
import numpy as np

class Hypercube:

    def __init__(self,
        bounds, # d, 2
        points, # k, 1+d
    ) -> None:
        d, two = bounds.shape
        if (two != 2) or (points.shape[1] != d+1):
            print(f"bounds : {bounds.shape}")
            print(f"points : {points.shape}")
            raise ValueError('Shape mismatch.')
        self.bounds = bounds
        self.points = points
    
    def divide(self):
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
            # If the child only has one point, it will be a leaf:
            if len(points) == 1:
                children.append(self.points[points[0]])
                continue
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