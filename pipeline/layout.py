"""Force-directed layout, precomputed so the viewers can render with the
GPU simulation off."""

import random

import numpy as np

SPACE = 8192  # must match spaceSize in the map viewers


def layout_positions(
    n_nodes: int, edges: list[tuple[int, int, float]], seed: int = 42
) -> np.ndarray:
    import igraph as ig

    # igraph draws from Python's random module, not numpy's. Seeding here
    # keeps reruns from rewriting every coordinate in the committed snapshot.
    ig.set_random_number_generator(random.Random(seed))
    graph = ig.Graph(n=n_nodes, edges=[(a, b) for a, b, _ in edges])
    weights = [w for _, _, w in edges]
    coords = graph.layout_fruchterman_reingold(weights=weights, niter=400)
    return scale_positions(np.asarray(coords.coords, dtype=np.float64))


def scale_positions(points: np.ndarray, space: int = SPACE, margin: float = 0.08) -> np.ndarray:
    """Fit arbitrary layout coordinates into the viewer's space with a margin."""
    if len(points) == 0:
        return points
    lo = points.min(axis=0)
    span = points.max(axis=0) - lo
    span[span == 0] = 1.0
    unit = (points - lo) / span
    return unit * space * (1 - 2 * margin) + space * margin
