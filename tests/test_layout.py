import numpy as np

from pipeline import layout


def test_scale_positions_fits_space_with_margin():
    points = np.array([[0.0, 0.0], [10.0, 5.0], [-4.0, 20.0]])
    scaled = layout.scale_positions(points, space=1000, margin=0.1)
    assert scaled.min() >= 100.0
    assert scaled.max() <= 900.0


def test_scale_positions_degenerate_axis():
    """All nodes on one line must not divide by zero."""
    points = np.array([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])
    scaled = layout.scale_positions(points)
    assert np.isfinite(scaled).all()


def test_scale_positions_empty():
    assert layout.scale_positions(np.empty((0, 2))).shape == (0, 2)
