"""Unit tests for the SymmetryPlane class."""

import numpy as np
import pytest
from pyclarcs._symmetry import SymmetryPlane


def test_reflection_identity():
    """Reflecting twice should return the original point."""
    plane = SymmetryPlane(n=[1, 0, 0], d=0.0)
    pts = np.random.default_rng(0).standard_normal((50, 3))
    np.testing.assert_allclose(plane.apply(plane.apply(pts)), pts, atol=1e-10)


def test_reflection_on_plane():
    """Points on the plane should be unchanged by reflection."""
    plane = SymmetryPlane(n=[0, 0, 1], d=2.0)
    # z = 2 is the plane; build points with z = 2
    pts = np.column_stack([
        np.linspace(0, 1, 20),
        np.linspace(0, 1, 20),
        np.full(20, 2.0),
    ])
    np.testing.assert_allclose(plane.apply(pts), pts, atol=1e-10)


def test_fit_exact():
    """Fitting to perfectly symmetric correspondences should recover the plane."""
    rng = np.random.default_rng(42)
    true_n = np.array([1.0, 0.0, 0.0])
    true_d = 3.0
    true_plane = SymmetryPlane(true_n, true_d)

    source = rng.standard_normal((200, 3)) * 5 + np.array([3.0, 0, 0])
    target = true_plane.apply(source)

    fitted = SymmetryPlane()
    fitted.fit(source, target)

    # The fitted plane should map source to target
    reflected = fitted.apply(source)
    np.testing.assert_allclose(reflected, target, atol=1e-6)


def test_save_load(tmp_path):
    """Save/load round-trip should preserve n and d."""
    plane = SymmetryPlane(n=[0.6, 0.8, 0.0], d=1.5)
    path = str(tmp_path / "plane.pl")
    plane.save(path)

    loaded = SymmetryPlane.load(path)
    np.testing.assert_allclose(loaded.n, plane.n, atol=1e-6)
    assert abs(loaded.d - plane.d) < 1e-6


def test_signed_distance():
    plane = SymmetryPlane(n=[0, 1, 0], d=0.0)
    pts = np.array([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0]])
    dists = plane.signed_distance(pts)
    np.testing.assert_allclose(dists, [1.0, -2.0], atol=1e-10)
