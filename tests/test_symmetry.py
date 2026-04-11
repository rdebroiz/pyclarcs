"""
Unit and integration tests for pyclarcs.

test_symmetry_*   : SymmetryPlane (core geometry and fitting)
test_principal_*  : principal-axis initialisation
test_pipeline_*   : full coarse + fine optimisation pipeline
"""

import numpy as np
import pytest
from pyclarcs._symmetry import SymmetryPlane


# ---------------------------------------------------------------------------
# SymmetryPlane — geometry
# ---------------------------------------------------------------------------

def test_reflection_identity():
    """Reflecting twice must return the original points (R² = I)."""
    plane = SymmetryPlane(n=[1, 0, 0], d=0.0)
    pts = np.random.default_rng(0).standard_normal((50, 3))
    np.testing.assert_allclose(plane.apply(plane.apply(pts)), pts, atol=1e-10)


def test_reflection_on_plane():
    """Points lying on the plane must be unchanged by reflection."""
    plane = SymmetryPlane(n=[0, 0, 1], d=2.0)
    # Build points with z = 2  (lying on the plane z = 2)
    pts = np.column_stack([
        np.linspace(0, 1, 20),
        np.linspace(0, 1, 20),
        np.full(20, 2.0),
    ])
    np.testing.assert_allclose(plane.apply(pts), pts, atol=1e-10)


def test_signed_distance():
    """Signed distance: positive on the normal side, negative on the other."""
    plane = SymmetryPlane(n=[0, 1, 0], d=0.0)
    pts = np.array([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0]])
    dists = plane.signed_distance(pts)
    np.testing.assert_allclose(dists, [1.0, -2.0], atol=1e-10)


def test_project_on_plane():
    """Projected points must lie on the plane (signed distance = 0)."""
    plane = SymmetryPlane(n=[1, 0, 0], d=3.0)
    pts = np.random.default_rng(1).standard_normal((30, 3)) * 10
    proj = plane.project(pts)
    np.testing.assert_allclose(plane.signed_distance(proj), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# SymmetryPlane — fitting (M-step)
# ---------------------------------------------------------------------------

def test_fit_exact():
    """Fitting perfectly symmetric correspondences must recover the plane exactly."""
    rng = np.random.default_rng(42)
    true_plane = SymmetryPlane(n=[1.0, 0.0, 0.0], d=3.0)

    source = rng.standard_normal((200, 3)) * 5 + np.array([3.0, 0, 0])
    target = true_plane.apply(source)

    fitted = SymmetryPlane()
    fitted.fit(source, target)

    # The fitted plane must map source to target within floating-point precision
    np.testing.assert_allclose(fitted.apply(source), target, atol=1e-6)


def test_fit_weighted():
    """Fitting with uniform weights must give the same result as no weights."""
    rng = np.random.default_rng(7)
    true_plane = SymmetryPlane(n=[0.0, 1.0, 0.0], d=2.0)
    source = rng.standard_normal((100, 3)) * 3
    target = true_plane.apply(source)

    p1 = SymmetryPlane(); p1.fit(source, target)
    p2 = SymmetryPlane(); p2.fit(source, target, weights=np.ones(100))

    np.testing.assert_allclose(p1.n, p2.n, atol=1e-8)
    assert abs(p1.d - p2.d) < 1e-8


# ---------------------------------------------------------------------------
# SymmetryPlane — serialisation
# ---------------------------------------------------------------------------

def test_save_load(tmp_path):
    """Save / load round-trip must preserve n and d to floating-point precision."""
    plane = SymmetryPlane(n=[0.6, 0.8, 0.0], d=1.5)
    path = str(tmp_path / "plane.pl")
    plane.save(path)

    loaded = SymmetryPlane.load(path)
    np.testing.assert_allclose(loaded.n, plane.n, atol=1e-6)
    assert abs(loaded.d - plane.d) < 1e-6


# ---------------------------------------------------------------------------
# Principal-axis initialisation
# ---------------------------------------------------------------------------

def _make_bilateral_ellipsoid(seed: int = 42) -> np.ndarray:
    """Generate a bilateral ellipsoid (a=15, b=60, c=45) centred at x=5.

    Each point on the x>5 side has its exact mirror on the x<5 side, then
    light Gaussian noise is added.  The unique symmetry plane is x = 5.
    """
    rng = np.random.default_rng(seed)
    N_half = 500
    a, b, c = 15.0, 60.0, 45.0

    half_pts: list[np.ndarray] = []
    while len(half_pts) < N_half:
        batch = rng.standard_normal((N_half * 4, 3))
        batch /= np.linalg.norm(batch, axis=1, keepdims=True)
        batch = batch[batch[:, 0] > 0][: N_half - len(half_pts)]
        half_pts.append(batch)
    half = np.vstack(half_pts)[:N_half] * [a, b, c] + [5.0, 0.0, 0.0]

    mirror = half.copy()
    mirror[:, 0] = 10.0 - half[:, 0]   # reflection: x' = 2*5 − x

    pts = np.vstack([half, mirror])
    pts += rng.standard_normal((len(pts), 3)) * 0.3
    return pts


def test_principal_axis_init():
    """The best principal-axis candidate must be close to the true plane x=5."""
    pts = _make_bilateral_ellipsoid()
    plane = __import__("pyclarcs._principal_axes", fromlist=["best_principal_axis_plane"]).best_principal_axis_plane(pts)

    angle_err = np.degrees(np.arccos(np.clip(abs(np.dot(plane.n, [1, 0, 0])), 0, 1)))
    assert angle_err < 5.0, f"Init angle error too large: {angle_err:.2f}°"
    assert abs(plane.d - 5.0) < 3.0, f"Init d error too large: {abs(plane.d-5.0):.3f}"


# ---------------------------------------------------------------------------
# Full pipeline integration test
# ---------------------------------------------------------------------------

def test_pipeline_bilateral_ellipsoid():
    """Full 4-stage pipeline must recover the symmetry plane x=5 accurately.

    The bilateral ellipsoid (a=15, b=60, c=45) with light noise (σ=0.3 mm)
    is a realistic proxy for an endocranial surface.

    Tolerance: angle < 1°, |Δd| < 0.5 mm.
    """
    from pyclarcs._principal_axes import best_principal_axis_plane
    from pyclarcs._coarse import coarse_symmetry
    from pyclarcs._fine import em_icp_sym, em_icp_sym_corres

    pts = _make_bilateral_ellipsoid(seed=0)

    plane = best_principal_axis_plane(pts)
    plane = coarse_symmetry(pts, plane, seed=0)
    plane = em_icp_sym(pts, plane, sigma_init=5.0, sigma_final=0.5)
    plane = em_icp_sym_corres(pts, plane, sigma=0.25)

    angle_err = np.degrees(np.arccos(np.clip(abs(np.dot(plane.n, [1, 0, 0])), 0, 1)))
    d_err = abs(plane.d - 5.0)

    assert angle_err < 1.0, f"Angle error: {angle_err:.3f}°"
    assert d_err < 0.5,     f"d error: {d_err:.4f}"
