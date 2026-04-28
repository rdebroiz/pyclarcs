"""
Tests for pyclarcs.alignment and the CLI alignment sub-commands.

test_centerofmass_* : align_center_of_mass
test_rescale_*      : align_rescale
test_orient_*       : reorient_axes
test_recenter_*     : align_to_symmetry_plane
test_cli_*          : Click CLI commands (centerofmass, normalize, recenter, reorient)
"""

import numpy as np
import pytest
from click.testing import CliRunner

from pyclarcs.alignment import (
    align_center_of_mass,
    align_rescale,
    align_to_symmetry_plane,
    reorient_axes,
)
from pyclarcs.symmetry import SymmetryPlane


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cloud(seed=0, n=300, scale=10.0, offset=(0.0, 0.0, 0.0)):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3)) * scale + np.array(offset)


def _triangles(n_pts):
    """Minimal triangle list so VTK doesn't discard unreferenced points."""
    return [[i, (i + 1) % n_pts, (i + 2) % n_pts] for i in range(0, n_pts - 2, 3)]


def _bilateral(seed=0, n_half=400, plane_n=(1, 0, 0), plane_d=5.0):
    """Bilateral point cloud with a known symmetry plane and light noise."""
    rng = np.random.default_rng(seed)
    n = np.asarray(plane_n, dtype=float)
    n /= np.linalg.norm(n)
    half = rng.standard_normal((n_half, 3)) * np.array([5.0, 20.0, 15.0])
    half += plane_d * n
    mirror = half - 2.0 * (half @ n - plane_d)[:, None] * n
    pts = np.vstack([half, mirror])
    pts += rng.standard_normal(pts.shape) * 0.1
    return pts


# ---------------------------------------------------------------------------
# align_center_of_mass
# ---------------------------------------------------------------------------

def test_centerofmass_centroid_matches_target():
    """After alignment the centroid of the result must equal the target centroid."""
    src = _cloud(seed=0, offset=(10.0, -5.0, 3.0))
    tgt = _cloud(seed=1, offset=(0.0, 0.0, 0.0))
    result = align_center_of_mass(src, tgt)
    np.testing.assert_allclose(result.mean(axis=0), tgt.mean(axis=0), atol=1e-10)


def test_centerofmass_is_pure_translation():
    """Relative positions between points must be unchanged (no rotation, no scale)."""
    src = _cloud(seed=2)
    tgt = _cloud(seed=3)
    result = align_center_of_mass(src, tgt)
    np.testing.assert_allclose(result - result[0], src - src[0], atol=1e-10)


def test_centerofmass_identical_inputs_is_noop():
    pts = _cloud(seed=4)
    np.testing.assert_allclose(align_center_of_mass(pts, pts), pts, atol=1e-10)


# ---------------------------------------------------------------------------
# align_rescale
# ---------------------------------------------------------------------------

def _dispersion(pts):
    return float(np.sqrt(np.sum((pts - pts.mean(axis=0)) ** 2, axis=1)).mean())


def test_rescale_matches_formula():
    """Result must exactly follow the C++ formula: pts * scale + (c_tgt - c_src)."""
    src = _cloud(seed=5, scale=3.0, offset=(20.0, 0.0, 0.0))
    tgt = _cloud(seed=6, scale=12.0, offset=(0.0, 0.0, 0.0))
    result = align_rescale(src, tgt)

    c_src = src.mean(axis=0)
    c_tgt = tgt.mean(axis=0)
    scale = _dispersion(tgt) / _dispersion(src)
    expected = src * scale + (c_tgt - c_src)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_rescale_identical_inputs_is_noop():
    pts = _cloud(seed=7)
    np.testing.assert_allclose(align_rescale(pts, pts), pts, atol=1e-10)


def test_rescale_preserves_shape():
    src = _cloud(seed=8)
    tgt = _cloud(seed=9)
    assert align_rescale(src, tgt).shape == src.shape


# ---------------------------------------------------------------------------
# reorient_axes
# ---------------------------------------------------------------------------

def test_orient_identity():
    pts = _cloud(seed=10)
    np.testing.assert_array_equal(reorient_axes(pts, 0, 1, 2), pts)


def test_orient_swap_x_z():
    pts = _cloud(seed=11)
    result = reorient_axes(pts, 2, 1, 0)
    np.testing.assert_array_equal(result[:, 0], pts[:, 2])
    np.testing.assert_array_equal(result[:, 1], pts[:, 1])
    np.testing.assert_array_equal(result[:, 2], pts[:, 0])


def test_orient_cyclic_permutation():
    """x→1, y→2, z→0."""
    pts = _cloud(seed=12)
    result = reorient_axes(pts, 1, 2, 0)
    np.testing.assert_array_equal(result[:, 1], pts[:, 0])
    np.testing.assert_array_equal(result[:, 2], pts[:, 1])
    np.testing.assert_array_equal(result[:, 0], pts[:, 2])


def test_orient_double_swap_is_identity():
    pts = _cloud(seed=13)
    twice = reorient_axes(reorient_axes(pts, 2, 1, 0), 2, 1, 0)
    np.testing.assert_array_equal(twice, pts)


def test_orient_invalid_permutation_raises():
    pts = _cloud(seed=14)
    with pytest.raises(ValueError):
        reorient_axes(pts, 0, 0, 1)


# ---------------------------------------------------------------------------
# align_to_symmetry_plane
# ---------------------------------------------------------------------------

def test_recenter_is_rigid():
    """align_to_symmetry_plane must preserve pairwise distances."""
    plane = SymmetryPlane(n=[0.6, 0.8, 0.0], d=3.0)
    pts = _bilateral(seed=0, plane_n=[0.6, 0.8, 0.0], plane_d=3.0)
    result = align_to_symmetry_plane(pts, plane)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), size=(100, 2), replace=True)
    d_before = np.linalg.norm(pts[idx[:, 0]] - pts[idx[:, 1]], axis=1)
    d_after  = np.linalg.norm(result[idx[:, 0]] - result[idx[:, 1]], axis=1)
    np.testing.assert_allclose(d_after, d_before, atol=1e-6)


def test_recenter_normal_maps_to_canonical():
    """The plane normal must become [1, 0, 0] after the transform."""
    n_orig = np.array([0.6, 0.8, 0.0])
    plane = SymmetryPlane(n=n_orig, d=4.0)
    pts = _bilateral(seed=1, plane_n=n_orig, plane_d=4.0)
    result = align_to_symmetry_plane(pts, plane)

    # The rotation R has n as its first row, so R @ n = [1, 0, 0].
    # Verify that the mean x-coordinate of the result is near 0
    # (projection of centroid onto the canonical plane):
    n = plane.n / np.linalg.norm(plane.n)
    centroid = pts.mean(axis=0)
    dep1 = centroid - (centroid @ n - plane.d) * n
    # After transform dep1 maps to origin → centroid maps to R @ (centroid - dep1)
    # The x-component = n · (centroid - dep1)
    x_of_centroid = float(n @ (centroid - dep1))
    centroid_result = result.mean(axis=0)
    np.testing.assert_allclose(centroid_result[0], x_of_centroid, atol=1e-6)


def test_recenter_canonical_plane_preserves_centroid_x():
    """If the plane is already n=[1,0,0], the x-centroid of the result is ~0."""
    plane = SymmetryPlane(n=[1, 0, 0], d=0.0)
    pts = _bilateral(seed=2, plane_n=[1, 0, 0], plane_d=0.0)
    result = align_to_symmetry_plane(pts, plane)
    # dep1 = projection of centroid onto x=0, so result centroid has x ≈ 0
    assert abs(result.mean(axis=0)[0]) < 0.5


def test_recenter_output_shape():
    plane = SymmetryPlane(n=[1, 0, 0], d=2.0)
    pts = _bilateral(seed=3)
    result = align_to_symmetry_plane(pts, plane)
    assert result.shape == pts.shape


def test_recenter_plane_at_x0_normal_positive_x():
    """Points on the plane must map to x = 0 when no normal flip is needed."""
    plane = SymmetryPlane(n=[0.6, 0.8, 0.0], d=5.0)
    pts = _bilateral(seed=20, plane_n=[0.6, 0.8, 0.0], plane_d=5.0)
    on_plane = plane.project(pts)          # exactly on the plane: n·p = d
    result = align_to_symmetry_plane(on_plane, plane)
    np.testing.assert_allclose(result[:, 0], 0.0, atol=1e-10)


def test_recenter_plane_at_x0_normal_negative_x():
    """Regression: when plane.n points toward -x, flipping n without flipping d
    caused the symmetry plane to land at x = -2*d instead of x = 0."""
    plane = SymmetryPlane(n=[-0.6, 0.8, 0.0], d=5.0)
    pts = _bilateral(seed=21, plane_n=[-0.6, 0.8, 0.0], plane_d=5.0)
    on_plane = plane.project(pts)          # exactly on the plane: n·p = d
    result = align_to_symmetry_plane(on_plane, plane)
    np.testing.assert_allclose(result[:, 0], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# CLI — centerofmass
# ---------------------------------------------------------------------------

def test_cli_centerofmass(tmp_path):
    from pyclarcs._cli import cli
    from pyclarcs.io import save_surface, load_surface

    src_pts = _cloud(seed=0, offset=(50.0, 0.0, 0.0))
    tgt_pts = _cloud(seed=1, offset=(0.0, 0.0, 0.0))

    src_file = str(tmp_path / "src.vtk")
    tgt_file = str(tmp_path / "tgt.vtk")
    out_file = str(tmp_path / "out.vtk")
    save_surface(src_file, src_pts, _triangles(len(src_pts)))
    save_surface(tgt_file, tgt_pts, _triangles(len(tgt_pts)))

    runner = CliRunner()
    result = runner.invoke(cli, ["centerofmass", src_file, out_file, "--target", tgt_file, "-q"])
    assert result.exit_code == 0, result.output

    out_pts, _ = load_surface(out_file)
    np.testing.assert_allclose(out_pts.mean(axis=0), tgt_pts.mean(axis=0), atol=1e-3)


# ---------------------------------------------------------------------------
# CLI — normalize
# ---------------------------------------------------------------------------

def test_cli_rescale(tmp_path):
    from pyclarcs._cli import cli
    from pyclarcs.io import save_surface, load_surface

    src_pts = _cloud(seed=2, scale=2.0, offset=(10.0, 0.0, 0.0))
    tgt_pts = _cloud(seed=3, scale=10.0)

    src_file = str(tmp_path / "src.vtk")
    tgt_file = str(tmp_path / "tgt.vtk")
    out_file = str(tmp_path / "out.vtk")
    save_surface(src_file, src_pts, _triangles(len(src_pts)))
    save_surface(tgt_file, tgt_pts, _triangles(len(tgt_pts)))

    runner = CliRunner()
    result = runner.invoke(cli, ["normalize", src_file, out_file, "--target", tgt_file, "-q"])
    assert result.exit_code == 0, result.output

    out_pts, _ = load_surface(out_file)
    assert out_pts.shape == src_pts.shape
    # Dispersion of result should be close to target's
    assert abs(_dispersion(out_pts) - _dispersion(tgt_pts)) < _dispersion(tgt_pts) * 0.05


# ---------------------------------------------------------------------------
# CLI — reorient
# ---------------------------------------------------------------------------

def test_cli_orient_swap(tmp_path):
    from pyclarcs._cli import cli
    from pyclarcs.io import save_surface, load_surface

    pts = _cloud(seed=4)

    src_file = str(tmp_path / "src.vtk")
    out_file = str(tmp_path / "out.vtk")
    save_surface(src_file, pts, _triangles(len(pts)))

    runner = CliRunner()
    result = runner.invoke(cli, ["reorient", src_file, out_file, "--axes", "2", "1", "0", "-q"])
    assert result.exit_code == 0, result.output

    out_pts, _ = load_surface(out_file)
    np.testing.assert_allclose(out_pts[:, 0], pts[:, 2], atol=1e-4)
    np.testing.assert_allclose(out_pts[:, 2], pts[:, 0], atol=1e-4)


# ---------------------------------------------------------------------------
# CLI — recenter (with and without --plane)
# ---------------------------------------------------------------------------

def test_cli_recenter_with_plane(tmp_path):
    from pyclarcs._cli import cli
    from pyclarcs.io import save_surface, load_surface

    plane = SymmetryPlane(n=[1, 0, 0], d=5.0)
    pts = _bilateral(seed=5, plane_n=[1, 0, 0], plane_d=5.0)

    src_file = str(tmp_path / "src.vtk")
    pl_file  = str(tmp_path / "plane.pl")
    out_file = str(tmp_path / "out.vtk")
    save_surface(src_file, pts, _triangles(len(pts)))
    plane.save(pl_file)

    runner = CliRunner()
    result = runner.invoke(cli, ["recenter", src_file, out_file, "--plane", pl_file, "-q"])
    assert result.exit_code == 0, result.output

    out_pts, _ = load_surface(out_file)
    assert len(out_pts) > 0
    assert abs(out_pts.mean(axis=0)[0]) < 1.0


def test_cli_recenter_without_plane_produces_output(tmp_path):
    """Without --plane the command must auto-estimate and still produce a valid file."""
    from pyclarcs._cli import cli
    from pyclarcs.io import save_surface, load_surface

    pts = _bilateral(seed=6, plane_n=[1, 0, 0], plane_d=5.0)

    src_file = str(tmp_path / "src.vtk")
    out_file = str(tmp_path / "out.vtk")
    save_surface(src_file, pts, _triangles(len(pts)))

    runner = CliRunner()
    result = runner.invoke(cli, ["recenter", src_file, out_file, "-q"])
    assert result.exit_code == 0, result.output

    out_pts, _ = load_surface(out_file)
    assert len(out_pts) > 0


def test_cli_recenter_save_plane(tmp_path):
    """--save-plane must write a valid .pl file alongside the output."""
    from pyclarcs._cli import cli
    from pyclarcs.io import save_surface

    plane = SymmetryPlane(n=[1, 0, 0], d=5.0)
    pts = _bilateral(seed=7, plane_n=[1, 0, 0], plane_d=5.0)

    src_file = str(tmp_path / "src.vtk")
    pl_file  = str(tmp_path / "plane.pl")
    out_file = str(tmp_path / "out.vtk")
    save_surface(src_file, pts, _triangles(len(pts)))
    plane.save(pl_file)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["recenter", src_file, out_file, "--plane", pl_file, "--save-plane", "-q"],
    )
    assert result.exit_code == 0, result.output

    saved_plane_path = tmp_path / "out.pl"
    assert saved_plane_path.exists(), ".pl file was not created by --save-plane"
    loaded = SymmetryPlane.load(str(saved_plane_path))
    np.testing.assert_allclose(abs(loaded.n[0]), 1.0, atol=0.01)
