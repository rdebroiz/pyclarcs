"""
Tests for pyclarcs.atlas and the clarcs atlas CLI command.

test_build_atlas_*  : build_atlas() library function (unit + integration)
test_cli_atlas_*    : CLI command (Click CliRunner)
"""

import numpy as np
import pytest
from click.testing import CliRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sphere_surface(offset=(0.0, 0.0, 0.0), phi_res=8, theta_res=8):
    """Return (pts, polygons) for a small connected VTK sphere mesh."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    src = vtk.vtkSphereSource()
    src.SetRadius(10.0)
    src.SetCenter(*offset)
    src.SetPhiResolution(phi_res)
    src.SetThetaResolution(theta_res)
    src.Update()
    poly = src.GetOutput()

    pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    cells = poly.GetPolys()
    cells.InitTraversal()
    id_list = vtk.vtkIdList()
    polys = []
    while cells.GetNextCell(id_list):
        polys.append([id_list.GetId(i) for i in range(id_list.GetNumberOfIds())])

    return pts, polys


def _save_sphere(tmp_path, name, offset=(0.0, 0.0, 0.0)):
    """Save a sphere surface to tmp_path/<name>.vtk and return the path."""
    from pyclarcs.io import save_surface
    pts, polys = _sphere_surface(offset=offset)
    path = str(tmp_path / f"{name}.vtk")
    save_surface(path, pts, polys)
    return path


def _make_subjects_dir(tmp_path, n_subjects=3, x_offsets=None):
    """Create a directory with n_subjects sphere surfaces displaced along x."""
    if x_offsets is None:
        x_offsets = [k * 3.0 for k in range(n_subjects)]
    subjects_dir = tmp_path / "subjects"
    subjects_dir.mkdir()
    for k, dx in enumerate(x_offsets):
        _save_sphere(subjects_dir, f"subject_{k:02d}", offset=(dx, 0.0, 0.0))
    return subjects_dir


def _load_subjects(paths):
    """Load a list of paths as (pts, polygons, normals) tuples."""
    from pyclarcs.io import load_surface_with_normals
    return [load_surface_with_normals(p) for p in paths]


# ---------------------------------------------------------------------------
# build_atlas — unit tests (no registration)
# ---------------------------------------------------------------------------

def test_build_atlas_requires_two_subjects(tmp_path):
    from pyclarcs.atlas import build_atlas
    from pyclarcs.io import load_surface_with_normals

    path = _save_sphere(tmp_path, "s0")
    subjects = _load_subjects([path])

    with pytest.raises(ValueError, match="at least 2"):
        build_atlas(subjects, atlas_iter=1, verbose=False)


# ---------------------------------------------------------------------------
# build_atlas — integration tests (run actual registration)
# ---------------------------------------------------------------------------

def test_build_atlas_output_shapes(tmp_path):
    """build_atlas returns arrays with consistent shapes."""
    from pyclarcs.atlas import build_atlas

    n_subjects = 3
    paths = [
        _save_sphere(tmp_path, f"s{k}", offset=(k * 3.0, 0.0, 0.0))
        for k in range(n_subjects)
    ]
    subjects = _load_subjects(paths)
    n_pts = len(subjects[0][0])

    mean_pts, mean_polys, registered = build_atlas(
        subjects,
        atlas_iter=1,
        verbose=False,
        n_levels=1,
        max_iter=5,
    )

    assert mean_pts.shape == (n_pts, 3)
    assert mean_polys is subjects[0][1]          # topology object unchanged
    assert len(registered) == n_subjects
    for r in registered:
        assert r.shape == (n_pts, 3)             # atlas topology throughout


def test_build_atlas_mean_shifts_toward_population(tmp_path):
    """After one atlas iteration the mean x-centroid must exceed the template's."""
    from pyclarcs.atlas import build_atlas

    x_offsets = [0.0, 10.0, 20.0]
    paths = [
        _save_sphere(tmp_path, f"s{k}", offset=(dx, 0.0, 0.0))
        for k, dx in enumerate(x_offsets)
    ]
    subjects = _load_subjects(paths)
    template_x = float(subjects[0][0][:, 0].mean())

    mean_pts, _, _ = build_atlas(
        subjects,
        atlas_iter=1,
        verbose=False,
        n_levels=1,
        max_iter=5,
    )

    assert float(mean_pts[:, 0].mean()) > template_x, (
        "Atlas did not shift toward the population (centroid unchanged)."
    )


def test_build_atlas_registered_closer_to_subjects(tmp_path):
    """Each registered[i] must be closer to subject i than the raw template."""
    from pyclarcs.atlas import build_atlas
    from scipy.spatial import KDTree

    x_offsets = [0.0, 15.0]
    paths = [
        _save_sphere(tmp_path, f"s{k}", offset=(dx, 0.0, 0.0))
        for k, dx in enumerate(x_offsets)
    ]
    subjects = _load_subjects(paths)
    template_pts = subjects[0][0]

    mean_pts, _, registered = build_atlas(
        subjects,
        atlas_iter=1,
        verbose=False,
        n_levels=1,
        max_iter=5,
    )

    # registered[1] should be closer to subject_1 than the template was
    sub1_pts = subjects[1][0]
    rms_before = float(np.sqrt(np.mean(
        KDTree(sub1_pts).query(template_pts, k=1, workers=-1)[0] ** 2
    )))
    rms_after = float(np.sqrt(np.mean(
        KDTree(sub1_pts).query(registered[1], k=1, workers=-1)[0] ** 2
    )))
    assert rms_after < rms_before, (
        f"registered[1] (RMS={rms_after:.3f}) is not closer to subject_1 "
        f"than the template (RMS={rms_before:.3f})."
    )


# ---------------------------------------------------------------------------
# CLI — atlas command
# ---------------------------------------------------------------------------

def test_cli_atlas_basic(tmp_path):
    """atlas command exits cleanly and produces a valid output file."""
    from pyclarcs._cli import cli
    from pyclarcs.io import load_surface

    subjects_dir = _make_subjects_dir(tmp_path)
    out_file = str(tmp_path / "atlas.vtk")

    result = CliRunner().invoke(cli, [
        "atlas", str(subjects_dir), out_file,
        "--atlas-iter", "1", "--n-levels", "1", "--max-iter", "5", "-q",
    ])
    assert result.exit_code == 0, result.output

    pts, polys = load_surface(out_file)
    assert pts.shape[1] == 3
    assert len(pts) > 0


def test_cli_atlas_rms_line_always_printed(tmp_path):
    """The RMS line must be printed even with --quiet."""
    from pyclarcs._cli import cli

    subjects_dir = _make_subjects_dir(tmp_path)
    out_file = str(tmp_path / "atlas.vtk")

    result = CliRunner().invoke(cli, [
        "atlas", str(subjects_dir), out_file,
        "--atlas-iter", "1", "--n-levels", "1", "--max-iter", "3", "-q",
    ])
    assert result.exit_code == 0, result.output
    assert "RMS" in result.output
    assert "→" in result.output


def test_cli_atlas_save_registered(tmp_path):
    """--save-registered writes one file per subject with atlas topology."""
    from pyclarcs._cli import cli
    from pyclarcs.io import load_surface

    n_subjects = 3
    subjects_dir = _make_subjects_dir(tmp_path, n_subjects=n_subjects)
    out_file = str(tmp_path / "atlas.vtk")

    result = CliRunner().invoke(cli, [
        "atlas", str(subjects_dir), out_file,
        "--atlas-iter", "1", "--n-levels", "1", "--max-iter", "3",
        "--save-registered", "-q",
    ])
    assert result.exit_code == 0, result.output

    registered_files = sorted(tmp_path.glob("atlas-registered-*.vtk"))
    assert len(registered_files) == n_subjects

    atlas_pts, _ = load_surface(out_file)
    for rf in registered_files:
        reg_pts, _ = load_surface(str(rf))
        assert reg_pts.shape == atlas_pts.shape


def test_cli_atlas_ignores_non_surface_files(tmp_path):
    """Non-surface files in SUBJECTS_DIR must be silently ignored."""
    from pyclarcs._cli import cli
    from pyclarcs.io import load_surface

    subjects_dir = _make_subjects_dir(tmp_path, n_subjects=2)
    (subjects_dir / "README.txt").write_text("ignore me")
    (subjects_dir / "params.json").write_text("{}")
    out_file = str(tmp_path / "atlas.vtk")

    result = CliRunner().invoke(cli, [
        "atlas", str(subjects_dir), out_file,
        "--atlas-iter", "1", "--n-levels", "1", "--max-iter", "3", "-q",
    ])
    assert result.exit_code == 0, result.output
    pts, _ = load_surface(out_file)
    assert len(pts) > 0


def test_cli_atlas_too_few_subjects(tmp_path):
    """atlas command must error when SUBJECTS_DIR has only one surface."""
    from pyclarcs._cli import cli

    subjects_dir = tmp_path / "subjects"
    subjects_dir.mkdir()
    _save_sphere(subjects_dir, "only_one")

    result = CliRunner().invoke(cli, [
        "atlas", str(subjects_dir), str(tmp_path / "atlas.vtk"),
    ])
    assert result.exit_code != 0


def test_cli_atlas_empty_dir(tmp_path):
    """atlas command must error when SUBJECTS_DIR has no surfaces at all."""
    from pyclarcs._cli import cli

    subjects_dir = tmp_path / "subjects"
    subjects_dir.mkdir()

    result = CliRunner().invoke(cli, [
        "atlas", str(subjects_dir), str(tmp_path / "atlas.vtk"),
    ])
    assert result.exit_code != 0
