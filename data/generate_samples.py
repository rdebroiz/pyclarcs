#!/usr/bin/env python3
"""
generate_samples.py
===================
Generates the test surface samples for pyclarcs.

Surfaces produced
-----------------
1. endocranium_mni_pial.vtk
   Brain pial surface (MNI standard, Brain for Blender dataset, A. Winkler).
   Left + right hemispheres merged.  Best free proxy for an endocranial
   surface: the pial surface is the outermost cortical surface and closely
   follows the inner skull table.
   Downloaded automatically from brainder.org and cached in
   ~/.cache/pyclarcs/pial_Full_ply/ on first run.

2. endocranium_mni_pial_10k.vtk
   Same as (1) but decimated to ~10 000 points for fast unit tests.

3. ellipsoid_skull_noisy.vtk
   Synthetic skull-shaped ellipsoid (a=85, b=65, c=70 mm), centred at
   (85, 0, 0) mm with σ=0.5 mm Gaussian noise.  The true symmetry plane
   is x = 85.

4. ellipsoid_a15_b60_c45.vtk
   Exact synthetic surface used in unit tests (a=15, b=60, c=45 mm),
   centred at (5, 0, 0) mm, σ=0.3 mm noise.  True symmetry plane: x = 5.

5. ellipsoid_oblate.vtk
   Oblate ellipsoid (a=40, b=90, c=90 mm), centred at (40, 0, 0) mm,
   σ=0.4 mm noise.  True symmetry plane: x = 40.

6. *_target.vtk (one per reference surface)
   Registration test pairs.  Each target is the corresponding reference
   surface after applying:
     (a) A small affine transform: translation (10, 7, −5) mm,
         rotation 12° around axis (0.3, 0.8, 0.5), uniform scale 1.03.
     (b) A smooth non-rigid deformation: 16 Gaussian bumps of amplitude
         15 mm spread across the surface.
   Intended to exercise the full pipeline:
       clarcs normalize  →  clarcs nlregister
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

import click
import numpy as np

# ---------------------------------------------------------------------------
# Make pyclarcs importable (works whether installed or run from source)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # pyclarcs/data/
_SRC  = _HERE.parent / "src"                     # pyclarcs/src/
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from pyclarcs.io import load_surface, save_surface  # noqa: E402

# ---------------------------------------------------------------------------
# MNI pial surface — download constants
# ---------------------------------------------------------------------------
_MNI_URL = (
    "https://s3.us-east-2.amazonaws.com/brainder/software/"
    "brain4blender/smallfiles/pial_Full_ply.tar.bz2"
)
_CACHE_DIR = Path.home() / ".cache" / "pyclarcs"
_ARCHIVE   = _CACHE_DIR / "pial_Full_ply.tar.bz2"
_CACHE     = _CACHE_DIR / "pial_Full_ply"
LH_PLY     = _CACHE / "lh.pial.ply"
RH_PLY     = _CACHE / "rh.pial.ply"


def _dl_progress(block_count: int, block_size: int, total: int) -> None:
    downloaded = block_count * block_size
    if total > 0:
        pct = min(100, 100 * downloaded / total)
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        click.echo(
            f"\r  [{bar}] {pct:5.1f}%  {downloaded/1e6:.1f}/{total/1e6:.1f} MB",
            nl=False,
        )


def download_mni_pial(verbose: bool = True) -> None:
    """Download and extract the Brain for Blender pial surfaces if not cached.

    Files are stored in ~/.cache/pyclarcs/ and reused on subsequent calls.
    Licence: Creative Commons Attribution 4.0 International (A. Winkler).
    """
    if LH_PLY.exists() and RH_PLY.exists():
        if verbose:
            click.echo(f"  Cache found: {_CACHE}")
        return

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not _ARCHIVE.exists():
        if verbose:
            click.echo("Downloading MNI pial surface from brainder.org…")
            click.echo(f"  {_MNI_URL}")
        urllib.request.urlretrieve(
            _MNI_URL, _ARCHIVE,
            reporthook=_dl_progress if verbose else None,
        )
        if verbose:
            click.echo("")

    if verbose:
        click.echo("Extracting archive…")
    result = subprocess.run(
        ["tar", "xjf", str(_ARCHIVE), "-C", str(_CACHE_DIR)],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Extraction failed: {result.stderr.decode()}\n"
            "Make sure 'tar' and 'bzip2' are installed."
        )
    if verbose:
        click.echo(f"  → {_CACHE}")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _make_ellipsoid(
    a: float,
    b: float,
    c: float,
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_lat: int = 100,
    n_lon: int = 180,
    noise_sigma: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, list[list[int]]]:
    """Return (points, triangles) for a triangulated ellipsoid.

    The half-axes are a (X), b (Y), c (Z).  A closed triangulation is built
    using a lat/lon grid; the poles are represented as fans.
    """
    rng = np.random.default_rng(seed)

    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 2)  # includes poles
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)

    pts: list[np.ndarray] = []
    for la in lat:
        for lo in lon:
            x = a * np.cos(la) * np.cos(lo)
            y = b * np.cos(la) * np.sin(lo)
            z = c * np.sin(la)
            pts.append([x, y, z])

    points = np.array(pts, dtype=float) + np.array(centre)
    if noise_sigma > 0:
        points += rng.standard_normal(points.shape) * noise_sigma

    faces: list[list[int]] = []
    total_rings = n_lat + 2

    for ring in range(total_rings - 1):
        for j in range(n_lon):
            j2 = (j + 1) % n_lon
            v00 = ring * n_lon + j
            v01 = ring * n_lon + j2
            v10 = (ring + 1) * n_lon + j
            v11 = (ring + 1) * n_lon + j2
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    return points, faces


def _rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation matrix: rotate `angle_deg` degrees around `axis`."""
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(angle_deg)
    K = np.array([
        [        0, -axis[2],  axis[1]],
        [ axis[2],         0, -axis[0]],
        [-axis[1],  axis[0],         0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _affine_perturb(
    points: np.ndarray,
    *,
    translation: tuple[float, float, float] = (10.0, 7.0, -5.0),
    rotation_axis: tuple[float, float, float] = (0.3, 0.8, 0.5),
    rotation_deg: float = 12.0,
    scale: float = 1.03,
) -> np.ndarray:
    """Apply a small rigid + uniform-scale perturbation.

    The rotation is applied about the cloud's centroid so that a "small"
    angle produces a correspondingly small displacement.

    Parameters
    ----------
    translation : (tx, ty, tz)  [mm]
    rotation_axis : arbitrary non-zero vector (normalised internally)
    rotation_deg : rotation angle [degrees]
    scale : uniform scale factor (1.0 = no change)
    """
    centroid = points.mean(axis=0)
    R = _rotation_matrix(np.array(rotation_axis, dtype=float), rotation_deg)
    centered = points - centroid
    return scale * (centered @ R.T) + centroid + np.array(translation)


def _smooth_deformation(
    points: np.ndarray,
    *,
    amplitude: float = 3.0,
    n_bumps: int = 8,
    seed: int = 99,
) -> np.ndarray:
    """Generate a smooth Gaussian-bump deformation field.

    Each of the `n_bumps` control points (randomly chosen among the surface
    vertices) produces a spatially-decaying displacement.  The influence
    radius is 15 % of the bounding-box diagonal, giving a smooth field that
    varies gradually across the surface.

    Parameters
    ----------
    amplitude : RMS magnitude of each bump displacement [mm]
    n_bumps   : number of independent Gaussian bumps
    seed      : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    N = len(points)
    bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    radius = bbox_diag * 0.15  # ≈ 15% of bounding-box diagonal

    field = np.zeros((N, 3), dtype=float)
    ctrl_idx = rng.choice(N, size=n_bumps, replace=False)
    ctrl_pts = points[ctrl_idx]

    for k in range(n_bumps):
        disp = rng.standard_normal(3) * amplitude          # random 3-D displacement
        dists = np.linalg.norm(points - ctrl_pts[k], axis=1)
        weights = np.exp(-(dists / radius) ** 2)            # Gaussian falloff
        field += weights[:, np.newaxis] * disp

    return field


def _load_ply(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    return load_surface(str(path))


def _build_polydata(points: np.ndarray, faces: list[list[int]]):
    """Build a vtkPolyData from numpy arrays."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(points.astype("float32"), deep=True))
    cells = vtk.vtkCellArray()
    for face in faces:
        cells.InsertNextCell(len(face))
        for idx in face:
            cells.InsertCellPoint(idx)
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_pts)
    poly.SetPolys(cells)
    return poly


def _polydata_to_arrays(out) -> tuple[np.ndarray, list[list[int]]]:
    """Extract (points, faces) from a vtkPolyData."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    pts = vtk_to_numpy(out.GetPoints().GetData()).astype(float)
    new_faces: list[list[int]] = []
    out.GetPolys().InitTraversal()
    id_list = vtk.vtkIdList()
    while out.GetPolys().GetNextCell(id_list):
        new_faces.append([id_list.GetId(i) for i in range(id_list.GetNumberOfIds())])
    return pts, new_faces


def _decimate_vtk(
    points: np.ndarray,
    faces: list[list[int]],
    target_n: int,
) -> tuple[np.ndarray, list[list[int]]]:
    """Decimate a mesh to approximately target_n vertices using VTK.

    Uses vtkQuadricDecimation (error-quadrics) which reliably hits high
    reduction ratios.  vtkDecimatePro with PreserveTopology caps out around
    50 % reduction on complex meshes like brain surfaces.
    """
    import vtk

    reduction = 1.0 - target_n / len(points)
    poly = _build_polydata(points, faces)

    deci = vtk.vtkQuadricDecimation()
    deci.SetInputData(poly)
    deci.SetTargetReduction(reduction)
    deci.Update()

    return _polydata_to_arrays(deci.GetOutput())


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def gen_mni_pial(out_dir: Path, verbose: bool = True) -> None:
    """Download (if needed) and convert the Brain for Blender pial surfaces to VTK."""
    download_mni_pial(verbose=verbose)

    if verbose:
        click.echo("Loading LH + RH pial surfaces…")
    lh_pts, lh_faces = _load_ply(LH_PLY)
    rh_pts, rh_faces = _load_ply(RH_PLY)

    offset = len(lh_pts)
    rh_faces_shifted = [[v + offset for v in f] for f in rh_faces]

    all_pts = np.vstack([lh_pts, rh_pts])
    all_faces = lh_faces + rh_faces_shifted

    out = out_dir / "endocranium_mni_pial.vtk"
    save_surface(str(out), all_pts, all_faces)
    if verbose:
        click.echo(f"  → {out.name}  ({len(all_pts):,} pts, {len(all_faces):,} faces)")

    if verbose:
        click.echo("Decimating to ~10 000 points…")
    dec_pts, dec_faces = _decimate_vtk(all_pts, all_faces, target_n=10_000)
    out2 = out_dir / "endocranium_mni_pial_10k.vtk"
    save_surface(str(out2), dec_pts, dec_faces)
    if verbose:
        click.echo(f"  → {out2.name}  ({len(dec_pts):,} pts, {len(dec_faces):,} faces)")


def gen_synthetic(out_dir: Path, verbose: bool = True) -> None:
    """Generate the three synthetic bilateral ellipsoid surfaces."""
    specs = [
        (
            "ellipsoid_skull_noisy.vtk",
            85.0, 65.0, 70.0, (85.0, 0.0, 0.0), 0.5, 0,
            "Skull-shaped ellipsoid (a=85 b=65 c=70 mm), true plane x=85",
        ),
        (
            "ellipsoid_a15_b60_c45.vtk",
            15.0, 60.0, 45.0, (5.0, 0.0, 0.0), 0.3, 1,
            "Unit-test ellipsoid (a=15 b=60 c=45 mm), true plane x=5",
        ),
        (
            "ellipsoid_oblate.vtk",
            40.0, 90.0, 90.0, (40.0, 0.0, 0.0), 0.4, 2,
            "Oblate ellipsoid (a=40 b=90 c=90 mm), true plane x=40",
        ),
    ]

    for fname, a, b, c, centre, sigma, seed, desc in specs:
        pts, faces = _make_ellipsoid(a, b, c, centre=centre,
                                     noise_sigma=sigma, seed=seed)
        out = out_dir / fname
        save_surface(str(out), pts, faces)
        if verbose:
            click.echo(f"  → {out.name}  ({len(pts):,} pts)  [{desc}]")


def gen_registration_samples(out_dir: Path, verbose: bool = True) -> None:
    """Generate ref/target pairs to test the registration pipeline.

    For each available reference surface a matching *target* is built by:
      1. Affine perturbation:
           translation  (10, 7, −5) mm
           rotation     12° around axis (0.3, 0.8, 0.5)
           scale        1.03  (3 % larger)
      2. Smooth non-rigid deformation:
           16 Gaussian bumps, amplitude 15 mm,
           influence radius 15 % of bounding-box diagonal.

    Output files (same polygon connectivity as the reference):
        <stem>_target.vtk

    Intended pipeline:
        clarcs normalize   <target>      --target <ref>
        clarcs nlregister  <normalized>  <ref>
    """
    refs = [
        "ellipsoid_skull_noisy.vtk",
        "endocranium_mni_pial_10k.vtk",
        "endocranium_mni_pial.vtk",
    ]

    for ref_fname in refs:
        ref_path = out_dir / ref_fname
        if not ref_path.exists():
            if verbose:
                click.echo(f"  SKIP {ref_fname} (not found)")
            continue

        pts, faces = load_surface(str(ref_path))

        # Step 1 — affine perturbation
        perturbed = _affine_perturb(pts)

        # Step 2 — smooth non-rigid deformation applied to the already-perturbed cloud
        field = _smooth_deformation(perturbed, amplitude=15.0, n_bumps=16)
        target_pts = perturbed + field

        stem = ref_path.stem
        out = out_dir / f"{stem}_target.vtk"
        save_surface(str(out), target_pts, faces)
        if verbose:
            click.echo(
                f"  {ref_fname}  →  {out.name}"
                f"  ({len(target_pts):,} pts)"
                f"  [t=(10,7,-5) mm | R=12° | s=1.03 | 16 bumps ×15 mm]"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--no-mni", is_flag=True, help="Skip MNI pial surfaces (skips download).")
@click.option("--no-reg", is_flag=True, help="Skip registration test pairs (*_target.vtk).")
@click.option("-q", "--quiet", is_flag=True, help="Suppress all output.")
def main(no_mni, no_reg, quiet):
    """Generate pyclarcs test surfaces (synthetic ellipsoids + MNI pial)."""
    verbose = not quiet
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(exist_ok=True)

    if verbose:
        click.echo("=== Generating pyclarcs test surfaces ===\n")

    if not no_mni:
        gen_mni_pial(out_dir, verbose=verbose)
    elif verbose:
        click.echo("Skipping MNI surfaces (--no-mni).")

    if verbose:
        click.echo("\nGenerating synthetic surfaces…")
    gen_synthetic(out_dir, verbose=verbose)

    if not no_reg:
        if verbose:
            click.echo("\nGenerating registration test pairs…")
        gen_registration_samples(out_dir, verbose=verbose)
    elif verbose:
        click.echo("Skipping registration pairs (--no-reg).")

    if verbose:
        click.echo("\nDone.")


if __name__ == "__main__":
    main()
