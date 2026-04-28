#!/usr/bin/env python3
"""
generate_synth_data.py
======================
Generate synthetic test surfaces for pyclarcs.

Surfaces produced
-----------------
1. ellipsoid_skull_noisy.vtk
   Synthetic skull-shaped ellipsoid (a=85, b=65, c=70 mm), centred at
   (85, 0, 0) mm with σ=0.5 mm Gaussian noise.  True symmetry plane: x=85.

2. ellipsoid_a15_b60_c45.vtk
   Unit-test ellipsoid (a=15, b=60, c=45 mm), centred at (5, 0, 0) mm,
   σ=0.3 mm noise.  True symmetry plane: x=5.

3. ellipsoid_oblate.vtk
   Oblate ellipsoid (a=40, b=90, c=90 mm), centred at (40, 0, 0) mm,
   σ=0.4 mm noise.  True symmetry plane: x=40.

4. *_target.vtk (one per available reference surface)
   Registration test pairs.  Each target is the corresponding reference
   surface after applying:
     (a) Affine perturbation: translation (10, 7, −5) mm, rotation 12°
         around axis (0.3, 0.8, 0.5), uniform scale 1.03.
     (b) Smooth non-rigid deformation: 8 Gaussian bumps of amplitude 5 mm.
   Run download_mni.py first to also generate MNI-based pairs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from pyclarcs.io import load_surface, save_surface  # noqa: E402

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
    rng = np.random.default_rng(seed)

    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 2)
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
    rng = np.random.default_rng(seed)
    N = len(points)
    bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    radius = bbox_diag * 0.15

    field = np.zeros((N, 3), dtype=float)
    ctrl_idx = rng.choice(N, size=n_bumps, replace=False)
    ctrl_pts = points[ctrl_idx]

    for k in range(n_bumps):
        disp = rng.standard_normal(3) * amplitude
        dists = np.linalg.norm(points - ctrl_pts[k], axis=1)
        weights = np.exp(-(dists / radius) ** 2)
        field += weights[:, np.newaxis] * disp

    return field


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

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
           scale        1.03
      2. Smooth non-rigid deformation:
           8 Gaussian bumps, amplitude 5 mm.

    MNI-based pairs are skipped if the MNI surfaces are not present — run
    download_mni.py first to generate them.
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

        perturbed = _affine_perturb(pts)
        field = _smooth_deformation(perturbed, amplitude=5.0, n_bumps=8)
        target_pts = perturbed + field

        stem = ref_path.stem
        out = out_dir / f"{stem}_target.vtk"
        save_surface(str(out), target_pts, faces)
        if verbose:
            click.echo(
                f"  {ref_fname}  →  {out.name}"
                f"  ({len(target_pts):,} pts)"
                f"  [t=(10,7,-5) mm | R=12° | s=1.03 | 8 bumps ×5 mm]"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--no-reg", is_flag=True,
              help="Skip registration test pairs (*_target.vtk).")
@click.option("-q", "--quiet", is_flag=True, help="Suppress all output.")
def main(no_reg, quiet):
    """Generate synthetic pyclarcs test surfaces.

    Produces three ellipsoid surfaces and, unless --no-reg, matching
    *_target.vtk registration pairs for all available reference surfaces.
    Run download_mni.py first to also generate MNI-based pairs.
    """
    verbose = not quiet
    out_dir = _HERE

    if verbose:
        click.echo("Generating synthetic surfaces…")
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
