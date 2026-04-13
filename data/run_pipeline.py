#!/usr/bin/env python3
"""
run_pipeline.py
===============
Run the full clarcs registration pipeline on the test pairs produced by
generate_samples.py.

Pipeline (per pair)
-------------------
  1. symplane    — estimate the symmetry plane of each surface independently
  2. recenter    — align each surface's symmetry plane to x = 0
  3. normalize   — match the recentered target's size and centre-of-mass to
                   the recentered reference
  4. nlregister  — non-rigid EM-ICP to warp the normalized target onto the
                   recentered reference

All intermediate and final surfaces are written to OUTPUT_DIR.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import click
import numpy as np

# ---------------------------------------------------------------------------
# Make pyclarcs importable from source tree
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # pyclarcs/data/
_SRC  = _HERE.parent / "src"                     # pyclarcs/src/
if _SRC.exists():
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Available test pairs  (ref stem → target file stem)
# ---------------------------------------------------------------------------
_PAIRS = {
    "ellipsoid_skull_noisy":       "ellipsoid_skull_noisy_target",
    "endocranium_mni_pial_10k":    "endocranium_mni_pial_10k_target",
    "endocranium_mni_pial":        "endocranium_mni_pial_target",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _invoke(cli, args: list[str]) -> "click.testing.Result":
    """Invoke a CLI command and always return the result (never raise)."""
    from click.testing import CliRunner
    return CliRunner().invoke(cli, args, catch_exceptions=True)


def _check(result, step: str) -> None:
    """Raise SystemExit with full diagnostics if the step failed."""
    if result.exit_code == 0:
        return
    click.echo(f"  [ERROR] {step} failed (exit {result.exit_code})", err=True)
    if result.output:
        click.echo(result.output, err=True, nl=False)
    if result.exception is not None:
        lines = traceback.format_exception(
            type(result.exception), result.exception,
            result.exception.__traceback__,
        )
        click.echo("".join(lines), err=True, nl=False)
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _step_symplane(
    surface_path: Path, out_dir: Path, verbose: bool
) -> tuple[Path, Path]:
    """Estimate the symmetry plane of *surface_path*.

    Returns
    -------
    (vtk_path, pl_path)
        Path to the saved plane patch (.vtk) and the plane parameters (.pl).
    """
    from pyclarcs._cli import cli

    out_path = out_dir / (surface_path.stem + "-symplane.vtk")
    # --save-plane writes <out_stem>.pl alongside the vtk
    args = ["symplane", str(surface_path), str(out_path), "--save-plane"]
    if not verbose:
        args.append("-q")

    t0 = time.perf_counter()
    result = _invoke(cli, args)
    elapsed = time.perf_counter() - t0

    _check(result, f"symplane({surface_path.name})")
    if verbose:
        click.echo(result.output, nl=False)
    pl_path = out_path.with_suffix(".pl")
    click.echo(f"  symplane   → {out_path.name}  ({elapsed:.1f} s)")
    return out_path, pl_path


def _step_recenter(
    surface_path: Path, pl_path: Path, out_dir: Path, verbose: bool
) -> Path:
    """Align *surface_path* to the canonical symmetry plane (x = 0)."""
    from pyclarcs._cli import cli

    out_path = out_dir / (surface_path.stem + "-recentered.vtk")
    args = ["recenter", str(surface_path), str(out_path), "--plane", str(pl_path)]
    if not verbose:
        args.append("-q")

    t0 = time.perf_counter()
    result = _invoke(cli, args)
    elapsed = time.perf_counter() - t0

    _check(result, f"recenter({surface_path.name})")
    if verbose:
        click.echo(result.output, nl=False)
    click.echo(f"  recenter   → {out_path.name}  ({elapsed:.1f} s)")
    return out_path


def _step_normalize(
    target_path: Path, ref_path: Path, out_dir: Path, verbose: bool
) -> Path:
    """Match *target_path* centre-of-mass and scale to *ref_path*."""
    from pyclarcs._cli import cli

    out_path = out_dir / (target_path.stem + "-normalized.vtk")
    args = ["normalize", str(target_path), str(out_path), "--target", str(ref_path)]
    if not verbose:
        args.append("-q")

    t0 = time.perf_counter()
    result = _invoke(cli, args)
    elapsed = time.perf_counter() - t0

    _check(result, f"normalize({target_path.name})")
    if verbose:
        click.echo(result.output, nl=False)
    click.echo(f"  normalize  → {out_path.name}  ({elapsed:.1f} s)")
    return out_path


def _step_nlregister(
    normalized_path: Path,
    ref_path: Path,
    out_dir: Path,
    reg_kwargs: dict,
    verbose: bool,
) -> Path:
    """Non-rigid EM-ICP registration of *normalized_path* onto *ref_path*."""
    from pyclarcs._cli import cli

    out_path = out_dir / (normalized_path.stem + "-nlregistered.vtk")
    def_path = out_dir / (normalized_path.stem + "-deformation.vtk")

    args = [
        "nlregister",
        str(normalized_path),
        str(ref_path),
        str(out_path),
        "--deformation", str(def_path),
        "--beta",         str(reg_kwargs["beta"]),
        "--max-iter",     str(reg_kwargs["max_iter"]),
        "--icm-iter",     str(reg_kwargs["icm_iter"]),
    ]
    # Only pass auto-estimable params when the user explicitly overrode them.
    if reg_kwargs["sigma"] is not None:
        args += ["--sigma", str(reg_kwargs["sigma"])]
    if reg_kwargs["dist_cutoff"] is not None:
        args += ["--dist-cutoff", str(reg_kwargs["dist_cutoff"])]
    if reg_kwargs["period_sigma"] is not None:
        args += ["--period-sigma", str(reg_kwargs["period_sigma"])]
    args += ["--n-levels",          str(reg_kwargs["n_levels"])]
    args += ["--coarsest-n",        str(reg_kwargs["coarsest_n"])]
    args += ["--beta-coarse-factor", str(reg_kwargs["beta_coarse_factor"])]
    if not verbose:
        args.append("-q")

    t0 = time.perf_counter()
    result = _invoke(cli, args)
    elapsed = time.perf_counter() - t0

    _check(result, f"nlregister({normalized_path.name})")
    if verbose:
        click.echo(result.output, nl=False)
    click.echo(f"  nlregister → {out_path.name}  ({elapsed:.1f} s)")
    click.echo(f"  deformation→ {def_path.name}")
    return out_path


def _rms_nn_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """RMS nearest-neighbour distance from pts_a to pts_b."""
    from scipy.spatial import KDTree
    dists, _ = KDTree(pts_b).query(pts_a, k=1, workers=-1)
    return float(np.sqrt(np.mean(dists ** 2)))


def _print_summary(ref_path: Path, before_path: Path, registered_path: Path) -> None:
    """Print NN-distance RMS before/after registration.

    Uses nearest-neighbour distances so the metric is valid even when
    the reference and target meshes have different vertex counts or
    orderings (which is the usual case for inter-subject surfaces).
    """
    from pyclarcs.io import load_surface

    ref_pts,  _ = load_surface(str(ref_path))
    bef_pts,  _ = load_surface(str(before_path))
    reg_pts,  _ = load_surface(str(registered_path))

    before = _rms_nn_distance(bef_pts, ref_pts)
    after  = _rms_nn_distance(reg_pts, ref_pts)
    click.echo(
        f"  RMS NN before: {before:.2f} mm   after: {after:.2f} mm"
        f"   improvement: {(before - after) / before * 100:.1f} %"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("output_dir", metavar="OUTPUT_DIR",
                type=click.Path(file_okay=False, writable=True))
@click.option(
    "--pairs", multiple=True,
    type=click.Choice(list(_PAIRS)),
    metavar="STEM",
    help=(
        "Test pair(s) to process. "
        f"Choices: {', '.join(_PAIRS)}. "
        "Default: all. Repeat to select several."
    ),
)
@click.option("--no-nlregister", is_flag=True,
              help="Stop after normalize (skip the EM-ICP step).")
@click.option("--sigma",        default=None,  type=float,
              help="Initial bandwidth [mm] (auto-estimated if omitted).")
@click.option("--beta",         default=10.0,  show_default=True, type=float,
              help="Regularisation weight (higher = smoother).")
@click.option("--dist-cutoff",  default=None,  type=float,
              help="Search radius [mm] (auto-estimated if omitted).")
@click.option("--max-iter",     default=80,    show_default=True, type=int,
              help="Number of outer EM iterations.")
@click.option("--icm-iter",     default=50,    show_default=True, type=int,
              help="Max conjugate gradient iterations per outer iteration.")
@click.option("--period-sigma", default=None,  type=int,
              help="Halve sigma every N iterations (auto-estimated if omitted).")
@click.option("--n-levels",          default=3,    show_default=True, type=int,
              help="Resolution levels for multi-res registration (1 = single-res).")
@click.option("--coarsest-n",        default=2000, show_default=True, type=int,
              help="Target vertices at the coarsest level.")
@click.option("--beta-coarse-factor", default=3.0, show_default=True, type=float,
              help="Per-level beta multiplier toward coarser levels (multi-res only).")
@click.option("-q", "--quiet",  is_flag=True, help="Suppress all output.")
def main(
    output_dir, pairs, no_nlregister,
    sigma, beta, dist_cutoff, max_iter, icm_iter, period_sigma,
    n_levels, coarsest_n, beta_coarse_factor,
    quiet,
):
    """Run the clarcs pipeline (symplane → recenter → normalize → nlregister).

    Both the reference and the target surfaces are independently recentered
    to their own symmetry plane before the non-rigid registration step.
    """
    verbose = not quiet
    selected = list(pairs) if pairs else list(_PAIRS)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reg_kwargs = {
        "sigma":              sigma,
        "beta":               beta,
        "beta_coarse_factor": beta_coarse_factor,
        "dist_cutoff":        dist_cutoff,
        "max_iter":           max_iter,
        "icm_iter":           icm_iter,
        "period_sigma":       period_sigma,
        "n_levels":           n_levels,
        "coarsest_n":         coarsest_n,
    }

    t_global = time.perf_counter()

    for ref_stem in selected:
        tgt_stem  = _PAIRS[ref_stem]
        ref_path  = _HERE / f"{ref_stem}.vtk"
        tgt_path  = _HERE / f"{tgt_stem}.vtk"

        if not ref_path.exists():
            click.echo(f"\nSKIP {ref_stem}: reference not found ({ref_path})")
            continue
        if not tgt_path.exists():
            click.echo(f"\nSKIP {ref_stem}: target not found ({tgt_path})")
            click.echo("  Run  python generate_samples.py  first.")
            continue

        sep = "─" * 60
        click.echo(f"\n{sep}")
        click.echo(f"  Pair : {ref_stem}")
        click.echo(f"  ref  : {ref_path.name}")
        click.echo(f"  tgt  : {tgt_path.name}")
        click.echo(f"  out  : {output_dir}/")
        click.echo(sep)

        # Step 1 — symmetry planes (ref and target independently)
        _, ref_pl  = _step_symplane(ref_path,  output_dir, verbose)
        _, tgt_pl  = _step_symplane(tgt_path,  output_dir, verbose)

        # Step 2 — recenter both surfaces to x = 0
        ref_recentered = _step_recenter(ref_path, ref_pl,  output_dir, verbose)
        tgt_recentered = _step_recenter(tgt_path, tgt_pl,  output_dir, verbose)

        # Step 3 — normalize recentered target to match recentered reference
        tgt_normalized = _step_normalize(
            tgt_recentered, ref_recentered, output_dir, verbose
        )

        if not no_nlregister:
            # Step 4 — register normalized target onto recentered reference
            registered = _step_nlregister(
                tgt_normalized, ref_recentered, output_dir, reg_kwargs, verbose
            )
            _print_summary(ref_recentered, tgt_normalized, registered)

    elapsed_total = time.perf_counter() - t_global
    click.echo(f"\nTotal time: {elapsed_total:.1f} s")
    click.echo(f"Results in: {output_dir}")


if __name__ == "__main__":
    main()
