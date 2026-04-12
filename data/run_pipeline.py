#!/usr/bin/env python3
"""
run_pipeline.py
===============
Run the full clarcs registration pipeline on the test pairs produced by
generate_samples.py.

Pipeline (per pair)
-------------------
  1. normalize   — match the target's size and centre-of-mass to ref
  2. nlregister  — non-rigid EM-ICP to warp the normalized target onto ref

All intermediate and final surfaces are written to OUTPUT_DIR.
"""

from __future__ import annotations

import sys
import time
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
# Pipeline steps
# ---------------------------------------------------------------------------

def _step_normalize(
    target_path: Path, ref_path: Path, out_dir: Path, verbose: bool
) -> Path:
    """Step 1 — match size and centre-of-mass to reference."""
    from pyclarcs._cli import cli
    from click.testing import CliRunner

    out_path = out_dir / (target_path.stem + "-normalized.vtk")
    args = ["normalize", str(target_path), str(out_path), "--target", str(ref_path)]
    if not verbose:
        args.append("-q")

    t0 = time.perf_counter()
    result = CliRunner().invoke(cli, args)
    elapsed = time.perf_counter() - t0

    if result.exit_code != 0:
        click.echo(f"  [ERROR] normalize failed:\n{result.output}", err=True)
        raise SystemExit(1)

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
    """Step 3 — non-rigid EM-ICP registration."""
    from pyclarcs._cli import cli
    from click.testing import CliRunner

    out_path = out_dir / (normalized_path.stem + "-nlregistered.vtk")
    def_path = out_dir / (normalized_path.stem + "-deformation.vtk")

    args = [
        "nlregister",
        str(normalized_path),
        str(ref_path),
        str(out_path),
        "--deformation", str(def_path),
        "--sigma",        str(reg_kwargs["sigma"]),
        "--beta",         str(reg_kwargs["beta"]),
        "--dist-cutoff",  str(reg_kwargs["dist_cutoff"]),
        "--max-iter",     str(reg_kwargs["max_iter"]),
        "--icm-iter",     str(reg_kwargs["icm_iter"]),
        "--period-sigma", str(reg_kwargs["period_sigma"]),
    ]
    if not verbose:
        args.append("-q")

    t0 = time.perf_counter()
    result = CliRunner().invoke(cli, args)
    elapsed = time.perf_counter() - t0

    if result.exit_code != 0:
        click.echo(f"  [ERROR] nlregister failed:\n{result.output}", err=True)
        raise SystemExit(1)

    if verbose:
        click.echo(result.output, nl=False)
    click.echo(f"  nlregister → {out_path.name}  ({elapsed:.1f} s)")
    click.echo(f"  deformation→ {def_path.name}")
    return out_path


def _rms_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((pts_a - pts_b) ** 2, axis=1))))


def _print_summary(ref_path: Path, target_path: Path, registered_path: Path) -> None:
    from pyclarcs.io import load_surface

    ref_pts, _ = load_surface(str(ref_path))
    tgt_pts, _ = load_surface(str(target_path))
    reg_pts, _ = load_surface(str(registered_path))

    n = min(len(ref_pts), len(tgt_pts), len(reg_pts))
    before = _rms_distance(ref_pts[:n], tgt_pts[:n])
    after  = _rms_distance(ref_pts[:n], reg_pts[:n])
    click.echo(
        f"  RMS before: {before:.2f} mm   after: {after:.2f} mm"
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
@click.option("--sigma",        default=3.0,   show_default=True, type=float,
              help="Initial bandwidth of the correspondence kernel.")
@click.option("--beta",         default=100.0, show_default=True, type=float,
              help="Regularisation weight (higher = smoother).")
@click.option("--dist-cutoff",  default=25.0,  show_default=True, type=float,
              help="Maximum search radius for correspondences.")
@click.option("--max-iter",     default=80,    show_default=True, type=int,
              help="Number of outer EM iterations.")
@click.option("--icm-iter",     default=120,   show_default=True, type=int,
              help="Number of Jacobi ICM steps per outer iteration.")
@click.option("--period-sigma", default=40,    show_default=True, type=int,
              help="Halve sigma every this many iterations.")
@click.option("-q", "--quiet",  is_flag=True, help="Suppress all output.")
def main(
    output_dir, pairs, no_nlregister,
    sigma, beta, dist_cutoff, max_iter, icm_iter, period_sigma,
    quiet,
):
    """Run the clarcs pipeline (normalize → nlregister) on test pairs."""
    verbose = not quiet
    selected = list(pairs) if pairs else list(_PAIRS)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reg_kwargs = {
        "sigma":        sigma,
        "beta":         beta,
        "dist_cutoff":  dist_cutoff,
        "max_iter":     max_iter,
        "icm_iter":     icm_iter,
        "period_sigma": period_sigma,
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

        normalized = _step_normalize(tgt_path, ref_path, output_dir, verbose)

        if not no_nlregister:
            registered = _step_nlregister(
                normalized, ref_path, output_dir, reg_kwargs, verbose
            )
            _print_summary(ref_path, tgt_path, registered)

    elapsed_total = time.perf_counter() - t_global
    click.echo(f"\nTotal time: {elapsed_total:.1f} s")
    click.echo(f"Results in: {output_dir}")


if __name__ == "__main__":
    main()
