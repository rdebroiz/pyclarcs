#!/usr/bin/env python3
"""
run_pipeline.py
===============
Run the full clarcs registration pipeline on the test pairs produced by
generate_samples.py.

Pipeline (per pair)
-------------------
  1. recenter    — align the target's symmetry plane to x = 0
  2. normalize   — match the recentered target's size and centre-of-mass to ref
  3. nlregister  — non-rigid EM-ICP to warp the normalized target onto ref

All intermediate and final surfaces are written to OUTPUT_DIR.

Usage
-----
    python run_pipeline.py OUTPUT_DIR [options]

Examples
--------
    python run_pipeline.py results/
    python run_pipeline.py results/ --pairs ellipsoid_skull_noisy
    python run_pipeline.py results/ --no-nlregister   # only recenter + normalize
    python run_pipeline.py results/ -q                # quiet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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

def _step_recenter(
    target_path: Path,
    out_dir: Path,
    verbose: bool,
) -> Path:
    """Step 1 — estimate symmetry plane and align target to x = 0."""
    from pyclarcs._cli import cli
    from click.testing import CliRunner

    out_path  = out_dir / (target_path.stem + "-recentered.vtk")
    pl_path   = out_dir / (target_path.stem + "-recentered.pl")

    args = [
        "recenter",
        str(target_path),
        str(out_path),
        "--save-plane",
    ]
    if not verbose:
        args.append("-q")

    t0 = time.perf_counter()
    result = CliRunner().invoke(cli, args)
    elapsed = time.perf_counter() - t0

    if result.exit_code != 0:
        print(f"  [ERROR] recenter failed:\n{result.output}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(result.output, end="")
    print(f"  recenter   → {out_path.name}  ({elapsed:.1f} s)")
    return out_path


def _step_normalize(
    recentered_path: Path,
    ref_path: Path,
    out_dir: Path,
    verbose: bool,
) -> Path:
    """Step 2 — match size and centre-of-mass to reference."""
    from pyclarcs._cli import cli
    from click.testing import CliRunner

    out_path = out_dir / (recentered_path.stem + "-normalized.vtk")

    args = [
        "normalize",
        str(recentered_path),
        str(out_path),
        "--target", str(ref_path),
    ]
    if not verbose:
        args.append("-q")

    t0 = time.perf_counter()
    result = CliRunner().invoke(cli, args)
    elapsed = time.perf_counter() - t0

    if result.exit_code != 0:
        print(f"  [ERROR] normalize failed:\n{result.output}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(result.output, end="")
    print(f"  normalize  → {out_path.name}  ({elapsed:.1f} s)")
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

    out_path  = out_dir / (normalized_path.stem + "-nlregistered.vtk")
    def_path  = out_dir / (normalized_path.stem + "-deformation.vtk")

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
        print(f"  [ERROR] nlregister failed:\n{result.output}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(result.output, end="")
    print(f"  nlregister → {out_path.name}  ({elapsed:.1f} s)")
    print(f"  deformation→ {def_path.name}")
    return out_path


def _rms_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Mean point-to-point distance between two same-size point clouds."""
    return float(np.sqrt(np.mean(np.sum((pts_a - pts_b) ** 2, axis=1))))


def _print_summary(ref_path: Path, target_path: Path, registered_path: Path) -> None:
    """Print before/after RMS distances to quantify registration quality."""
    from pyclarcs.io import load_surface

    ref_pts, _    = load_surface(str(ref_path))
    tgt_pts, _    = load_surface(str(target_path))
    reg_pts, _    = load_surface(str(registered_path))

    # Align point counts for comparison (clouds may differ after VTK clean)
    n = min(len(ref_pts), len(tgt_pts), len(reg_pts))
    before = _rms_distance(ref_pts[:n], tgt_pts[:n])
    after  = _rms_distance(ref_pts[:n], reg_pts[:n])
    print(f"  RMS before: {before:.2f} mm   after: {after:.2f} mm"
          f"   improvement: {(before - after) / before * 100:.1f} %")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "output_dir", metavar="OUTPUT_DIR",
        help="Directory where results are written (created if absent).",
    )
    parser.add_argument(
        "--pairs", nargs="+", metavar="STEM",
        choices=list(_PAIRS),
        default=list(_PAIRS),
        help=(
            "Which test pairs to process. "
            f"Choices: {', '.join(_PAIRS)}. "
            "Default: all available."
        ),
    )
    parser.add_argument(
        "--no-nlregister", action="store_true",
        help="Stop after normalize (skip the EM-ICP step).",
    )
    # Registration parameters
    parser.add_argument("--sigma",        type=float, default=3.0,   metavar="F")
    parser.add_argument("--beta",         type=float, default=100.0, metavar="F")
    parser.add_argument("--dist-cutoff",  type=float, default=15.0,  metavar="F")
    parser.add_argument("--max-iter",     type=int,   default=80,    metavar="N")
    parser.add_argument("--icm-iter",     type=int,   default=120,   metavar="N")
    parser.add_argument("--period-sigma", type=int,   default=40,    metavar="N")
    parser.add_argument("-q", "--quiet",  action="store_true")

    args = parser.parse_args(argv)
    verbose = not args.quiet

    data_dir   = _HERE
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reg_kwargs = {
        "sigma":        args.sigma,
        "beta":         args.beta,
        "dist_cutoff":  args.dist_cutoff,
        "max_iter":     args.max_iter,
        "icm_iter":     args.icm_iter,
        "period_sigma": args.period_sigma,
    }

    t_global = time.perf_counter()

    for ref_stem in args.pairs:
        tgt_stem  = _PAIRS[ref_stem]
        ref_path  = data_dir / f"{ref_stem}.vtk"
        tgt_path  = data_dir / f"{tgt_stem}.vtk"

        if not ref_path.exists():
            print(f"\nSKIP {ref_stem}: reference not found ({ref_path})")
            continue
        if not tgt_path.exists():
            print(f"\nSKIP {ref_stem}: target not found ({tgt_path})")
            print("  Run  python generate_samples.py  first.")
            continue

        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  Pair : {ref_stem}")
        print(f"  ref  : {ref_path.name}")
        print(f"  tgt  : {tgt_path.name}")
        print(f"  out  : {output_dir}/")
        print(sep)

        # -- Step 1 : recenter
        recentered = _step_recenter(tgt_path, output_dir, verbose)

        # -- Step 2 : normalize
        normalized = _step_normalize(recentered, ref_path, output_dir, verbose)

        # -- Step 3 : nlregister (optional)
        if not args.no_nlregister:
            registered = _step_nlregister(
                normalized, ref_path, output_dir, reg_kwargs, verbose
            )
            _print_summary(ref_path, tgt_path, registered)

    elapsed_total = time.perf_counter() - t_global
    print(f"\nTotal time: {elapsed_total:.1f} s")
    print(f"Results in: {output_dir}")


if __name__ == "__main__":
    main()
