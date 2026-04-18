#!/usr/bin/env python3
"""
benchmark_compare.py
====================
Compare non-rigid registration methods on pyclarcs test pairs.

Methods benchmarked
-------------------
1. clarcs  — EM-ICP with Gaussian kernel + Laplacian regularisation (this repo)
2. cpd     — Coherent Point Drift (non-rigid), via pycpd
3. bcpd    — Bayesian Coherent Point Drift, via the ohirose/bcpd binary

Usage
-----
    python benchmark_compare.py [OUTPUT_DIR] [options]

    --pairs STEM      Test pair(s) to run (default: all available)
    --methods M [M…]  Methods to include: clarcs cpd bcpd (default: all available)
    --max-iter N      Outer iterations for clarcs and CPD (default: 80)
    --no-preprocess   Skip symplane/recenter/normalize — use raw test pairs
    -q / --quiet      Suppress per-iteration output
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import click
import numpy as np
from scipy.spatial import KDTree

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rms_nn(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """One-sided RMS nearest-neighbour distance from pts_a to pts_b."""
    d, _ = KDTree(pts_b).query(pts_a, k=1, workers=-1)
    return float(np.sqrt(np.mean(d ** 2)))


def symmetric_rms_nn(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Symmetric RMS NN: mean of both directions."""
    return (rms_nn(pts_a, pts_b) + rms_nn(pts_b, pts_a)) / 2.0


# ---------------------------------------------------------------------------
# Method wrappers
# ---------------------------------------------------------------------------

def run_clarcs(
    mov_pts: np.ndarray,
    mov_normals: np.ndarray,
    ref_pts: np.ndarray,
    ref_normals: np.ndarray,
    mov_polygons: list,
    max_iter: int = 80,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """Run clarcs EM-ICP.  Returns (warped_pts, elapsed_seconds)."""
    from pyclarcs.nonrigid import nonrigid_icp_multires, apply_deformation
    from pyclarcs.mesh import adjacency_csr
    from pyclarcs.nonrigid import estimate_registration_params

    t0 = time.perf_counter()
    def_field = nonrigid_icp_multires(
        mov_pts, mov_normals,
        ref_pts, ref_normals,
        mov_polygons,
        max_iter=max_iter,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - t0
    return apply_deformation(mov_pts, def_field), elapsed


def run_cpd(
    mov_pts: np.ndarray,
    ref_pts: np.ndarray,
    max_iter: int = 80,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """Run CPD non-rigid (pycpd).  Returns (warped_pts, elapsed_seconds)."""
    try:
        from pycpd import DeformableRegistration
    except ImportError:
        raise ImportError("pycpd not installed — run: pip install pycpd")

    def callback(iteration, error, X, Y):
        if verbose:
            click.echo(f"  [CPD] iter {iteration:3d}  err={error:.4f}")

    t0 = time.perf_counter()
    reg = DeformableRegistration(
        X=ref_pts.astype(np.float64),
        Y=mov_pts.astype(np.float64),
        max_iterations=max_iter,
        tolerance=1e-5,
    )
    warped, _ = reg.register(callback if verbose else None)
    elapsed = time.perf_counter() - t0
    return warped.astype(float), elapsed


def run_bcpd(
    mov_pts: np.ndarray,
    ref_pts: np.ndarray,
    bcpd_bin: str | None = None,
    max_iter: int = 300,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """Run BCPD via the ohirose/bcpd C++ binary.

    Parameters tuned for anatomical surfaces (pre-aligned, ~mm scale):
      -w 0.1      10 % outlier probability (consistent with clarcs)
      -l 2        lambda: expected deformation = sqrt(3/2) ≈ 1.2 in normalised units
      -b 2        beta: Gaussian kernel bandwidth in normalised units
      -g 0.1      gamma: small because shapes are already pre-aligned
      -J 300      Nyström rank for P  ─┐
      -K 70       Nyström rank for G  ─┤ mandatory acceleration for N > 2000
      -p          KD-tree search      ─┘
      -u e        normalise both point sets to unit scale (bcpd default)
      -c 1e-6     tight convergence tolerance
      -n <N>      max VB iterations

    Build the binary from https://github.com/ohirose/bcpd.
    """
    import subprocess, tempfile

    bin_path = bcpd_bin or "bcpd"
    if not Path(bin_path).exists() and not __import__("shutil").which(bin_path):
        raise FileNotFoundError(
            f"BCPD binary '{bin_path}' not found on PATH.  "
            "Build from https://github.com/ohirose/bcpd."
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src_f = tmp / "Y.txt"   # source = moving (bcpd convention: -y)
        tgt_f = tmp / "X.txt"   # target = reference (bcpd convention: -x)

        np.savetxt(src_f, mov_pts, fmt="%.6f")
        np.savetxt(tgt_f, ref_pts, fmt="%.6f")

        cmd = [
            bin_path,
            "-x", "X.txt",
            "-y", "Y.txt",
            "-o", "output_",
            "-w", "0.1",        # outlier weight
            "-l", "2",          # lambda
            "-b", "2",          # beta (kernel bandwidth)
            "-g", "0.1",        # gamma (low = shapes already aligned)
            "-J", "300",        # Nystrom rank P
            "-K", "70",         # Nystrom rank G
            "-p",               # KD-tree search
            "-u", "e",          # normalize both point sets
            "-c", "1e-6",       # convergence tolerance
            "-n", str(max_iter),
            "-s", "y",          # save only deformed source
        ]
        if not verbose:
            cmd += ["-q"]

        t0 = time.perf_counter()
        result = subprocess.run(
            cmd, capture_output=not verbose, text=True, cwd=str(tmp)
        )
        elapsed = time.perf_counter() - t0

        if result.returncode != 0:
            raise RuntimeError(
                f"BCPD failed (exit {result.returncode}):\n"
                f"{result.stderr or result.stdout or '(no output)'}"
            )

        out_file = tmp / "output_y.txt"
        if not out_file.exists():
            candidates = list(tmp.glob("*_y.txt")) + list(tmp.glob("output_*.txt"))
            if not candidates:
                raise FileNotFoundError(
                    f"BCPD output not found in {tmp}: {list(tmp.iterdir())}"
                )
            out_file = candidates[0]

        warped = np.loadtxt(out_file)
        return warped.astype(float), elapsed


# ---------------------------------------------------------------------------
# Pre-processing pipeline (symplane → recenter → normalize)
# ---------------------------------------------------------------------------

def _preprocess_pair(
    ref_path: Path,
    tgt_path: Path,
    work_dir: Path,
    verbose: bool,
) -> tuple[Path, Path]:
    """Run symplane+recenter on both surfaces, then normalize target.

    Returns (ref_recentered_path, tgt_normalized_path).
    """
    from pyclarcs._cli import cli
    from click.testing import CliRunner

    def invoke(args):
        r = CliRunner().invoke(cli, args, catch_exceptions=True)
        if r.exit_code != 0:
            tb_str = ""
            if r.exception is not None:
                tb_str = "".join(traceback.format_exception(
                    type(r.exception), r.exception, r.exception.__traceback__))
            raise RuntimeError(f"Step {args[0]} failed:\n{r.output}\n{tb_str}")
        return r

    work_dir.mkdir(parents=True, exist_ok=True)

    q = ["-q"] if not verbose else []

    # symplane + recenter for ref
    ref_sym_vtk = work_dir / (ref_path.stem + "-sym.vtk")
    ref_sym     = ref_sym_vtk.with_suffix(".pl")
    invoke(["symplane", str(ref_path), str(ref_sym_vtk), "--save-plane"] + q)
    ref_rc = work_dir / (ref_path.stem + "-recentered.vtk")
    invoke(["recenter", str(ref_path), str(ref_rc), "--plane", str(ref_sym)] + q)

    # symplane + recenter for target
    tgt_sym_vtk = work_dir / (tgt_path.stem + "-sym.vtk")
    tgt_sym     = tgt_sym_vtk.with_suffix(".pl")
    invoke(["symplane", str(tgt_path), str(tgt_sym_vtk), "--save-plane"] + q)
    tgt_rc = work_dir / (tgt_path.stem + "-recentered.vtk")
    invoke(["recenter", str(tgt_path), str(tgt_rc), "--plane", str(tgt_sym)] + q)

    # normalize target to match recentered reference
    tgt_norm = work_dir / (tgt_path.stem + "-recentered-normalized.vtk")
    invoke(["normalize", str(tgt_rc), str(tgt_norm), "--target", str(ref_rc)] + q)

    return ref_rc, tgt_norm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_PAIRS = {
    "ellipsoid_skull_noisy":    "ellipsoid_skull_noisy_target",
    "endocranium_mni_pial_10k": "endocranium_mni_pial_10k_target",
}

_METHODS = ["clarcs", "cpd", "bcpd"]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("output_dir", default="results_compare",
                metavar="OUTPUT_DIR", type=click.Path(file_okay=False, writable=True))
@click.option("--pairs", multiple=True, type=click.Choice(list(_PAIRS)),
              help="Test pair(s) to run (default: all).")
@click.option("--methods", multiple=True, type=click.Choice(_METHODS),
              help="Methods to benchmark (default: all available).")
@click.option("--max-iter",  default=80, show_default=True, type=int,
              help="Outer iterations for clarcs and CPD.")
@click.option("--no-preprocess", is_flag=True,
              help="Skip symplane/recenter/normalize — use raw test pairs.")
@click.option("--bcpd-bin", default="bcpd", show_default=True,
              help="Path to the BCPD binary (ohirose/bcpd).")
@click.option("-q", "--quiet", is_flag=True, help="Suppress per-iteration output.")
def main(output_dir, pairs, methods, max_iter, no_preprocess, bcpd_bin, quiet):
    """Compare non-rigid registration methods: clarcs, CPD, BCPD.

    Runs each method on the selected test pairs (pre-processed via the clarcs
    pipeline unless --no-preprocess) and reports RMS nearest-neighbour
    distances before and after registration.
    """
    from pyclarcs.io import load_surface, load_surface_with_normals

    verbose = not quiet
    selected_pairs   = list(pairs)   or list(_PAIRS)
    selected_methods = list(methods) or _METHODS

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 68
    results: list[dict] = []

    for ref_stem in selected_pairs:
        tgt_stem = _PAIRS[ref_stem]
        ref_path = _HERE / f"{ref_stem}.vtk"
        tgt_path = _HERE / f"{tgt_stem}.vtk"

        if not ref_path.exists() or not tgt_path.exists():
            click.echo(f"\nSKIP {ref_stem}: data not found — run generate_samples.py first.")
            continue

        click.echo(f"\n{sep}")
        click.echo(f"  Pair: {ref_stem}")
        click.echo(sep)

        work_dir = output_dir / ref_stem

        if no_preprocess:
            ref_final = ref_path
            tgt_final = tgt_path
        else:
            click.echo("  Pre-processing (symplane → recenter → normalize)…")
            t_pre = time.perf_counter()
            ref_final, tgt_final = _preprocess_pair(ref_path, tgt_path, work_dir, verbose)
            click.echo(f"  Pre-processing done  ({time.perf_counter()-t_pre:.1f} s)")

        # Load surfaces
        ref_pts, ref_poly = load_surface(str(ref_final))
        tgt_pts, tgt_poly, tgt_normals = load_surface_with_normals(str(tgt_final))
        ref_pts_n, _, ref_normals = load_surface_with_normals(str(ref_final))

        before = symmetric_rms_nn(tgt_pts, ref_pts)
        click.echo(f"\n  RMS NN before: {before:.3f} mm  "
                   f"({len(tgt_pts):,} moving pts, {len(ref_pts):,} ref pts)")

        for method in selected_methods:
            click.echo(f"\n  [{method.upper()}]")
            try:
                if method == "clarcs":
                    warped, elapsed = run_clarcs(
                        tgt_pts, tgt_normals, ref_pts_n, ref_normals, tgt_poly,
                        max_iter=max_iter, verbose=verbose,
                    )
                elif method == "cpd":
                    warped, elapsed = run_cpd(
                        tgt_pts, ref_pts, max_iter=max_iter, verbose=verbose,
                    )
                elif method == "bcpd":
                    warped, elapsed = run_bcpd(
                        tgt_pts, ref_pts, bcpd_bin=bcpd_bin,
                        verbose=verbose,
                    )
                else:
                    continue

                after = symmetric_rms_nn(warped, ref_pts)
                improvement = (before - after) / before * 100.0

                click.echo(f"  RMS after: {after:.3f} mm  "
                           f"improvement: {improvement:.1f} %  "
                           f"({elapsed:.1f} s)")

                from pyclarcs.io import save_surface
                out_path = work_dir / f"{tgt_path.stem}-{method}.vtk"
                save_surface(str(out_path), warped, tgt_poly)
                click.echo(f"  Saved: {out_path.name}")

                results.append({
                    "pair": ref_stem, "method": method,
                    "before_mm": round(before, 3),
                    "after_mm": round(after, 3),
                    "improvement_pct": round(improvement, 1),
                    "time_s": round(elapsed, 1),
                })

            except Exception as exc:
                click.echo(f"  ERROR: {exc}", err=True)
                if verbose:
                    traceback.print_exc()

    # Summary table
    if results:
        click.echo(f"\n{sep}")
        click.echo("  SUMMARY")
        click.echo(sep)
        click.echo(f"  {'Pair':<30} {'Method':<8} {'Before':>8} {'After':>8} {'Improv':>8} {'Time':>7}")
        click.echo("  " + "-" * 64)
        for r in results:
            click.echo(
                f"  {r['pair']:<30} {r['method']:<8}"
                f"  {r['before_mm']:>6.2f}mm  {r['after_mm']:>6.2f}mm"
                f"  {r['improvement_pct']:>6.1f}%  {r['time_s']:>5.1f}s"
            )

        # Save CSV
        import csv
        csv_path = output_dir / "results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)
        click.echo(f"\n  Results saved: {csv_path}")


if __name__ == "__main__":
    main()
