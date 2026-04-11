"""
Command-line interface for pyclarcs.

Replicates the ZZ_SYMC CLI:
    ZZ_SYMC -i <input> [-o <output>] [-O <plan>] [--init auto|<file>] ...

Usage examples::

    pyclarcs -i surface.vtk -O results/plane -o results/symmetric.vtk
    pyclarcs -i surface.vtk --init plane.pl -O results/plane
    pyclarcs -i surface.vtk --no-fine --no-sym -O results/plane
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pyclarcs",
        description=(
            "Automatic symmetry plane estimation for 3-D surfaces. "
            "Python port of the ZZ_SYMC tool from the CLARCS project."
        ),
    )
    p.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input .vtk surface file.",
    )
    p.add_argument(
        "-o", "--output",
        default="",
        metavar="FILE",
        help="Output .vtk file: the input surface reflected through the estimated plane.",
    )
    p.add_argument(
        "-O", "--Output",
        default="",
        metavar="PREFIX",
        help=(
            "Output prefix for the symmetry plane files. "
            "Writes <PREFIX>.pl (plane parameters) and <PREFIX>.vtk (plane patch)."
        ),
    )
    p.add_argument(
        "--init",
        default="auto",
        metavar="auto|FILE",
        help=(
            "'auto' (default): pick the best principal-axis plane automatically. "
            "Or provide a path to a .pl file for a custom initial plane."
        ),
    )
    p.add_argument(
        "--no-coarse",
        action="store_true",
        help="Skip the coarse ICP optimisation step.",
    )
    p.add_argument(
        "--no-fine",
        action="store_true",
        help="Skip the EM-ICP fine optimisation step.",
    )
    p.add_argument(
        "--no-sym",
        action="store_true",
        help="Skip the doubly-stochastic EM-ICP final refinement.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Print progress information (default: on).",
    )
    p.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all progress output.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    verbose = args.verbose and not args.quiet

    # ------------------------------------------------------------------
    # Lazy imports so that the CLI stays fast when invoked with --help
    # ------------------------------------------------------------------
    from pyclarcs._io import load_surface, save_surface, save_plane_vtk
    from pyclarcs._principal_axes import best_principal_axis_plane
    from pyclarcs._coarse import coarse_symmetry
    from pyclarcs._fine import em_icp_sym, em_icp_sym_corres
    from pyclarcs._symmetry import SymmetryPlane

    # ------------------------------------------------------------------
    # Load surface
    # ------------------------------------------------------------------
    if verbose:
        print(f"Loading surface: {args.input}")
    points, polygons = load_surface(args.input)
    if verbose:
        print(f"  {len(points)} points, {len(polygons)} faces")

    bounds = (
        float(points[:, 0].min()), float(points[:, 0].max()),
        float(points[:, 1].min()), float(points[:, 1].max()),
        float(points[:, 2].min()), float(points[:, 2].max()),
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    if args.init == "auto":
        if verbose:
            print("Computing principal-axis initialisation…")
        plane = best_principal_axis_plane(points)
        if verbose:
            print(f"  Initial plane: {plane}")
    else:
        if verbose:
            print(f"Loading initial plane from: {args.init}")
        plane = SymmetryPlane.load(args.init)
        if verbose:
            print(f"  Loaded plane: {plane}")

    # ------------------------------------------------------------------
    # Coarse optimisation (ICP + trimmed estimator)
    # ------------------------------------------------------------------
    if not args.no_coarse:
        if verbose:
            print("Coarse optimisation (ICP)…")
        plane = coarse_symmetry(points, plane, verbose=verbose)
        if verbose:
            print(f"  Coarse plane: {plane}")

    # ------------------------------------------------------------------
    # Fine optimisation (EM-ICP with annealing)
    # ------------------------------------------------------------------
    if not args.no_fine:
        if verbose:
            print("Fine optimisation (EM-ICP)…")
        plane = em_icp_sym(points, plane, verbose=verbose)
        if verbose:
            print(f"  Fine plane: {plane}")

    # ------------------------------------------------------------------
    # Final refinement (doubly-stochastic EM-ICP)
    # ------------------------------------------------------------------
    if not args.no_sym:
        if verbose:
            print("Final refinement (EM-ICP symmetric correspondences)…")
        plane = em_icp_sym_corres(points, plane, verbose=verbose)
        if verbose:
            print(f"  Final plane: {plane}")

    # ------------------------------------------------------------------
    # Output: reflected surface
    # ------------------------------------------------------------------
    if args.output:
        if verbose:
            print(f"Saving reflected surface: {args.output}")
        reflected_pts = plane.apply(points)
        save_surface(args.output, reflected_pts, polygons)

    # ------------------------------------------------------------------
    # Output: symmetry plane files
    # ------------------------------------------------------------------
    if args.Output:
        pl_path = args.Output + ".pl"
        vtk_path = args.Output + ".vtk"
        if verbose:
            print(f"Saving symmetry plane: {pl_path}  {vtk_path}")
        plane.save(pl_path)
        save_plane_vtk(vtk_path, plane, bounds)

    if verbose:
        print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
