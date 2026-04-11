"""
Command-line interface for clarcs.

clarcs is the top-level command; sub-commands provide specific tools:

    clarcs sym-plane <input> [output] [--save-plane] [options]

Examples::

    clarcs sym-plane surface.vtk
    clarcs sym-plane surface.vtk plane.vtk
    clarcs sym-plane surface.vtk --save-plane
    clarcs sym-plane surface.vtk --init plane.pl --save-plane
    clarcs sym-plane surface.vtk --no-fine --no-sym
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# sym-plane sub-command
# ---------------------------------------------------------------------------

def _build_sym_plane_parser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "sym-plane",
        help="Automatic symmetry plane estimation for a 3-D surface.",
        description=(
            "Find the best symmetry plane of a 3-D surface (e.g. endocranial). "
            "Writes a VTK patch visualising the plane to OUTPUT (auto-named when omitted). "
            "Use --save-plane to also write the plane parameters (.pl)."
        ),
    )
    p.add_argument(
        "input",
        metavar="INPUT",
        help="Input .vtk surface file.",
    )
    p.add_argument(
        "output",
        nargs="?",
        default=None,
        metavar="OUTPUT",
        help=(
            "Output .vtk file for the symmetry plane patch. "
            "Defaults to <INPUT_STEM>-sym-plane<EXT>."
        ),
    )
    p.add_argument(
        "--save-plane",
        action="store_true",
        help=(
            "Also save the plane parameters to <OUTPUT_STEM>.pl."
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


def _run_sym_plane(args: argparse.Namespace) -> int:
    verbose = args.verbose and not args.quiet

    # ------------------------------------------------------------------
    # Resolve output path
    # ------------------------------------------------------------------
    input_path = Path(args.input)
    if args.output is None:
        output_path = input_path.with_stem(input_path.stem + "-sym-plane")
    else:
        output_path = Path(args.output)

    # ------------------------------------------------------------------
    # Lazy imports so that the CLI stays fast when invoked with --help
    # ------------------------------------------------------------------
    from pyclarcs.io import load_surface, save_surface, save_plane_vtk
    from pyclarcs.principal_axes import best_principal_axis_plane
    from pyclarcs.coarse import coarse_symmetry
    from pyclarcs.fine import em_icp_sym, em_icp_sym_corres
    from pyclarcs.symmetry import SymmetryPlane

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
    # Output: symmetry plane patch (VTK rectangular patch)
    # ------------------------------------------------------------------
    if verbose:
        print(f"Saving symmetry plane patch: {output_path}")
    save_plane_vtk(str(output_path), plane, bounds)

    # ------------------------------------------------------------------
    # Output: plane parameters (optional)
    # ------------------------------------------------------------------
    if args.save_plane:
        pl_path = output_path.with_suffix(".pl")
        if verbose:
            print(f"Saving symmetry plane parameters: {pl_path}")
        plane.save(str(pl_path))

    if verbose:
        print("Done.")
    return 0


# ---------------------------------------------------------------------------
# Top-level clarcs command
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clarcs",
        description="CLARCS — tools for 3-D surface analysis.",
    )
    subparsers = p.add_subparsers(dest="subcommand", metavar="COMMAND")
    subparsers.required = True
    _build_sym_plane_parser(subparsers)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "sym-plane":
        return _run_sym_plane(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
