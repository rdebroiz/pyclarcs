"""
Command-line interface for clarcs (Click-based).

    clarcs sym-plane    INPUT [OUTPUT] [--save-plane] [options]
    clarcs centerofmass INPUT [OUTPUT] --target TARGET
    clarcs rescale      INPUT [OUTPUT] --target TARGET
    clarcs recenter     INPUT [OUTPUT] --plane  PLANE.pl
    clarcs orient       INPUT [OUTPUT] --axes   X Y Z
"""

from __future__ import annotations

import sys
from pathlib import Path

import click


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_output(input_path: str, suffix: str) -> str:
    p = Path(input_path)
    return str(p.with_stem(p.stem + suffix))


_verbose_option = click.option(
    "-q", "--quiet", "quiet",
    is_flag=True, default=False,
    help="Suppress all output.",
)


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="pyclarcs")
def cli():
    """CLARCS — tools for 3-D surface analysis."""


# ---------------------------------------------------------------------------
# sym-plane
# ---------------------------------------------------------------------------

@cli.command("sym-plane")
@click.argument("input_path",  metavar="INPUT",  type=click.Path(exists=True))
@click.argument("output_path", metavar="OUTPUT", required=False, default=None)
@click.option("--save-plane", is_flag=True,
              help="Also save plane parameters to <OUTPUT_STEM>.pl.")
@click.option("--init", default="auto", metavar="auto|FILE", show_default=True,
              help="'auto' (principal axes) or path to a .pl file.")
@click.option("--no-coarse", is_flag=True, help="Skip the coarse ICP stage.")
@click.option("--no-fine",   is_flag=True, help="Skip the EM-ICP annealing stage.")
@click.option("--no-sym",    is_flag=True, help="Skip the doubly-stochastic refinement.")
@_verbose_option
def sym_plane(input_path, output_path, save_plane, init, no_coarse, no_fine, no_sym, quiet):
    """Find the best bilateral symmetry plane of a 3-D surface."""
    verbose = not quiet

    if output_path is None:
        output_path = _default_output(input_path, "-sym-plane")

    from pyclarcs.io import load_surface, save_plane_vtk
    from pyclarcs.principal_axes import best_principal_axis_plane
    from pyclarcs.coarse import coarse_symmetry
    from pyclarcs.fine import em_icp_sym, em_icp_sym_corres
    from pyclarcs.symmetry import SymmetryPlane

    if verbose:
        click.echo(f"Loading surface: {input_path}")
    points, polygons = load_surface(input_path)
    if verbose:
        click.echo(f"  {len(points)} points, {len(polygons)} faces")

    bounds = (
        float(points[:, 0].min()), float(points[:, 0].max()),
        float(points[:, 1].min()), float(points[:, 1].max()),
        float(points[:, 2].min()), float(points[:, 2].max()),
    )

    if init == "auto":
        if verbose:
            click.echo("Computing principal-axis initialisation…")
        plane = best_principal_axis_plane(points)
        if verbose:
            click.echo(f"  Initial plane: {plane}")
    else:
        if verbose:
            click.echo(f"Loading initial plane from: {init}")
        plane = SymmetryPlane.load(init)
        if verbose:
            click.echo(f"  Loaded plane: {plane}")

    if not no_coarse:
        if verbose:
            click.echo("Coarse optimisation (ICP)…")
        plane = coarse_symmetry(points, plane, verbose=verbose)
        if verbose:
            click.echo(f"  Coarse plane: {plane}")

    if not no_fine:
        if verbose:
            click.echo("Fine optimisation (EM-ICP)…")
        plane = em_icp_sym(points, plane, verbose=verbose)
        if verbose:
            click.echo(f"  Fine plane: {plane}")

    if not no_sym:
        if verbose:
            click.echo("Final refinement (EM-ICP symmetric correspondences)…")
        plane = em_icp_sym_corres(points, plane, verbose=verbose)
        if verbose:
            click.echo(f"  Final plane: {plane}")

    if verbose:
        click.echo(f"Saving symmetry plane patch: {output_path}")
    save_plane_vtk(output_path, plane, bounds)

    if save_plane:
        pl_path = str(Path(output_path).with_suffix(".pl"))
        if verbose:
            click.echo(f"Saving plane parameters: {pl_path}")
        plane.save(pl_path)

    if verbose:
        click.echo("Done.")


# ---------------------------------------------------------------------------
# centerofmass
# ---------------------------------------------------------------------------

@cli.command("centerofmass")
@click.argument("input_path",  metavar="INPUT",  type=click.Path(exists=True))
@click.argument("output_path", metavar="OUTPUT", required=False, default=None)
@click.option("--target", required=True, type=click.Path(exists=True), metavar="TARGET",
              help="Reference surface whose centre of mass to match.")
@_verbose_option
def centerofmass(input_path, output_path, target, quiet):
    """Translate a surface to align its centre of mass with a reference."""
    verbose = not quiet

    if output_path is None:
        output_path = _default_output(input_path, "-centerofmass")

    from pyclarcs.io import load_surface, save_surface
    from pyclarcs.alignment import align_center_of_mass

    if verbose:
        click.echo(f"Loading surface: {input_path}")
    points, polygons = load_surface(input_path)

    if verbose:
        click.echo(f"Loading target: {target}")
    target_pts, _ = load_surface(target)

    if verbose:
        click.echo("Aligning centres of mass…")
    result = align_center_of_mass(points, target_pts)

    if verbose:
        click.echo(f"Saving: {output_path}")
    save_surface(output_path, result, polygons)

    if verbose:
        click.echo("Done.")


# ---------------------------------------------------------------------------
# rescale
# ---------------------------------------------------------------------------

@cli.command("rescale")
@click.argument("input_path",  metavar="INPUT",  type=click.Path(exists=True))
@click.argument("output_path", metavar="OUTPUT", required=False, default=None)
@click.option("--target", required=True, type=click.Path(exists=True), metavar="TARGET",
              help="Reference surface to match.")
@_verbose_option
def rescale(input_path, output_path, target, quiet):
    """Translate and uniformly scale a surface to match a reference's size and position."""
    verbose = not quiet

    if output_path is None:
        output_path = _default_output(input_path, "-rescale")

    from pyclarcs.io import load_surface, save_surface
    from pyclarcs.alignment import align_rescale

    if verbose:
        click.echo(f"Loading surface: {input_path}")
    points, polygons = load_surface(input_path)

    if verbose:
        click.echo(f"Loading target: {target}")
    target_pts, _ = load_surface(target)

    if verbose:
        click.echo("Rescaling to match target centre of mass and dispersion…")
    result = align_rescale(points, target_pts)

    if verbose:
        click.echo(f"Saving: {output_path}")
    save_surface(output_path, result, polygons)

    if verbose:
        click.echo("Done.")


# ---------------------------------------------------------------------------
# recenter
# ---------------------------------------------------------------------------

@cli.command("recenter")
@click.argument("input_path",  metavar="INPUT",  type=click.Path(exists=True))
@click.argument("output_path", metavar="OUTPUT", required=False, default=None)
@click.option("--plane", required=True, type=click.Path(exists=True), metavar="PLANE",
              help="Symmetry plane file (.pl) from  clarcs sym-plane --save-plane.")
@_verbose_option
def recenter(input_path, output_path, plane, quiet):
    """Rigidly align a surface so its symmetry plane coincides with x = 0."""
    verbose = not quiet

    if output_path is None:
        output_path = _default_output(input_path, "-recentered")

    from pyclarcs.io import load_surface, save_surface
    from pyclarcs.symmetry import SymmetryPlane
    from pyclarcs.alignment import align_to_symmetry_plane

    if verbose:
        click.echo(f"Loading surface: {input_path}")
    points, polygons = load_surface(input_path)

    if verbose:
        click.echo(f"Loading symmetry plane: {plane}")
    sym_plane = SymmetryPlane.load(plane)
    if verbose:
        click.echo(f"  {sym_plane}")
        click.echo("Aligning to canonical symmetry plane (n=[1,0,0], d=0)…")

    result = align_to_symmetry_plane(points, sym_plane)

    if verbose:
        click.echo(f"Saving: {output_path}")
    save_surface(output_path, result, polygons)

    if verbose:
        click.echo("Done.")


# ---------------------------------------------------------------------------
# orient
# ---------------------------------------------------------------------------

@cli.command("orient")
@click.argument("input_path",  metavar="INPUT",  type=click.Path(exists=True))
@click.argument("output_path", metavar="OUTPUT", required=False, default=None)
@click.option("--axes", nargs=3, type=int, default=(0, 1, 2), metavar="X Y Z",
              show_default=True,
              help="Destination column indices for the current x, y, z axes.")
@_verbose_option
def orient(input_path, output_path, axes, quiet):
    """Permute the coordinate axes of a surface.

    Example: --axes 2 1 0 swaps x and z.
    """
    verbose = not quiet

    if output_path is None:
        output_path = _default_output(input_path, "-oriented")

    from pyclarcs.io import load_surface, save_surface
    from pyclarcs.alignment import reorient_axes

    if verbose:
        click.echo(f"Loading surface: {input_path}")
    points, polygons = load_surface(input_path)

    x_to, y_to, z_to = axes
    if verbose:
        click.echo(f"Permuting axes: x→{x_to}, y→{y_to}, z→{z_to}")

    result = reorient_axes(points, x_to, y_to, z_to)

    if verbose:
        click.echo(f"Saving: {output_path}")
    save_surface(output_path, result, polygons)

    if verbose:
        click.echo("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cli()


if __name__ == "__main__":
    main()
