"""
Command-line interface for clarcs (Click-based).

    clarcs reorient     INPUT [OUTPUT] --axes   X Y Z
    clarcs symplane     INPUT [OUTPUT] [--save-plane] [options]
    clarcs recenter     INPUT [OUTPUT] --plane  PLANE.pl
    clarcs centerofmass INPUT [OUTPUT] --target TARGET
    clarcs normalize    INPUT [OUTPUT] --target TARGET
    clarcs nlregister   INPUT REF     [OUTPUT] [--deformation FIELD] [options]
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
# reorient
# ---------------------------------------------------------------------------

@cli.command("reorient")
@click.argument("input_path",  metavar="INPUT",  type=click.Path(exists=True))
@click.argument("output_path", metavar="OUTPUT", required=False, default=None)
@click.option("--axes", nargs=3, type=int, default=(0, 1, 2), metavar="X Y Z",
              show_default=True,
              help="Destination column indices for the current x, y, z axes.")
@_verbose_option
def reorient(input_path, output_path, axes, quiet):
    """Permute the coordinate axes of a surface.

    Example: --axes 2 1 0 swaps x and z.
    """
    verbose = not quiet

    if output_path is None:
        output_path = _default_output(input_path, "-reoriented")

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
# symplane
# ---------------------------------------------------------------------------

@cli.command("symplane")
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
def symplane(input_path, output_path, save_plane, init, no_coarse, no_fine, no_sym, quiet):
    """Find the best bilateral symmetry plane of a 3-D surface."""
    verbose = not quiet

    if output_path is None:
        output_path = _default_output(input_path, "-symplane")

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
# recenter
# ---------------------------------------------------------------------------

@cli.command("recenter")
@click.argument("input_path",  metavar="INPUT",  type=click.Path(exists=True))
@click.argument("output_path", metavar="OUTPUT", required=False, default=None)
@click.option("--plane", required=False, default=None,
              type=click.Path(exists=True), metavar="PLANE",
              help=(
                  "Symmetry plane file (.pl). "
                  "If omitted, the plane is computed automatically "
                  "(equivalent to running  clarcs symplane  first)."
              ))
@click.option("--save-plane", is_flag=True,
              help="Save the (computed or loaded) plane to <OUTPUT_STEM>.pl.")
@_verbose_option
def recenter(input_path, output_path, plane, save_plane, quiet):
    """Rigidly align a surface so its symmetry plane coincides with x = 0.

    If --plane is omitted the symmetry plane is estimated automatically
    from the surface itself before applying the alignment.
    """
    verbose = not quiet

    if output_path is None:
        output_path = _default_output(input_path, "-recentered")

    from pyclarcs.io import load_surface, save_surface
    from pyclarcs.symmetry import SymmetryPlane
    from pyclarcs.alignment import align_to_symmetry_plane

    if verbose:
        click.echo(f"Loading surface: {input_path}")
    points, polygons = load_surface(input_path)

    if plane is not None:
        if verbose:
            click.echo(f"Loading symmetry plane: {plane}")
        sym_plane = SymmetryPlane.load(plane)
        if verbose:
            click.echo(f"  {sym_plane}")
    else:
        if verbose:
            click.echo("No plane provided — estimating symmetry plane…")
        from pyclarcs.principal_axes import best_principal_axis_plane
        from pyclarcs.coarse import coarse_symmetry
        from pyclarcs.fine import em_icp_sym, em_icp_sym_corres

        sym_plane = best_principal_axis_plane(points)
        sym_plane = coarse_symmetry(points, sym_plane, verbose=verbose)
        sym_plane = em_icp_sym(points, sym_plane, verbose=verbose)
        sym_plane = em_icp_sym_corres(points, sym_plane, verbose=verbose)
        if verbose:
            click.echo(f"  Estimated plane: {sym_plane}")

    if verbose:
        click.echo("Aligning to canonical symmetry plane (n=[1,0,0], d=0)…")
    result = align_to_symmetry_plane(points, sym_plane)

    if verbose:
        click.echo(f"Saving: {output_path}")
    save_surface(output_path, result, polygons)

    if save_plane:
        pl_path = str(Path(output_path).with_suffix(".pl"))
        if verbose:
            click.echo(f"Saving plane parameters: {pl_path}")
        sym_plane.save(pl_path)

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
# normalize
# ---------------------------------------------------------------------------

@cli.command("normalize")
@click.argument("input_path",  metavar="INPUT",  type=click.Path(exists=True))
@click.argument("output_path", metavar="OUTPUT", required=False, default=None)
@click.option("--target", required=True, type=click.Path(exists=True), metavar="TARGET",
              help="Reference surface to match.")
@_verbose_option
def normalize(input_path, output_path, target, quiet):
    """Translate and uniformly scale a surface to match a reference's size and position."""
    verbose = not quiet

    if output_path is None:
        output_path = _default_output(input_path, "-normalized")

    from pyclarcs.io import load_surface, save_surface
    from pyclarcs.alignment import align_rescale

    if verbose:
        click.echo(f"Loading surface: {input_path}")
    points, polygons = load_surface(input_path)

    if verbose:
        click.echo(f"Loading target: {target}")
    target_pts, _ = load_surface(target)

    if verbose:
        click.echo("Normalizing to match target centre of mass and dispersion…")
    result = align_rescale(points, target_pts)

    if verbose:
        click.echo(f"Saving: {output_path}")
    save_surface(output_path, result, polygons)

    if verbose:
        click.echo("Done.")


# ---------------------------------------------------------------------------
# nlregister
# ---------------------------------------------------------------------------

@cli.command("nlregister")
@click.argument("input_path", metavar="INPUT", type=click.Path(exists=True))
@click.argument("ref_path",   metavar="REF",   type=click.Path(exists=True))
@click.argument("output_path", metavar="OUTPUT", required=False, default=None)
@click.option("--deformation", default=None, metavar="FIELD",
              help="Also save the deformation field to this VTK file.")
@click.option("--sigma",        default=None,  type=float,
              help="Initial bandwidth [mm]. Auto-estimated from surfaces if omitted.")
@click.option("--beta",         default=100.0, show_default=True, type=float,
              help="Regularisation weight (higher = smoother).")
@click.option("--dist-cutoff",  default=None,  type=float,
              help="Search radius [mm]. Auto-estimated from surfaces if omitted.")
@click.option("--max-iter",     default=80,    show_default=True, type=int,
              help="Number of outer EM iterations.")
@click.option("--icm-iter",     default=120,   show_default=True, type=int,
              help="Number of Jacobi ICM steps per outer iteration.")
@click.option("--period-sigma", default=None,  type=int,
              help="Halve sigma every this many iterations. Auto-estimated if omitted.")
@click.option("--sigma-min",    default=0.1,   show_default=True, type=float,
              help="Minimum sigma (annealing floor).")
@click.option("--e-chunk",      default=2000,  show_default=True, type=int,
              help="Vertices per KDTree batch in the E-step (lower = less RAM).")
@_verbose_option
def nlregister(input_path, ref_path, output_path, deformation,
               sigma, beta, dist_cutoff, max_iter, icm_iter, period_sigma,
               sigma_min, e_chunk, quiet):
    """Non-linearly register INPUT onto REF using EM-ICP.

    Outputs the warped INPUT surface.  Optionally saves the per-vertex
    deformation field as a VTK file with VECTORS point data.
    """
    verbose = not quiet

    if output_path is None:
        output_path = _default_output(input_path, "-nlregistered")

    from pyclarcs.io import (
        load_surface_with_normals, save_surface, save_deformation_vtk,
    )
    from pyclarcs.mesh import adjacency_csr
    from pyclarcs.nonrigid import nonrigid_icp, apply_deformation

    if verbose:
        click.echo(f"Loading moving surface: {input_path}")
    mov_pts, mov_poly, mov_normals = load_surface_with_normals(input_path)
    if verbose:
        click.echo(f"  {len(mov_pts)} points, {len(mov_poly)} faces")

    if verbose:
        click.echo(f"Loading reference surface: {ref_path}")
    ref_pts, _, ref_normals = load_surface_with_normals(ref_path)
    if verbose:
        click.echo(f"  {len(ref_pts)} points")

    if sigma is None or dist_cutoff is None or period_sigma is None:
        from pyclarcs.nonrigid import estimate_registration_params
        auto = estimate_registration_params(
            mov_pts, ref_pts,
            max_iter=max_iter, sigma_min=sigma_min,
        )
        if sigma is None:
            sigma = auto["sigma"]
        if dist_cutoff is None:
            dist_cutoff = auto["dist_cutoff"]
        if period_sigma is None:
            period_sigma = auto["period_sigma"]
        if verbose:
            click.echo(
                f"Auto params:  σ={sigma}  r={dist_cutoff}"
                f"  period_σ={period_sigma}"
            )

    if verbose:
        click.echo("Building mesh adjacency…")
    adj = adjacency_csr(mov_poly, len(mov_pts))

    if verbose:
        click.echo(
            f"Non-linear EM-ICP  "
            f"σ={sigma}  β={beta}  r={dist_cutoff}  "
            f"iter={max_iter}×{icm_iter}"
        )
    def_field = nonrigid_icp(
        mov_pts, mov_normals,
        ref_pts, ref_normals,
        adj,
        sigma=sigma,
        beta=beta,
        dist_cutoff=dist_cutoff,
        max_iter=max_iter,
        icm_iter=icm_iter,
        period_sigma=period_sigma,
        sigma_min=sigma_min,
        e_chunk=e_chunk,
        verbose=verbose,
    )

    warped = apply_deformation(mov_pts, def_field)

    if verbose:
        click.echo(f"Saving warped surface: {output_path}")
    save_surface(output_path, warped, mov_poly)

    if deformation is not None:
        if verbose:
            click.echo(f"Saving deformation field: {deformation}")
        save_deformation_vtk(deformation, mov_pts, mov_poly, def_field)

    if verbose:
        click.echo("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cli()


if __name__ == "__main__":
    main()
