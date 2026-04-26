"""
Command-line interface for clarcs (Click-based).

    clarcs reorient     INPUT [OUTPUT] --axes X Y Z
    clarcs symplane     INPUT [OUTPUT] [--save-plane] [options]
    clarcs recenter     INPUT [OUTPUT] --plane PLANE.pl
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
# mirror
# ---------------------------------------------------------------------------

@cli.command("mirror")
@click.argument("input_path",  metavar="INPUT",  type=click.Path(exists=True))
@click.argument("output_path", metavar="OUTPUT", required=False, default=None)
@click.option("--plane", default=None, type=click.Path(exists=True), metavar="PLANE.pl",
              help="Symmetry plane file (.pl). Estimated automatically if omitted.")
@click.option("--save-plane", is_flag=True,
              help="Save the (computed or loaded) plane to <OUTPUT_STEM>.pl.")
@_verbose_option
def mirror(input_path, output_path, plane, save_plane, quiet):
    """Reflect a surface across its bilateral symmetry plane.

    If --plane is omitted the symmetry plane is estimated automatically
    from the surface itself (equivalent to running  clarcs symplane  first).
    """
    verbose = not quiet

    if output_path is None:
        output_path = _default_output(input_path, "-mirror")

    from pyclarcs.io import load_surface, save_surface
    from pyclarcs.symmetry import SymmetryPlane
    from pyclarcs.alignment import reflect_surface

    if verbose:
        click.echo(f"Loading surface: {input_path}")
    points, polygons = load_surface(input_path)
    if verbose:
        click.echo(f"  {len(points)} points, {len(polygons)} faces")

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

    plane_normal = sym_plane.n
    plane_point  = sym_plane.n * sym_plane.d

    if verbose:
        click.echo("Reflecting surface…")
    mirrored = reflect_surface(points, plane_normal, plane_point)

    # Reflection reverses triangle winding (right-hand rule flips sign).
    # Reversing each face restores outward-pointing normals.
    import numpy as np
    flipped_polygons = [f[::-1] for f in polygons]

    if verbose:
        click.echo(f"Saving: {output_path}")
    save_surface(output_path, mirrored, flipped_polygons)

    if save_plane:
        pl_path = str(Path(output_path).with_suffix(".pl"))
        if verbose:
            click.echo(f"Saving plane parameters: {pl_path}")
        sym_plane.save(pl_path)

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
              help="Initial bandwidth [mm]. Auto-estimated per level if omitted.")
@click.option("--beta",         default=None,  type=float,
              help="Regularisation weight. Auto-estimated from mesh spacing if omitted.")
@click.option("--dist-cutoff",  default=None,  type=float,
              help="Search radius [mm]. Auto-estimated per level if omitted.")
@click.option("--max-iter",     default=80,    show_default=True, type=int,
              help="Number of outer EM iterations.")
@click.option("--icm-iter",     default=50,    show_default=True, type=int,
              help="Max conjugate gradient iterations per outer iteration.")
@click.option("--period-sigma", default=None,  type=int,
              help="Halve sigma every this many iterations. Auto-estimated if omitted.")
@click.option("--sigma-min",    default=None,  type=float,
              help="Minimum sigma (annealing floor). Auto-estimated from mesh spacing if omitted.")
@click.option("--e-chunk",      default=2000,  show_default=True, type=int,
              help="Vertices per KDTree batch in the E-step (lower = less RAM).")
@click.option("--n-levels",          default=None, type=int,
              help="Resolution levels (1=single-res, ≥2=multi-res). Auto from surface size if omitted.")
@click.option("--coarsest-n",        default=2000, show_default=True, type=int,
              help="Target vertex count at the coarsest level (multi-res only).")
@click.option("--beta-coarse-factor", default=1.0, show_default=True, type=float,
              help="Per-level beta multiplier toward coarser levels (only used when --beta is set). "
                   "beta at level idx = beta × factor^idx (idx=0: finest).")
@click.option("--outlier-weight", default=0.1, show_default=True, type=float,
              help="Prior probability of a vertex being an outlier (0=disabled). "
                   "CPD-style: down-weights vertices with few/poor correspondences.")
@click.option("--normal-min-dot", default=0.0, show_default=True, type=float,
              help="Minimum dot-product of source/reference normals to accept a correspondence "
                   "(0=same hemisphere, 1=perfectly aligned).")
@click.option("--no-symmetric",  is_flag=True, default=False,
              help="Disable symmetric correspondences (A+B, Reg2). Enabled by default.")
@click.option("--no-tgd",        is_flag=True, default=False,
              help="Disable the TGD geodesic shape prior (Reg3). Enabled by default.")
@click.option("--no-rkhs",       is_flag=True, default=False,
              help="Disable RKHS Wu-kernel M-step; fall back to Laplacian M-step.")
@click.option("--rkhs-lambda",   default=0.01, show_default=True, type=float,
              help="RKHS regularisation weight (smaller = larger deformations).")
@_verbose_option
def nlregister(input_path, ref_path, output_path, deformation,
               sigma, beta, dist_cutoff, max_iter, icm_iter, period_sigma,
               sigma_min, e_chunk, n_levels, coarsest_n, beta_coarse_factor,
               outlier_weight, normal_min_dot,
               no_symmetric, no_tgd, no_rkhs, rkhs_lambda,
               quiet):
    """Non-linearly register INPUT onto REF using EM-ICP.

    Outputs the warped INPUT surface.  Optionally saves the per-vertex
    deformation field as a VTK file with VECTORS point data.

    All registration parameters (sigma, beta, sigma-min, n-levels) are
    auto-estimated from the surfaces when omitted.  sigma, dist-cutoff and
    period-sigma are re-estimated at each resolution level from the current
    residual; beta and sigma-min are derived from the level's mesh spacing.
    """
    verbose = not quiet

    if output_path is None:
        output_path = _default_output(input_path, "-nlregistered")

    from pyclarcs.io import (
        load_surface_with_normals, save_surface, save_deformation_vtk,
    )
    from pyclarcs.nonrigid import register, apply_deformation

    if verbose:
        click.echo(f"Loading moving surface: {input_path}")
    mov_pts, mov_poly, mov_normals = load_surface_with_normals(input_path)
    if verbose:
        click.echo(f"  {len(mov_pts)} points, {len(mov_poly)} faces")

    if verbose:
        click.echo(f"Loading reference surface: {ref_path}")
    from pyclarcs.io import load_surface
    ref_pts, ref_poly = load_surface(ref_path)
    _, _, ref_normals = load_surface_with_normals(ref_path)
    if verbose:
        click.echo(f"  {len(ref_pts)} points")

    # Resolve n_levels (the only parameter that cannot be deferred to multires
    # because it controls how many levels to build).
    if n_levels is None:
        N = len(mov_pts)
        n_levels = 1 if N <= 5_000 else (2 if N <= 30_000 else 3)
        if verbose:
            click.echo(
                f"  auto n_levels={n_levels}  ({N} moving vertices)"
            )

    if verbose:
        auto_str = []
        if beta      is None: auto_str.append("β")
        if sigma_min is None: auto_str.append("σ_min")
        if sigma     is None: auto_str.append("σ")
        if dist_cutoff   is None: auto_str.append("r")
        if period_sigma  is None: auto_str.append("period_σ")
        auto_note = (
            f"  auto per level: {', '.join(auto_str)}" if auto_str else ""
        )
        click.echo(
            f"Non-linear EM-ICP  {n_levels} level(s)"
            f"  coarsest={coarsest_n} pts"
            f"  iter={max_iter}×{icm_iter}"
            + auto_note
        )

    def_field = register(
        mov_pts, mov_normals,
        ref_pts, ref_normals,
        mov_poly,
        ref_poly,
        n_levels=n_levels,
        target_n_coarsest=coarsest_n,
        sigma=sigma,
        beta=beta,
        beta_coarse_factor=beta_coarse_factor,
        dist_cutoff=dist_cutoff,
        max_iter=max_iter,
        icm_iter=icm_iter,
        period_sigma=period_sigma,
        sigma_min=sigma_min,
        outlier_weight=outlier_weight,
        normal_min_dot=normal_min_dot,
        e_chunk=e_chunk,
        symmetric=not no_symmetric,
        use_tgd=not no_tgd,
        use_rkhs=not no_rkhs,
        rkhs_lambda=rkhs_lambda,
        verbose=verbose,
    )

    warped = apply_deformation(mov_pts, def_field)

    import numpy as np
    from scipy.spatial import KDTree
    ref_tree = KDTree(ref_pts)
    dists0, _ = ref_tree.query(mov_pts, k=1, workers=-1)
    rms0 = float(np.sqrt(np.mean(dists0 ** 2)))
    dists, _ = ref_tree.query(warped, k=1, workers=-1)
    rms = float(np.sqrt(np.mean(dists ** 2)))
    improvement = 100.0 * (rms0 - rms) / rms0 if rms0 > 0 else 0.0
    click.echo(f"RMS: {rms0:.4f} mm → {rms:.4f} mm  ({improvement:+.1f}%)")

    if verbose:
        click.echo(f"Saving warped surface: {output_path}")
    save_surface(output_path, warped, mov_poly)

    if deformation is not None:
        deformation_path = deformation
        _def_ext = Path(deformation_path).suffix.lower()
        if _def_ext not in {".vtk", ".vtp"}:
            deformation_path = str(Path(deformation_path).with_suffix(".vtk"))
            click.echo(
                f"Warning: '{_def_ext}' does not support vector point data — "
                f"saving deformation field as '{deformation_path}' instead."
            )
        if verbose:
            click.echo(f"Saving deformation field: {deformation_path}")
        save_deformation_vtk(deformation_path, mov_pts, mov_poly, def_field)

    if verbose:
        click.echo("Done.")


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------

@cli.command("downsample")
@click.argument("input_path",  metavar="INPUT",  type=click.Path(exists=True))
@click.argument("output_path", metavar="OUTPUT", required=False, default=None)
@click.option("--target-n", default=None, type=int, metavar="N",
              help="Target vertex count.")
@click.option("--ratio",    default=None, type=float, metavar="R",
              help="Target fraction of the original vertex count (e.g. 0.1 = 10 %).")
@_verbose_option
def downsample(input_path, output_path, target_n, ratio, quiet):
    """Decimate a surface to a lower vertex count.

    Exactly one of --target-n or --ratio must be provided.

    \b
    Examples:
      clarcs downsample brain.ply brain-5k.ply --target-n 5000
      clarcs downsample brain.ply brain-10pct.ply --ratio 0.1
    """
    verbose = not quiet

    if (target_n is None) == (ratio is None):
        raise click.UsageError("Provide exactly one of --target-n or --ratio.")

    if output_path is None:
        output_path = _default_output(input_path, "-downsampled")

    from pyclarcs.io import load_surface, save_surface
    from pyclarcs.mesh import decimate_surface

    if verbose:
        click.echo(f"Loading surface: {input_path}")
    pts, polygons = load_surface(input_path)
    n_orig = len(pts)
    if verbose:
        click.echo(f"  {n_orig} vertices, {len(polygons)} faces")

    n_target = target_n if target_n is not None else max(3, int(n_orig * ratio))

    if n_target >= n_orig:
        click.echo(
            f"Warning: target {n_target} ≥ current {n_orig} vertices — "
            "nothing to decimate, copying input."
        )
        pts_d, faces_d = pts, polygons
    else:
        if verbose:
            click.echo(f"Decimating to ~{n_target} vertices…")
        pts_d, faces_d = decimate_surface(pts, polygons, n_target)

    if verbose:
        click.echo(f"  {n_orig} → {len(pts_d)} vertices")
        click.echo(f"Saving: {output_path}")
    save_surface(output_path, pts_d, faces_d)

    if verbose:
        click.echo("Done.")


# ---------------------------------------------------------------------------
# atlas
# ---------------------------------------------------------------------------

_SURFACE_EXTS = {".vtk", ".vtp", ".vtu", ".ply", ".stl", ".obj"}


@cli.command("atlas")
@click.argument("subjects_dir", metavar="SUBJECTS_DIR",
                type=click.Path(exists=True, file_okay=False))
@click.argument("output_path", metavar="OUTPUT")
@click.option("--atlas-iter", default=3, show_default=True, type=int,
              help="Number of register-all → average cycles.")
@click.option("--save-registered", is_flag=True,
              help=(
                  "Also save the atlas warped toward each subject as "
                  "<OUTPUT_STEM>-registered-<subject_name><EXT>."
              ))
@click.option("--max-iter", default=80, show_default=True, type=int,
              help="EM iterations per registration.")
@click.option("--n-levels", default=None, type=int,
              help="Resolution levels. Auto-estimated from the first subject if omitted.")
@click.option("--no-symmetric", is_flag=True, default=False,
              help="Disable symmetric correspondences (Reg2). Enabled by default.")
@click.option("--no-tgd", is_flag=True, default=False,
              help="Disable the TGD geodesic shape prior (Reg3). Enabled by default.")
@click.option("--no-rkhs", is_flag=True, default=False,
              help="Disable RKHS Wu-kernel M-step; fall back to Laplacian.")
@_verbose_option
def atlas(subjects_dir, output_path, atlas_iter, save_registered,
          max_iter, n_levels, no_symmetric, no_tgd, no_rkhs, quiet):
    """Build a mean shape atlas from a directory of surfaces.

    Iteratively registers the current mean shape (moving) toward every
    subject (reference), then replaces the mean shape with the pointwise
    average of all warped copies.  Repeats for --atlas-iter cycles.

    The atlas topology matches the first subject (alphabetical order).
    Subjects may have different vertex counts.
    """
    verbose = not quiet

    import numpy as np
    from scipy.spatial import KDTree
    from pyclarcs.io import load_surface_with_normals, save_surface
    from pyclarcs.atlas import build_atlas

    subject_files = sorted(
        p for p in Path(subjects_dir).iterdir()
        if p.suffix.lower() in _SURFACE_EXTS
    )
    if len(subject_files) < 2:
        raise click.UsageError(
            f"SUBJECTS_DIR must contain at least 2 supported surfaces "
            f"({', '.join(sorted(_SURFACE_EXTS))}), "
            f"found {len(subject_files)}."
        )

    if verbose:
        click.echo(
            f"Loading {len(subject_files)} subjects from: {subjects_dir}"
        )
    subjects = []
    for p in subject_files:
        pts, polys, normals = load_surface_with_normals(str(p))
        subjects.append((pts, polys, normals))
        if verbose:
            click.echo(f"  {p.name}: {len(pts)} points, {len(polys)} faces")

    if n_levels is None:
        N = len(subjects[0][0])
        n_levels = 1 if N <= 5_000 else (2 if N <= 30_000 else 3)
        if verbose:
            click.echo(
                f"  auto n_levels={n_levels} ({N} template vertices)"
            )

    if verbose:
        click.echo(
            f"Building atlas: {len(subjects)} subjects, "
            f"{atlas_iter} atlas iteration(s), "
            f"{max_iter} EM iter/registration"
        )

    rms0_values = []
    template_pts = subjects[0][0]
    for sub_pts, _, _ in subjects[1:]:
        dists0, _ = KDTree(template_pts).query(sub_pts, k=1, workers=-1)
        rms0_values.append(float(np.sqrt(np.mean(dists0 ** 2))))
    rms0_mean = float(np.mean(rms0_values))

    mean_pts, mean_polygons, registered = build_atlas(
        subjects,
        atlas_iter=atlas_iter,
        verbose=verbose,
        n_levels=n_levels,
        max_iter=max_iter,
        symmetric=not no_symmetric,
        use_tgd=not no_tgd,
        use_rkhs=not no_rkhs,
    )

    if verbose:
        click.echo(f"Saving atlas: {output_path}")
    save_surface(output_path, mean_pts, mean_polygons)

    if save_registered:
        out_p = Path(output_path)
        out_stem = out_p.stem
        out_dir = out_p.parent
        out_ext = out_p.suffix or ".vtk"
        for warped, p in zip(registered, subject_files):
            reg_path = str(out_dir / f"{out_stem}-registered-{p.stem}{out_ext}")
            save_surface(reg_path, warped, mean_polygons)
            if verbose:
                click.echo(f"  Saved: {reg_path}")

    rms_values = []
    for (sub_pts, _, _), warped in zip(subjects, registered):
        dists, _ = KDTree(sub_pts).query(warped, k=1, workers=-1)
        rms_values.append(float(np.sqrt(np.mean(dists ** 2))))
    rms_mean = float(np.mean(rms_values))
    rms_max = float(np.max(rms_values))
    improvement = 100.0 * (rms0_mean - rms_mean) / rms0_mean if rms0_mean > 0 else 0.0
    click.echo(
        f"RMS (atlas→subjects): {rms0_mean:.4f} mm → {rms_mean:.4f} mm "
        f"({improvement:+.1f}%)  max={rms_max:.4f} mm"
    )

    if verbose:
        click.echo("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cli()


if __name__ == "__main__":
    main()
