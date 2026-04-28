"""
Command-line interface for clarcs (Click-based).

    clarcs reorient     INPUT [OUTPUT] --axes X Y Z
    clarcs symplane     INPUT [OUTPUT] [--save-plane] [options]
    clarcs recenter     INPUT [OUTPUT] --plane PLANE.pl
    clarcs centerofmass INPUT [OUTPUT] --target TARGET
    clarcs normalize    INPUT [OUTPUT] --target TARGET
    clarcs nlregister   INPUT REF     [OUTPUT] [--deformation FIELD] [options]
    clarcs download mni        [OUTPUT] [options]
    clarcs download paleobrain [OUTPUT] [options]
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
# project-asym
# ---------------------------------------------------------------------------

@cli.command("project-asym")
@click.argument("subjects_dir", metavar="SUBJECTS_DIR",
                type=click.Path(exists=True, file_okay=False))
@click.argument("output_path",  metavar="OUTPUT")
@click.option("--atlas", "atlas_path", default=None,
              type=click.Path(exists=True), metavar="ATLAS",
              help="Pre-existing atlas surface.  If omitted, the atlas is built "
                   "from SUBJECTS_DIR using  clarcs atlas  parameters.")
@click.option("--save-atlas", default=None, metavar="PATH",
              help="Save the built atlas to this path (ignored when --atlas is given).")
@click.option("--atlas-iter", default=3, show_default=True, type=int,
              help="Atlas construction cycles (ignored when --atlas is given).")
@click.option("--no-prealign", is_flag=True, default=False,
              help="Disable sym-plane + CoM pre-alignment during atlas build.")
@click.option("--registered-dir", default=None,
              type=click.Path(exists=True, file_okay=False), metavar="DIR",
              help="Pre-computed registered atlas surfaces "
                   "(from  clarcs atlas --save-registered).  "
                   "Only used when --atlas is provided.  "
                   "Files matched to subjects by alphabetical sort order.")
@click.option("--asymmetry-dir", default=None,
              type=click.Path(exists=True, file_okay=False), metavar="DIR",
              help="Pre-computed asymmetry fields "
                   "(from  clarcs asymmetry).  "
                   "Files matched to subjects by alphabetical sort order.  "
                   "If omitted, asymmetry is computed on the fly.")
@click.option("--save-individual", is_flag=True,
              help="Save per-subject projected fields as "
                   "<OUTPUT_STEM>-<subject_name><EXT>.")
@click.option("--save-stats", is_flag=True,
              help="Save per-vertex norm statistics (std / min / max) alongside OUTPUT.")
@click.option("--max-iter",  default=80,  show_default=True, type=int,
              help="EM iterations per registration.")
@click.option("--n-levels",  default=None, type=int,
              help="Resolution levels. Auto-estimated if omitted.")
@click.option("--no-symmetric", is_flag=True, default=False)
@click.option("--no-tgd",       is_flag=True, default=False)
@click.option("--no-rkhs",      is_flag=True, default=False)
@_verbose_option
def project_asym(subjects_dir, output_path,
                 atlas_path, save_atlas, atlas_iter, no_prealign,
                 registered_dir, asymmetry_dir,
                 save_individual, save_stats,
                 max_iter, n_levels,
                 no_symmetric, no_tgd, no_rkhs, quiet):
    """Build an atlas and project each subject's asymmetry onto it.

    \b
    One-pass pipeline (no pre-computed files):
      clarcs project-asym subjects/ mean-asym.vtk

    \b
    With pre-existing atlas (skip atlas build):
      clarcs project-asym subjects/ mean-asym.vtk --atlas atlas.vtk

    \b
    Fully incremental (all pre-computed):
      clarcs project-asym subjects/ mean-asym.vtk \\
        --atlas atlas.vtk \\
        --registered-dir reg/ \\
        --asymmetry-dir  asym/

    \b
    Output:
      OUTPUT               Mean asymmetry field  (VECTORS on atlas)
      --save-atlas PATH    Also save the built atlas surface
      --save-stats         <OUTPUT_STEM>-std/min/max<EXT>
      --save-individual    <OUTPUT_STEM>-<subject><EXT> per subject
    """
    verbose = not quiet

    import numpy as np
    from pyclarcs.io import (
        load_surface_with_normals, load_vector_field,
        save_surface, save_deformation_vtk, compute_surface_normals,
    )
    from pyclarcs.nonrigid import register, apply_deformation
    from pyclarcs.atlas import build_atlas, project_asymmetry_to_atlas

    # ---- Validate OUTPUT -----------------------------------------------
    out_p = Path(output_path)
    if out_p.is_dir():
        raise click.UsageError(
            f"OUTPUT must be a file path, not a directory: {output_path!r}"
        )
    if not out_p.suffix:
        out_p = out_p.with_suffix(".vtk")
        output_path = str(out_p)
    if out_p.suffix.lower() not in {".vtk", ".vtp"}:
        raise click.UsageError(
            f"OUTPUT must be .vtk or .vtp (VECTORS require VTK format), "
            f"got {out_p.suffix!r}."
        )
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # ---- Enumerate subjects --------------------------------------------
    subject_files = sorted(
        p for p in Path(subjects_dir).iterdir()
        if p.suffix.lower() in _SURFACE_EXTS
    )
    if len(subject_files) < 2:
        raise click.UsageError(
            f"SUBJECTS_DIR must contain at least 2 surfaces, "
            f"found {len(subject_files)}."
        )

    # ---- Resolve n_levels from first subject ---------------------------
    if n_levels is None:
        _first_pts, _, _ = load_surface_with_normals(str(subject_files[0]))
        N0 = len(_first_pts)
        n_levels = 1 if N0 <= 5_000 else (2 if N0 <= 30_000 else 3)
        if verbose:
            click.echo(f"auto n_levels={n_levels} ({N0} vertices in first subject)")

    reg_kwargs = dict(
        n_levels=n_levels, max_iter=max_iter,
        symmetric=not no_symmetric,
        use_tgd=not no_tgd, use_rkhs=not no_rkhs,
    )

    # ---- Load all subjects ---------------------------------------------
    if verbose:
        click.echo(f"Loading {len(subject_files)} subjects from: {subjects_dir}")
    subjects = []
    for p in subject_files:
        pts, polys, normals = load_surface_with_normals(str(p))
        subjects.append((pts, polys, normals))
        if verbose:
            click.echo(f"  {p.name}: {len(pts)} pts")

    # ====================================================================
    # ATLAS — build or load
    # ====================================================================
    if atlas_path is None:
        # -- Build from scratch ------------------------------------------
        if registered_dir:
            click.echo(
                "Warning: --registered-dir is ignored when building the atlas "
                "from scratch (registered surfaces come from build_atlas)."
            )
        if verbose:
            click.echo(
                f"\nBuilding atlas: {len(subjects)} subjects, "
                f"{atlas_iter} iteration(s)…"
            )
        atlas_pts, atlas_poly, all_registered = build_atlas(
            subjects,
            atlas_iter=atlas_iter,
            prealign=not no_prealign,
            verbose=verbose,
            **reg_kwargs,
        )
        atlas_normals = compute_surface_normals(atlas_pts, atlas_poly)

        if save_atlas:
            _ap = Path(save_atlas)
            if not _ap.suffix:
                _ap = _ap.with_suffix(".vtk")
            _ap.parent.mkdir(parents=True, exist_ok=True)
            save_surface(str(_ap), atlas_pts, atlas_poly)
            if verbose:
                click.echo(f"Atlas saved: {_ap}")

    else:
        # -- Load pre-existing atlas -------------------------------------
        if verbose:
            click.echo(f"Loading atlas: {atlas_path}")
        atlas_pts, atlas_poly, atlas_normals = load_surface_with_normals(atlas_path)
        if verbose:
            click.echo(f"  {len(atlas_pts)} vertices")

        # Registered surfaces: pre-computed or compute on the fly
        def _list_dir(d):
            return sorted(
                p for p in Path(d).iterdir()
                if p.suffix.lower() in (_SURFACE_EXTS | {".vtk", ".vtp"})
            ) if d else []

        reg_files = _list_dir(registered_dir)
        if registered_dir and len(reg_files) != len(subject_files):
            raise click.UsageError(
                f"--registered-dir has {len(reg_files)} file(s) "
                f"but SUBJECTS_DIR has {len(subject_files)}."
            )

        all_registered = []
        for k, (sub_pts, sub_poly, sub_normals) in enumerate(subjects):
            if registered_dir:
                if verbose:
                    click.echo(f"  Loading registered [{k+1}]: {reg_files[k].name}")
                reg_pts, _, _ = load_surface_with_normals(str(reg_files[k]))
            else:
                if verbose:
                    click.echo(
                        f"  Registering atlas → subject {k+1}/{len(subjects)}…"
                    )
                df = register(
                    atlas_pts, atlas_normals,
                    sub_pts, sub_normals,
                    atlas_poly, sub_poly,
                    verbose=verbose,
                    **reg_kwargs,
                )
                reg_pts = apply_deformation(atlas_pts, df)
            all_registered.append(reg_pts)

    # ====================================================================
    # ASYMMETRY — load or compute per subject
    # ====================================================================
    def _list_asym(d):
        return sorted(
            p for p in Path(d).iterdir()
            if p.suffix.lower() in (_SURFACE_EXTS | {".vtk", ".vtp"})
        ) if d else []

    asym_files = _list_asym(asymmetry_dir)
    if asymmetry_dir and len(asym_files) != len(subject_files):
        raise click.UsageError(
            f"--asymmetry-dir has {len(asym_files)} file(s) "
            f"but SUBJECTS_DIR has {len(subject_files)}."
        )

    all_asym_fields = []
    all_subject_pts = []
    n = len(subjects)

    for k, (sub_pts, sub_poly, sub_normals) in enumerate(subjects):
        all_subject_pts.append(sub_pts)

        if asymmetry_dir:
            if verbose:
                click.echo(f"  Loading asymmetry [{k+1}]: {asym_files[k].name}")
            _, _, asym_field = load_vector_field(str(asym_files[k]))
        else:
            if verbose:
                click.echo(f"  Computing asymmetry [{k+1}/{n}]…")
            from pyclarcs.alignment import reflect_surface
            from pyclarcs.principal_axes import best_principal_axis_plane
            from pyclarcs.coarse import coarse_symmetry
            from pyclarcs.fine import em_icp_sym, em_icp_sym_corres

            plane = best_principal_axis_plane(sub_pts)
            plane = coarse_symmetry(sub_pts, plane, verbose=verbose)
            plane = em_icp_sym(sub_pts, plane, verbose=verbose)
            plane = em_icp_sym_corres(sub_pts, plane, verbose=verbose)

            mir_pts = reflect_surface(sub_pts, plane.n, plane.n * plane.d)
            mir_poly = [f[::-1] for f in sub_poly]
            mir_normals = compute_surface_normals(mir_pts, mir_poly)

            asym_field = register(
                mir_pts, mir_normals,
                sub_pts, sub_normals,
                mir_poly, sub_poly,
                verbose=verbose,
                **reg_kwargs,
            )
        all_asym_fields.append(asym_field)

    # ====================================================================
    # PROJECTION + AGGREGATION
    # ====================================================================
    if verbose:
        click.echo("\nProjecting asymmetry fields onto atlas…")
    projected = project_asymmetry_to_atlas(
        all_registered, all_asym_fields, all_subject_pts
    )

    arr = np.stack(projected)               # (n, N, 3)
    mean_field = arr.mean(axis=0)           # (N, 3)
    norms = np.linalg.norm(arr, axis=2)     # (n, N)

    click.echo(
        f"Mean asymmetry norm: "
        f"{float(np.linalg.norm(mean_field, axis=1).mean()):.4f} mm"
    )

    if verbose:
        click.echo(f"Saving mean asymmetry field: {output_path}")
    save_deformation_vtk(output_path, atlas_pts, atlas_poly, mean_field,
                         deformation_name="asymmetry_mean")

    if save_stats:
        for stat_name, stat_vals in (
            ("std", norms.std(axis=0)),
            ("min", norms.min(axis=0)),
            ("max", norms.max(axis=0)),
        ):
            stat_path = str(
                out_p.parent / f"{out_p.stem}-{stat_name}{out_p.suffix}"
            )
            save_surface(stat_path, atlas_pts, atlas_poly,
                         scalars=stat_vals, scalars_name=f"asymmetry_{stat_name}")
            if verbose:
                click.echo(f"  Saved {stat_name}: {stat_path}")

    if save_individual:
        for proj_i, sub_file in zip(projected, subject_files):
            ind_path = str(
                out_p.parent / f"{out_p.stem}-{sub_file.stem}{out_p.suffix}"
            )
            save_deformation_vtk(ind_path, atlas_pts, atlas_poly, proj_i,
                                 deformation_name="asymmetry")
            if verbose:
                click.echo(f"  Saved individual: {ind_path}")

    if verbose:
        click.echo("Done.")


# ---------------------------------------------------------------------------
# asymmetry
# ---------------------------------------------------------------------------

@cli.command("asymmetry")
@click.argument("input_path",  metavar="INPUT",  type=click.Path(exists=True))
@click.argument("output_path", metavar="OUTPUT", required=False, default=None)
@click.option("--plane", default=None, type=click.Path(exists=True), metavar="PLANE.pl",
              help="Symmetry plane (.pl). Estimated automatically if omitted.")
@click.option("--save-warped", is_flag=True,
              help="Also save the registered mirror as <OUTPUT_STEM>-warped<EXT>.")
@click.option("--max-iter",  default=80,   show_default=True, type=int,
              help="EM iterations.")
@click.option("--n-levels",  default=None, type=int,
              help="Resolution levels. Auto-estimated if omitted.")
@click.option("--no-symmetric", is_flag=True, default=False,
              help="Disable symmetric correspondences (Reg2).")
@click.option("--no-tgd",       is_flag=True, default=False,
              help="Disable TGD geodesic shape prior (Reg3).")
@click.option("--no-rkhs",      is_flag=True, default=False,
              help="Disable RKHS Wu-kernel M-step; fall back to Laplacian.")
@_verbose_option
def asymmetry(input_path, output_path, plane, save_warped,
              max_iter, n_levels, no_symmetric, no_tgd, no_rkhs, quiet):
    """Compute the pointwise asymmetry field of a bilateral surface.

    \b
    Pipeline:
      1. Estimate the bilateral symmetry plane (or load --plane).
      2. Reflect the surface across the plane.
      3. Non-rigidly register the reflected copy onto the original.
      4. Save the deformation field as VECTORS on the original geometry.

    The norm of each vector is the local asymmetry magnitude.  The output
    can be opened in ParaView and coloured by the vector norm, or warped by
    the vectors to visualise left-right differences.
    """
    verbose = not quiet

    if output_path is None:
        output_path = _default_output(input_path, "-asymmetry")

    # Deformation fields require VTK format to store VECTORS point data.
    _def_ext = Path(output_path).suffix.lower()
    if _def_ext not in {".vtk", ".vtp"}:
        output_path = str(Path(output_path).with_suffix(".vtk"))
        if verbose:
            click.echo(
                f"Note: asymmetry field requires VTK format — "
                f"saving as '{output_path}'."
            )

    import numpy as np
    from pyclarcs.io import load_surface_with_normals, save_surface, save_deformation_vtk
    from pyclarcs.symmetry import SymmetryPlane
    from pyclarcs.alignment import reflect_surface
    from pyclarcs.nonrigid import register, apply_deformation

    if verbose:
        click.echo(f"Loading surface: {input_path}")
    points, polygons, normals = load_surface_with_normals(input_path)
    if verbose:
        click.echo(f"  {len(points)} points, {len(polygons)} faces")

    # ---- Symmetry plane ------------------------------------------------
    if plane is not None:
        if verbose:
            click.echo(f"Loading symmetry plane: {plane}")
        sym_plane = SymmetryPlane.load(plane)
        if verbose:
            click.echo(f"  {sym_plane}")
    else:
        if verbose:
            click.echo("Estimating symmetry plane…")
        from pyclarcs.principal_axes import best_principal_axis_plane
        from pyclarcs.coarse import coarse_symmetry
        from pyclarcs.fine import em_icp_sym, em_icp_sym_corres

        sym_plane = best_principal_axis_plane(points)
        sym_plane = coarse_symmetry(points, sym_plane, verbose=verbose)
        sym_plane = em_icp_sym(points, sym_plane, verbose=verbose)
        sym_plane = em_icp_sym_corres(points, sym_plane, verbose=verbose)
        if verbose:
            click.echo(f"  {sym_plane}")

    # ---- Mirror --------------------------------------------------------
    if verbose:
        click.echo("Reflecting surface…")
    plane_normal = sym_plane.n
    plane_point  = sym_plane.n * sym_plane.d
    mir_pts = reflect_surface(points, plane_normal, plane_point)
    # Reflection reverses winding → flip faces to restore outward normals.
    mir_polygons = [f[::-1] for f in polygons]
    from pyclarcs.io import compute_surface_normals
    mir_normals = compute_surface_normals(mir_pts, mir_polygons)

    # ---- n_levels ------------------------------------------------------
    if n_levels is None:
        N = len(mir_pts)
        n_levels = 1 if N <= 5_000 else (2 if N <= 30_000 else 3)
        if verbose:
            click.echo(f"  auto n_levels={n_levels} ({N} vertices)")

    # ---- Non-rigid registration: mirror → original ---------------------
    if verbose:
        click.echo(
            f"Non-rigid registration (mirror → original)  "
            f"{n_levels} level(s)  {max_iter} iter…"
        )
    def_field = register(
        mir_pts, mir_normals,
        points, normals,
        mir_polygons,
        polygons,
        n_levels=n_levels,
        max_iter=max_iter,
        symmetric=not no_symmetric,
        use_tgd=not no_tgd,
        use_rkhs=not no_rkhs,
        verbose=verbose,
    )

    # ---- RMS report (always printed) -----------------------------------
    from scipy.spatial import KDTree
    ref_tree = KDTree(points)
    dists0, _ = ref_tree.query(mir_pts, k=1, workers=-1)
    rms0 = float(np.sqrt(np.mean(dists0 ** 2)))
    warped_mir = apply_deformation(mir_pts, def_field)
    dists,  _ = ref_tree.query(warped_mir, k=1, workers=-1)
    rms  = float(np.sqrt(np.mean(dists  ** 2)))
    improvement = 100.0 * (rms0 - rms) / rms0 if rms0 > 0 else 0.0
    click.echo(
        f"RMS mirror→original: {rms0:.4f} mm → {rms:.4f} mm  ({improvement:+.1f}%)"
    )

    # ---- Save asymmetry field on original geometry ---------------------
    # def_field[i] displaces mirrored vertex i toward original vertex i.
    # Vertex indices match (same topology), so the field is displayed on
    # the original surface for intuitive visualisation.
    if verbose:
        click.echo(f"Saving asymmetry field: {output_path}")
    save_deformation_vtk(output_path, points, polygons, def_field,
                         deformation_name="asymmetry")

    if save_warped:
        warped_path = str(
            Path(output_path).with_stem(Path(output_path).stem + "-warped")
        )
        if verbose:
            click.echo(f"Saving registered mirror: {warped_path}")
        save_surface(warped_path, warped_mir, mir_polygons)

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
@click.option("--no-prealign", is_flag=True, default=False,
              help=(
                  "Disable symmetry-plane + centre-of-mass pre-alignment "
                  "before each registration. Enabled by default."
              ))
@click.option("--no-symmetric", is_flag=True, default=False,
              help="Disable symmetric correspondences (Reg2). Enabled by default.")
@click.option("--no-tgd", is_flag=True, default=False,
              help="Disable the TGD geodesic shape prior (Reg3). Enabled by default.")
@click.option("--no-rkhs", is_flag=True, default=False,
              help="Disable RKHS Wu-kernel M-step; fall back to Laplacian.")
@_verbose_option
def atlas(subjects_dir, output_path, atlas_iter, save_registered,
          max_iter, n_levels, no_prealign, no_symmetric, no_tgd, no_rkhs, quiet):
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

    # ------------------------------------------------------------------
    # Resolve and validate OUTPUT early so we fail fast, before the
    # expensive atlas computation.
    # ------------------------------------------------------------------
    out_p = Path(output_path)
    if out_p.is_dir():
        raise click.UsageError(
            f"OUTPUT must be a file path, not a directory: {output_path!r}\n"
            f"Hint: {output_path}/atlas.vtk"
        )
    if not out_p.suffix:
        out_p = out_p.with_suffix(".vtk")
        output_path = str(out_p)
        if verbose:
            click.echo(f"  (no extension in OUTPUT — defaulting to .vtk: {output_path})")
    _out_fmt = out_p.suffix.lower()
    _write_exts = {".vtk", ".vtp", ".ply", ".stl", ".obj"}
    if _out_fmt not in _write_exts:
        raise click.UsageError(
            f"Unsupported output format {out_p.suffix!r}. "
            f"Supported: {', '.join(sorted(_write_exts))}"
        )
    out_p.parent.mkdir(parents=True, exist_ok=True)

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
        prealign=not no_prealign,
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
        for warped, p in zip(registered, subject_files):
            reg_path = str(
                out_p.parent / f"{out_p.stem}-registered-{p.stem}{out_p.suffix}"
            )
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
# download
# ---------------------------------------------------------------------------

def _decimate_surface_inplace(path: Path, target_n: int, verbose: bool) -> None:
    from pyclarcs.io import load_surface, save_surface
    from pyclarcs.mesh import decimate_surface

    pts, faces = load_surface(str(path))
    n_orig = len(pts)
    if n_orig <= target_n:
        if verbose:
            click.echo(f"  {path.name}: {n_orig} ≤ {target_n} vertices, skipped.")
        return
    pts_d, faces_d = decimate_surface(pts, faces, target_n)
    save_surface(str(path), pts_d, faces_d)
    if verbose:
        click.echo(f"  {path.name}: {n_orig} → {len(pts_d)} vertices.")


@cli.group("download")
def download():
    """Download reference datasets."""


_MNI152_URLS = {
    "L": (
        "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
        "tpl-MNI152NLin2009cAsym_hemi-L_desc-pial_surf.surf.gii"
    ),
    "R": (
        "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
        "tpl-MNI152NLin2009cAsym_hemi-R_desc-pial_surf.surf.gii"
    ),
}


def _load_gifti(path: Path) -> "tuple[np.ndarray, list[list[int]]]":
    """Read a GIfTI surface file (.surf.gii) with nibabel."""
    try:
        import nibabel as nib
    except ImportError:
        raise click.ClickException(
            "nibabel is required to read MNI152 GIfTI surfaces:\n"
            "  pip install nibabel"
        )
    import numpy as np

    img = nib.load(str(path))
    pts   = img.darrays[0].data.astype(float)        # (N, 3) float32 coords
    tris  = img.darrays[1].data.astype(int)           # (F, 3) int32 triangles
    faces = tris.tolist()
    return pts, faces


@download.command("mni")
@click.argument("output_dir", metavar="OUTPUT", default=".", required=False)
@click.option("--target-n", default=10_000, show_default=True, type=int, metavar="N",
              help="Decimate the merged surface to ~N vertices. Pass 0 to skip.")
@click.option("--atlas", "with_atlas", is_flag=True,
              help="Also download the MNI152NLin2009cAsym pial atlas (TemplateFlow). "
                   "Requires nibabel (pip install nibabel).")
@click.option("--force", is_flag=True,
              help="Re-download even if files are already cached.")
@_verbose_option
def download_mni(output_dir, target_n, with_atlas, force, quiet):
    """Download MNI pial surfaces to OUTPUT.

    \b
    Without --atlas (Brain for Blender, A. Winkler — CC BY 4.0):
      endocranium_mni_pial.vtk       full-resolution LH+RH merged surface
      endocranium_mni_pial_10k.vtk   decimated to ~10 000 vertices

    \b
    With --atlas (MNI152NLin2009cAsym pial, TemplateFlow — CC0):
      mni152_pial_L.vtk   left hemisphere pial surface
      mni152_pial_R.vtk   right hemisphere pial surface
      mni152_pial.vtk     merged LH+RH atlas surface
    """
    import subprocess
    import urllib.request
    import numpy as np

    verbose = not quiet
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ pial
    _MNI_URL = (
        "https://s3.us-east-2.amazonaws.com/brainder/software/"
        "brain4blender/smallfiles/pial_Full_ply.tar.bz2"
    )
    _cache_dir = Path.home() / ".cache" / "pyclarcs"
    _archive   = _cache_dir / "pial_Full_ply.tar.bz2"
    _cache     = _cache_dir / "pial_Full_ply"
    lh_ply     = _cache / "lh.pial.ply"
    rh_ply     = _cache / "rh.pial.ply"

    def _reporthook(count, block_size, total):
        if total > 0:
            pct = min(100, count * block_size * 100 // total)
            click.echo(f"\r  {pct:3d}%", nl=False)

    if lh_ply.exists() and rh_ply.exists() and not force:
        if verbose:
            click.echo(f"Cache found: {_cache}")
    else:
        _cache_dir.mkdir(parents=True, exist_ok=True)
        if not _archive.exists() or force:
            if verbose:
                click.echo(f"Downloading MNI pial surface from brainder.org…\n  {_MNI_URL}")
            urllib.request.urlretrieve(
                _MNI_URL, _archive,
                reporthook=_reporthook if verbose else None,
            )
            if verbose:
                click.echo("")
        if verbose:
            click.echo("Extracting archive…")
        result = subprocess.run(
            ["tar", "xjf", str(_archive), "-C", str(_cache_dir)],
            capture_output=True,
        )
        if result.returncode != 0:
            raise click.ClickException(
                f"Extraction failed: {result.stderr.decode()}\n"
                "Make sure 'tar' and 'bzip2' are installed."
            )
        if verbose:
            click.echo(f"  → {_cache}")

    from pyclarcs.io import load_surface, save_surface

    if verbose:
        click.echo("Loading LH + RH pial surfaces…")
    lh_pts, lh_faces = load_surface(str(lh_ply))
    rh_pts, rh_faces = load_surface(str(rh_ply))

    offset = len(lh_pts)
    rh_faces_shifted = [[v + offset for v in f] for f in rh_faces]
    all_pts   = np.vstack([lh_pts, rh_pts])
    all_faces = lh_faces + rh_faces_shifted

    out = out_dir / "endocranium_mni_pial.vtk"
    save_surface(str(out), all_pts, all_faces)
    if verbose:
        click.echo(f"  → {out.name}  ({len(all_pts):,} pts, {len(all_faces):,} faces)")

    if target_n:
        from pyclarcs.mesh import decimate_surface
        n = target_n
        suffix = f"{n // 1000}k" if n % 1000 == 0 else str(n)
        if verbose:
            click.echo(f"Decimating to ~{n} vertices…")
        dec_pts, dec_faces = decimate_surface(all_pts, all_faces, n)
        out2 = out_dir / f"endocranium_mni_pial_{suffix}.vtk"
        save_surface(str(out2), dec_pts, dec_faces)
        if verbose:
            click.echo(f"  → {out2.name}  ({len(dec_pts):,} pts, {len(dec_faces):,} faces)")

    # ----------------------------------------------------------------- atlas
    if with_atlas:
        _atlas_cache = _cache_dir / "mni152"
        _atlas_cache.mkdir(parents=True, exist_ok=True)

        hemi_pts:   dict[str, np.ndarray]    = {}
        hemi_faces: dict[str, list[list[int]]] = {}

        for hemi, url in _MNI152_URLS.items():
            dest = _atlas_cache / f"mni152_pial_{hemi}.surf.gii"
            if not dest.exists() or force:
                if verbose:
                    click.echo(f"Downloading MNI152 {hemi} pial from TemplateFlow…\n  {url}")
                urllib.request.urlretrieve(
                    url, dest,
                    reporthook=_reporthook if verbose else None,
                )
                if verbose:
                    click.echo("")
            elif verbose:
                click.echo(f"Cache found: {dest.name}")

            pts, faces = _load_gifti(dest)
            hemi_pts[hemi]   = pts
            hemi_faces[hemi] = faces

            out_h = out_dir / f"mni152_pial_{hemi}.vtk"
            save_surface(str(out_h), pts, faces)
            if verbose:
                click.echo(f"  → {out_h.name}  ({len(pts):,} pts, {len(faces):,} faces)")

        # merged atlas
        offset_atlas = len(hemi_pts["L"])
        r_shifted = [[v + offset_atlas for v in f] for f in hemi_faces["R"]]
        atlas_pts   = np.vstack([hemi_pts["L"],   hemi_pts["R"]])
        atlas_faces = hemi_faces["L"] + r_shifted
        out_atlas = out_dir / "mni152_pial.vtk"
        save_surface(str(out_atlas), atlas_pts, atlas_faces)
        if verbose:
            click.echo(
                f"  → {out_atlas.name}  "
                f"({len(atlas_pts):,} pts, {len(atlas_faces):,} faces)"
            )

    click.echo(f"Done. Files ready in {out_dir}.")


@download.command("paleobrain")
@click.argument("output_dir", metavar="OUTPUT", default="paleobrain", required=False)
@click.option("--n", default=75, show_default=True, type=click.IntRange(1, 75), metavar="N",
              help="Number of subjects to download (1–75).")
@click.option("--type", "kind",
              type=click.Choice(["brain", "endocast", "all"]), default="all",
              show_default=True,
              help="Which surfaces to download.")
@click.option("--target-n", default=None, type=int, metavar="N",
              help="After download, decimate each surface to ~N vertices.")
@click.option("--force", is_flag=True,
              help="Re-download files that already exist.")
@_verbose_option
def download_paleobrain(output_dir, n, kind, target_n, force, quiet):
    """Download the PaleoBRAIN surface dataset to OUTPUT.

    \b
    75 brain surfaces (B01–B75.ply) and 75 endocasts (E01–E75.ply).
    Reference: Balzeau A. (2025). doi:10.48579/PRO/KZMMLM

    \b
    Examples:
      clarcs download paleobrain data/paleobrain
      clarcs download paleobrain data/paleobrain --n 10 --type brain
      clarcs download paleobrain data/paleobrain --n 10 --target-n 5000
    """
    import urllib.request
    import urllib.error

    verbose = not quiet

    _BASE_URL = "https://data.indores.fr/api/access/datafile/"
    _BRAIN_IDS: dict[str, int] = {
        "B01.ply": 25532, "B02.ply": 25611, "B03.ply": 25636, "B04.ply": 25646,
        "B05.ply": 25648, "B06.ply": 25607, "B07.ply": 25608, "B08.ply": 25580,
        "B09.ply": 25618, "B10.ply": 25619, "B11.ply": 25519, "B12.ply": 25556,
        "B13.ply": 25585, "B14.ply": 25574, "B15.ply": 25510, "B16.ply": 25605,
        "B17.ply": 25538, "B18.ply": 25591, "B19.ply": 25517, "B20.ply": 25638,
        "B21.ply": 25520, "B22.ply": 25613, "B23.ply": 25554, "B24.ply": 25612,
        "B25.ply": 25551, "B26.ply": 25557, "B27.ply": 25634, "B28.ply": 25629,
        "B29.ply": 25600, "B30.ply": 25576, "B31.ply": 25547, "B32.ply": 25560,
        "B33.ply": 25620, "B34.ply": 25578, "B35.ply": 25575, "B36.ply": 25540,
        "B37.ply": 25569, "B38.ply": 25584, "B39.ply": 25609, "B40.ply": 25573,
        "B41.ply": 25524, "B42.ply": 25643, "B43.ply": 25653, "B44.ply": 25587,
        "B45.ply": 25606, "B46.ply": 25604, "B47.ply": 25637, "B48.ply": 25601,
        "B49.ply": 25610, "B50.ply": 25539, "B51.ply": 25533, "B52.ply": 25603,
        "B53.ply": 25602, "B54.ply": 25617, "B55.ply": 25571, "B56.ply": 25526,
        "B57.ply": 25635, "B58.ply": 25566, "B59.ply": 25647, "B60.ply": 25625,
        "B61.ply": 25531, "B62.ply": 25509, "B63.ply": 25595, "B64.ply": 25577,
        "B65.ply": 25657, "B66.ply": 25568, "B67.ply": 25514, "B68.ply": 25645,
        "B69.ply": 25616, "B70.ply": 25511, "B71.ply": 25642, "B72.ply": 25624,
        "B73.ply": 25630, "B74.ply": 25552, "B75.ply": 25558,
    }
    _ENDOCAST_IDS: dict[str, int] = {
        "E01.ply": 25581, "E02.ply": 25599, "E03.ply": 25621, "E04.ply": 25516,
        "E05.ply": 25628, "E06.ply": 25632, "E07.ply": 25543, "E08.ply": 25582,
        "E09.ply": 25541, "E10.ply": 25650, "E11.ply": 25555, "E12.ply": 25631,
        "E13.ply": 25596, "E14.ply": 25523, "E15.ply": 25570, "E16.ply": 25525,
        "E17.ply": 25655, "E18.ply": 25542, "E19.ply": 25588, "E20.ply": 25652,
        "E21.ply": 25534, "E22.ply": 25594, "E23.ply": 25586, "E24.ply": 25537,
        "E25.ply": 25598, "E26.ply": 25615, "E27.ply": 25590, "E28.ply": 25513,
        "E29.ply": 25545, "E30.ply": 25656, "E31.ply": 25521, "E32.ply": 25527,
        "E33.ply": 25658, "E34.ply": 25579, "E35.ply": 25639, "E36.ply": 25614,
        "E37.ply": 25640, "E38.ply": 25654, "E39.ply": 25593, "E40.ply": 25512,
        "E41.ply": 25562, "E42.ply": 25651, "E43.ply": 25553, "E44.ply": 25589,
        "E45.ply": 25535, "E46.ply": 25518, "E47.ply": 25563, "E48.ply": 25529,
        "E49.ply": 25550, "E50.ply": 25561, "E51.ply": 25544, "E52.ply": 25641,
        "E53.ply": 25508, "E54.ply": 25564, "E55.ply": 25528, "E56.ply": 25592,
        "E57.ply": 25623, "E58.ply": 25548, "E59.ply": 25597, "E60.ply": 25626,
        "E61.ply": 25567, "E62.ply": 25522, "E63.ply": 25549, "E64.ply": 25644,
        "E65.ply": 25546, "E66.ply": 25559, "E67.ply": 25530, "E68.ply": 25572,
        "E69.ply": 25649, "E70.ply": 25622, "E71.ply": 25565, "E72.ply": 25633,
        "E73.ply": 25536, "E74.ply": 25583, "E75.ply": 25515,
    }

    entries: list[tuple[str, int]] = []
    if kind in ("brain", "all"):
        entries += [(k, _BRAIN_IDS[k]) for k in sorted(_BRAIN_IDS)[:n]]
    if kind in ("endocast", "all"):
        entries += [(k, _ENDOCAST_IDS[k]) for k in sorted(_ENDOCAST_IDS)[:n]]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        click.echo(f"PaleoBRAIN — {len(entries)} file(s) → {out_dir}")
        click.echo("doi:10.48579/PRO/KZMMLM")
        if target_n:
            click.echo(f"Resampling to ~{target_n} vertices after download.")

    def _reporthook(count, block_size, total):
        if total > 0:
            pct = min(100, count * block_size * 100 // total)
            click.echo(f"\r  {pct:3d}%", nl=False)

    errors: list[str] = []
    for i, (name, file_id) in enumerate(entries, 1):
        dest = out_dir / name
        prefix = f"[{i}/{len(entries)}] {name}"

        if dest.exists() and not force:
            if verbose:
                click.echo(f"{prefix}  (exists, skipped)")
            if target_n:
                _decimate_surface_inplace(dest, target_n, verbose)
            continue

        if verbose:
            click.echo(f"{prefix}  downloading…")
        tmp = dest.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(
                f"{_BASE_URL}{file_id}", tmp,
                reporthook=_reporthook if verbose else None,
            )
            if verbose:
                click.echo("")
            tmp.rename(dest)
        except urllib.error.URLError as exc:
            tmp.unlink(missing_ok=True)
            click.echo(f"  ERROR: {exc}", err=True)
            errors.append(name)
            continue

        if target_n:
            _decimate_surface_inplace(dest, target_n, verbose)

    click.echo(
        f"\nDone. {len(entries) - len(errors)}/{len(entries)} file(s) ready in {out_dir}."
    )
    if errors:
        click.echo(f"Failed: {', '.join(errors)}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cli()


if __name__ == "__main__":
    main()
