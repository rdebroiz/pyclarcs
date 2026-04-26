"""
Atlas construction by iterative non-rigid registration.

Algorithm (Abadie et al., MeshMed 2011):

    M = argmin_X  Σᵢ δ(X, Xⁱ)

Iterative procedure:

  1. Initialise M from the first subject (``subjects[0]``).
  2. For each atlas iteration:
       a. Optionally pre-align M and each subject to their canonical symmetry
          plane, then align M's centre of mass to each subject's.
       b. Register M (moving) toward every subject Xⁱ (reference).
       c. Replace M with the coordinate-wise mean of all warped copies.
  3. Optionally recenter the final M to its canonical symmetry plane.
  4. Return M and the warped copies from the last iteration.

The atlas always retains the topology of the initial subject.  Subjects may
have arbitrary and differing vertex counts because each is used only as a
registration target, never directly averaged.
"""
from __future__ import annotations

import numpy as np

from pyclarcs.nonrigid import register, apply_deformation
from pyclarcs.io import compute_surface_normals


# ---------------------------------------------------------------------------
# Pre-alignment helpers
# ---------------------------------------------------------------------------

def _sym_plane_fast(pts: np.ndarray):
    """Estimate the symmetry plane via inertia tensor (principal axes only)."""
    from pyclarcs.principal_axes import best_principal_axis_plane
    return best_principal_axis_plane(pts)


def _check_symmetry(
    pts: np.ndarray,
    plane,
    name: str,
    threshold: float = 0.05,
    n_sample: int = 2000,
) -> None:
    """Print a warning when a surface shows unexpectedly high asymmetry.

    Asymmetry is measured as the RMS nearest-neighbour distance between the
    surface and its mirror image across *plane*, normalised by the bounding-box
    diagonal.  Values above *threshold* (default 5 %) suggest the estimated
    symmetry plane may be unreliable.
    """
    from scipy.spatial import KDTree
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), size=min(n_sample, len(pts)), replace=False)
    sample = pts[idx]
    reflected = plane.apply(sample)
    dists, _ = KDTree(pts).query(reflected, k=1, workers=-1)
    rms = float(np.sqrt(np.mean(dists ** 2)))
    bbox_diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    ratio = rms / bbox_diag if bbox_diag > 0 else 0.0
    if ratio > threshold:
        print(
            f"  [prealign] WARNING {name}: high asymmetry "
            f"(RMS={rms:.1f} mm, {ratio * 100:.0f}% of bounding box). "
            f"Symmetry pre-alignment may be unreliable."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_atlas(
    subjects: list[tuple[np.ndarray, list, np.ndarray]],
    *,
    atlas_iter: int = 3,
    prealign: bool = True,
    verbose: bool = True,
    **reg_kwargs,
) -> tuple[np.ndarray, list, list[np.ndarray]]:
    """Compute a mean surface atlas from a population of surfaces.

    Parameters
    ----------
    subjects:
        ``(points, polygons, normals)`` tuples, one per subject.
        The first entry initialises the atlas topology and vertex count.
    atlas_iter:
        Number of register-all → average cycles.
    prealign:
        If True (default), rigidly pre-align each registration pair before
        calling :func:`register`:

        1. Estimate the bilateral symmetry plane of both the moving surface
           (mean shape) and the subject via principal-axis inertia tensor.
        2. Align both to the canonical plane (x = 0).
        3. Translate the moving surface so its centre of mass matches the
           subject's.

        Subject planes are computed once at load time; the mean-shape plane
        is recomputed once per atlas iteration.  A warning is printed when
        the symmetry residual exceeds 5 % of the bounding-box diagonal.

        At the end of all atlas iterations the final mean shape is recentered
        to its own canonical symmetry plane.
    verbose:
        Print progress to stdout.
    **reg_kwargs:
        Forwarded verbatim to :func:`register`
        (``n_levels``, ``max_iter``, ``symmetric``, ``use_tgd``, …).

    Returns
    -------
    mean_pts : ndarray (N, 3)
        Final atlas vertex positions.
    mean_polygons : list
        Mesh connectivity (topology of the first subject, unchanged).
    registered : list of ndarray (N, 3)
        Per-subject warped copies of the atlas from the last iteration.
        ``registered[i]`` is the atlas deformed to match subject *i*.
    """
    if len(subjects) < 2:
        raise ValueError("Atlas construction requires at least 2 subjects.")

    from pyclarcs.alignment import align_to_symmetry_plane, align_center_of_mass

    mean_pts = subjects[0][0].copy()
    mean_polygons = subjects[0][1]
    mean_normals = subjects[0][2].copy()
    n = len(subjects)

    # ------------------------------------------------------------------
    # Pre-compute subject symmetry planes and canonical-aligned subjects
    # (done once — subjects never change).
    # ------------------------------------------------------------------
    if prealign:
        if verbose:
            print("[atlas] Pre-computing subject symmetry planes…")
        sub_aligned: list[tuple[np.ndarray, list, np.ndarray]] = []
        for k, (sub_pts, sub_polys, sub_normals) in enumerate(subjects):
            plane = _sym_plane_fast(sub_pts)
            _check_symmetry(sub_pts, plane, name=f"subject {k + 1}")
            a_pts = align_to_symmetry_plane(sub_pts, plane)
            a_normals = compute_surface_normals(a_pts, sub_polys)
            sub_aligned.append((a_pts, sub_polys, a_normals))

    registered: list[np.ndarray] = []

    for atlas_it in range(atlas_iter):
        if verbose:
            print(f"[atlas] iteration {atlas_it + 1}/{atlas_iter}")

        # Canonical alignment of the current mean shape (once per iteration).
        if prealign:
            mean_plane = _sym_plane_fast(mean_pts)
            _check_symmetry(mean_pts, mean_plane, name="mean shape")
            aligned_mean_pts = align_to_symmetry_plane(mean_pts, mean_plane)
            # Rotation → recompute normals.
            aligned_mean_normals = compute_surface_normals(
                aligned_mean_pts, mean_polygons
            )

        warped = np.empty((n, len(mean_pts), 3))

        for k, (sub_pts, sub_polys, sub_normals) in enumerate(subjects):
            if verbose:
                print(f"[atlas]   subject {k + 1}/{n}")

            # First iteration, first subject: atlas == subject → no-op.
            if atlas_it == 0 and k == 0:
                warped[k] = (
                    align_center_of_mass(aligned_mean_pts, sub_aligned[0][0])
                    if prealign else mean_pts
                ).copy()
                continue

            if prealign:
                a_sub_pts, a_sub_polys, a_sub_normals = sub_aligned[k]
                # Centre-of-mass alignment: pure translation → normals unchanged.
                a_mean_pts = align_center_of_mass(aligned_mean_pts, a_sub_pts)
                a_mean_normals = aligned_mean_normals
            else:
                a_mean_pts, a_mean_normals = mean_pts, mean_normals
                a_sub_pts, a_sub_polys, a_sub_normals = sub_pts, sub_polys, sub_normals

            df = register(
                a_mean_pts, a_mean_normals,
                a_sub_pts, a_sub_normals,
                mean_polygons,
                a_sub_polys,
                verbose=verbose,
                **reg_kwargs,
            )
            warped[k] = apply_deformation(a_mean_pts, df)

        registered = [warped[k] for k in range(n)]
        mean_pts = warped.mean(axis=0)
        mean_normals = compute_surface_normals(mean_pts, mean_polygons)

    # ------------------------------------------------------------------
    # Recenter the final atlas to its canonical symmetry plane.
    # ------------------------------------------------------------------
    if prealign:
        final_plane = _sym_plane_fast(mean_pts)
        mean_pts = align_to_symmetry_plane(mean_pts, final_plane)
        mean_normals = compute_surface_normals(mean_pts, mean_polygons)
        if verbose:
            print("[atlas] Final mean shape recentered to canonical symmetry plane.")

    return mean_pts, mean_polygons, registered
