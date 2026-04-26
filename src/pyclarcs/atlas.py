"""
Atlas construction by iterative non-rigid registration.

Algorithm (Abadie et al., MeshMed 2011):

    M = argmin_X  Σᵢ δ(X, Xⁱ)

Iterative procedure:

  1. Initialise M from the first subject (``subjects[0]``).
  2. For each atlas iteration:
       a. Register M (moving) toward every subject Xⁱ (reference).
       b. Replace M with the coordinate-wise mean of all warped copies.
  3. Return M and the warped copies from the last iteration.

The atlas always retains the topology of the initial subject.  Subjects may
have arbitrary and differing vertex counts because each is used only as a
registration target, never directly averaged.
"""
from __future__ import annotations

import numpy as np

from pyclarcs.nonrigid import register, apply_deformation
from pyclarcs.io import compute_surface_normals


def build_atlas(
    subjects: list[tuple[np.ndarray, list, np.ndarray]],
    *,
    atlas_iter: int = 3,
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
    verbose:
        Print progress to stdout.
    **reg_kwargs:
        Forwarded verbatim to :func:`nonrigid_icp_multires`
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

    mean_pts = subjects[0][0].copy()
    mean_polygons = subjects[0][1]
    mean_normals = subjects[0][2].copy()
    n = len(subjects)

    registered: list[np.ndarray] = []

    for atlas_it in range(atlas_iter):
        if verbose:
            print(f"[atlas] iteration {atlas_it + 1}/{atlas_iter}")

        warped = np.empty((n, len(mean_pts), 3))

        for k, (sub_pts, sub_polys, sub_normals) in enumerate(subjects):
            if verbose:
                print(f"[atlas]   subject {k + 1}/{n}")
            # On the first iteration the atlas equals subjects[0], so
            # registering it onto itself would be a no-op; skip it.
            if atlas_it == 0 and k == 0:
                warped[k] = mean_pts.copy()
                continue
            df = register(
                mean_pts, mean_normals,
                sub_pts, sub_normals,
                mean_polygons,
                sub_polys,
                verbose=verbose,
                **reg_kwargs,
            )
            warped[k] = apply_deformation(mean_pts, df)

        registered = [warped[k] for k in range(n)]
        mean_pts = warped.mean(axis=0)
        mean_normals = compute_surface_normals(mean_pts, mean_polygons)

    return mean_pts, mean_polygons, registered
