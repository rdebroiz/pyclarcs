"""
Diffeomorphic surface demons registration.

Implements a point-cloud variant of the diffeomorphic demons algorithm
(Vercauteren et al., NeuroImage 2009) adapted for triangulated surfaces.

ALGORITHM
=========
Instead of operating on a voxel grid, deformations are represented as
per-vertex vector fields.  The three classical operations (demons force,
Gaussian regularisation, exponential map) are all approximated via KDTree
queries on the unstructured point cloud.

Outer loop (n_iter iterations)
-------------------------------

1. Compute demons force at each transformed vertex T_i = x_i + d_i:

       y_i = argmin_{y ∈ ref} ||T_i - y||   (nearest-neighbour)

       Passive demons force (Thirion 1998):
           f_i = (y_i - T_i) / max(||y_i - T_i||² + σ_d², ε)

       Normal filter: multiply f_i by max(0, n_i · (y_i - T_i) / ||y_i - T_i||)
       so that forces pointing into the surface are suppressed.

2. Gaussian smoothing of the force field (replaces grid convolution):
   For each vertex i, compute a Gaussian-weighted average of f over its
   σ_s-neighbourhood:
       f_smooth[i] = Σ_j exp(-||x_j - x_i||² / (2σ_s²)) · f[j]
                     ─────────────────────────────────────────────
                     Σ_j exp(-||x_j - x_i||² / (2σ_s²))

3. Diffeomorphic update via stationary velocity field (SVF):
   The SVF v is updated with a BCH-like first-order approximation:
       v ← v + f_smooth
   The deformation is recovered as φ = exp(v) via scaling-and-squaring
   with n_steps=6 (64 compositions):
       exp(v) ≈ (id + v / 2^n)^(2^n)

   Each squaring step requires interpolating the field at displaced
   positions, which is done with IDW (k=4 nearest neighbours).

For n_steps=0 the algorithm reduces to the additive (non-diffeomorphic)
demons with Gaussian regularisation.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gaussian_smooth(
    field: np.ndarray,
    pts: np.ndarray,
    sigma: float,
    radius_factor: float = 2.5,
) -> np.ndarray:
    """Gaussian smoothing of a vector field on an unstructured point cloud.

    For each vertex i, averages the field over all vertices within
    ``radius = sigma * radius_factor`` using Gaussian weights
    ``w ∝ exp(-r²/(2σ²))``.

    Parameters
    ----------
    field  : (N, D) vector field
    pts    : (N, 3) vertex positions
    sigma  : smoothing bandwidth [same units as pts]
    radius_factor : search radius as a multiple of sigma

    Returns
    -------
    (N, D) smoothed field
    """
    N = len(pts)
    radius = sigma * radius_factor
    tree = KDTree(pts)
    nbrs_list = tree.query_ball_point(pts, radius, workers=-1)

    out = np.empty_like(field)
    inv_two_s2 = 1.0 / (2.0 * sigma * sigma)
    for i, nbrs in enumerate(nbrs_list):
        if not nbrs:
            out[i] = field[i]
            continue
        nbrs_arr = np.asarray(nbrs)
        dists2 = np.sum((pts[nbrs_arr] - pts[i]) ** 2, axis=1)
        w = np.exp(-dists2 * inv_two_s2)
        w_sum = w.sum()
        if w_sum < 1e-12:
            out[i] = field[i]
        else:
            out[i] = (w[:, np.newaxis] * field[nbrs_arr]).sum(axis=0) / w_sum
    return out


def _interpolate_at(
    field: np.ndarray,
    pts_field: np.ndarray,
    query_pts: np.ndarray,
    k: int = 4,
) -> np.ndarray:
    """IDW interpolation of a vector field at arbitrary positions.

    Parameters
    ----------
    field      : (N, D) field values at pts_field
    pts_field  : (N, 3) positions where the field is defined
    query_pts  : (M, 3) positions where interpolation is requested

    Returns
    -------
    (M, D) interpolated field
    """
    dists, idxs = KDTree(pts_field).query(query_pts, k=k, workers=-1)
    w = 1.0 / np.maximum(dists, 1e-10)   # (M, k)
    w /= w.sum(axis=1, keepdims=True)
    return np.einsum("nk,nkd->nd", w, field[idxs])


def _exp_map(
    velocity: np.ndarray,
    pts: np.ndarray,
    n_steps: int = 6,
) -> np.ndarray:
    """Compute the exponential map of a stationary velocity field.

    Uses scaling-and-squaring: φ = exp(v) ≈ (id + v/2^n)^(2^n).

    Each squaring step φ ← φ ∘ φ is computed as:
        new_def[i] = def[i] + interpolate(def, pts, pts + def)[i]

    Parameters
    ----------
    velocity : (N, 3) SVF
    pts      : (N, 3) mesh vertex positions
    n_steps  : number of squaring steps (2^n_steps compositions)

    Returns
    -------
    def_field : (N, 3) deformation corresponding to exp(v)
    """
    n = 2 ** n_steps
    phi = velocity / n   # φ₀ = id + v/2^n_steps

    for _ in range(n_steps):
        # φ ← φ ∘ φ
        phi_at_phi = _interpolate_at(phi, pts, pts + phi)
        phi = phi + phi_at_phi

    return phi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def surface_demons(
    mov_pts: np.ndarray,
    mov_normals: np.ndarray,
    ref_pts: np.ndarray,
    ref_normals: np.ndarray,
    *,
    n_iter: int = 80,
    sigma_demons: float = 3.0,
    sigma_smooth: float | None = None,
    dist_cutoff: float | None = None,
    normal_min_dot: float = 0.0,
    diffeomorphic: bool = True,
    n_exp_steps: int = 6,
    verbose: bool = True,
) -> np.ndarray:
    """Register a moving surface to a reference using surface demons.

    Parameters
    ----------
    mov_pts : (N, 3)
    mov_normals : (N, 3)
    ref_pts : (M, 3)
    ref_normals : (M, 3)
    n_iter : int
        Number of demons iterations.
    sigma_demons : float
        Regularisation constant in the demons force denominator.
        Smaller → stronger force (but more sensitive to noise).
        None → auto-estimated from the NN-distance distribution.
    sigma_smooth : float or None
        Gaussian smoothing bandwidth.  None → auto = ``sigma_demons * 1.5``.
    dist_cutoff : float or None
        Maximum NN search radius.  None → auto = ``sigma_demons * 4``.
    normal_min_dot : float
        Minimum dot product n_i · m_j for a correspondence to be accepted.
    diffeomorphic : bool
        If True, use SVF + exp-map (Vercauteren 2009).
        If False, additive demons (faster but not guaranteed bijective).
    n_exp_steps : int
        Number of scaling-and-squaring steps (only when diffeomorphic=True).
        6 steps = 64 compositions; 5 = 32 steps.
    verbose : bool

    Returns
    -------
    def_field : (N, 3) per-vertex deformation field.
        Warped surface = ``mov_pts + def_field``.
    """
    mov_pts     = np.asarray(mov_pts,     dtype=float)
    mov_normals = np.asarray(mov_normals, dtype=float)
    ref_pts     = np.asarray(ref_pts,     dtype=float)
    ref_normals = np.asarray(ref_normals, dtype=float)
    N = len(mov_pts)

    if sigma_smooth  is None: sigma_smooth  = sigma_demons * 1.5
    if dist_cutoff   is None: dist_cutoff   = sigma_demons * 4.0

    ref_tree = KDTree(ref_pts)

    def_field = np.zeros((N, 3), dtype=float)   # additive deformation
    velocity  = np.zeros((N, 3), dtype=float)   # SVF (diffeomorphic mode)

    for it in range(n_iter):
        transformed = mov_pts + def_field

        # ------------------------------------------------------------------
        # E-step: demons force — NN correspondence + Thirion formula
        # ------------------------------------------------------------------
        dists, idxs = ref_tree.query(transformed, k=1, workers=-1)

        diff = ref_pts[idxs] - transformed          # (N, 3) residual
        dist2 = (diff ** 2).sum(axis=1)             # (N,)

        # Passive demons force
        denom = np.maximum(dist2 + sigma_demons ** 2, 1e-10)
        force = diff / denom[:, np.newaxis]          # (N, 3)

        # Normal filter: only accept forces compatible with surface orientation
        n_dot = (ref_normals[idxs] * mov_normals).sum(axis=1)  # (N,)
        valid = (n_dot >= normal_min_dot) & (dists < dist_cutoff)
        force[~valid] = 0.0

        # ------------------------------------------------------------------
        # Gaussian smoothing of the force field
        # ------------------------------------------------------------------
        force_smooth = _gaussian_smooth(force, mov_pts, sigma_smooth)

        # ------------------------------------------------------------------
        # Update deformation
        # ------------------------------------------------------------------
        if diffeomorphic:
            velocity  += force_smooth
            def_field  = _exp_map(velocity, mov_pts, n_steps=n_exp_steps)
        else:
            def_field += force_smooth

        if verbose:
            rms_res = float(np.sqrt(np.mean(dist2[valid]))) if valid.any() else float("nan")
            print(
                f"  iter {it + 1:3d}/{n_iter}"
                f"  inliers={valid.sum()}/{N}"
                f"  RMS={rms_res:.3f}"
            )

    return def_field


def surface_demons_multires(
    mov_pts: np.ndarray,
    mov_normals: np.ndarray,
    ref_pts: np.ndarray,
    ref_normals: np.ndarray,
    mov_polygons: list,
    *,
    n_levels: int | None = None,
    target_n_coarsest: int = 2000,
    n_iter: int = 80,
    sigma_demons: float | None = None,
    sigma_smooth: float | None = None,
    dist_cutoff: float | None = None,
    diffeomorphic: bool = True,
    n_exp_steps: int = 6,
    verbose: bool = True,
) -> np.ndarray:
    """Multi-resolution surface demons registration.

    Builds the same coarse-to-fine hierarchy as ``nonrigid_icp_multires``
    and runs ``surface_demons`` at each level, warm-starting finer levels
    from the interpolated coarser deformation.

    Auto-estimates ``sigma_demons``, ``sigma_smooth``, and ``dist_cutoff``
    from the surface geometry when not provided.
    """
    from pyclarcs.mesh import decimate_surface, compute_vertex_normals, adjacency_csr
    from pyclarcs.nonrigid import _interpolate_field, estimate_registration_params

    mov_pts     = np.asarray(mov_pts,     dtype=float)
    mov_normals = np.asarray(mov_normals, dtype=float)
    ref_pts     = np.asarray(ref_pts,     dtype=float)
    ref_normals = np.asarray(ref_normals, dtype=float)
    N = len(mov_pts)

    # Auto n_levels
    if n_levels is None:
        n_levels = 1 if N <= 5_000 else (2 if N <= 30_000 else 3)

    # Build hierarchy (same as nonrigid_icp_multires)
    hierarchy: list[tuple[np.ndarray, np.ndarray, list]] = [
        (mov_pts, mov_normals, mov_polygons)
    ]
    for lev in range(1, n_levels):
        t = lev / (n_levels - 1) if n_levels > 1 else 1.0
        target_n = max(target_n_coarsest, int(N * (target_n_coarsest / N) ** t))
        if target_n >= len(hierarchy[0][0]) * 0.85:
            break
        if verbose:
            print(f"  [demons] building level {lev}: {N} → ~{target_n} vertices…")
        d_pts, d_faces = decimate_surface(mov_pts, mov_polygons, target_n)
        d_normals = compute_vertex_normals(d_pts, d_faces)
        hierarchy.append((d_pts, d_normals, d_faces))

    def_field_prev: np.ndarray | None = None
    pts_prev: np.ndarray | None       = None

    for idx in range(len(hierarchy) - 1, -1, -1):
        pts_l, normals_l, _ = hierarchy[idx]
        N_l = len(pts_l)
        is_finest = (idx == 0)

        # Warm-start
        if def_field_prev is None:
            init_l = np.zeros((N_l, 3), dtype=float)
        else:
            init_l = _interpolate_field(def_field_prev, pts_prev, pts_l)

        # Auto-estimate params from current residual
        transformed_l = pts_l + init_l
        auto = estimate_registration_params(transformed_l, ref_pts, max_iter=n_iter)
        sigma_d = sigma_demons if sigma_demons is not None else auto["sigma"]
        sigma_s = sigma_smooth if sigma_smooth is not None else sigma_d * 1.5
        cutoff  = dist_cutoff  if dist_cutoff  is not None else auto["dist_cutoff"]

        if verbose:
            label = "finest" if is_finest else f"level {idx}"
            print(
                f"\n  [demons] {label}  {N_l} vertices  σ_d={sigma_d:.2f}"
                f"  σ_s={sigma_s:.2f}  r={cutoff:.2f}"
                f"  {'diffeomorphic' if diffeomorphic else 'additive'}"
            )

        # Temporarily shift mov_pts_l by init_l and run demons from zero
        # (equivalent to warm-starting from init_l)
        df_l = surface_demons(
            pts_l + init_l, normals_l,
            ref_pts, ref_normals,
            n_iter=n_iter,
            sigma_demons=sigma_d,
            sigma_smooth=sigma_s,
            dist_cutoff=cutoff,
            diffeomorphic=diffeomorphic,
            n_exp_steps=n_exp_steps,
            verbose=verbose,
        )
        # Total deformation = warm-start + incremental
        def_field_l = init_l + df_l

        def_field_prev = def_field_l
        pts_prev       = pts_l

    return def_field_prev
