"""
Non-rigid EM-ICP surface registration.

THEORETICAL BACKGROUND
=======================
This module implements a non-rigid registration algorithm from the CLARCS
framework.  It is a variant of the EM-ICP method where the transform is a
smooth, per-vertex deformation field regularised by a first-order graph
Laplacian prior on the mesh topology.

ALGORITHM
=========

Notation
--------
  x_i  : vertex i of the moving surface (i = 0 … N-1)
  n_i  : unit normal at x_i
  y_j  : vertex j of the reference surface (j = 0 … M-1)
  m_j  : unit normal at y_j
  d_i  : deformation vector attached to x_i  (unknown, initialised to 0)
  N_i  : set of mesh-edge neighbours of x_i

Outer loop (max_iter iterations)
---------------------------------

E-step — doubly-stochastic fuzzy correspondences
  For each vertex i:
      T_i = x_i + d_i                       (current transformed position)

      For each y_j within radius r of T_i whose normal is compatible
      (n_i · m_j ≥ 0):
          w_ij = exp( −‖T_i − y_j‖ / σ )

      Row sum:  sR_i = Σ_j w_ij
      Col sum:  sC_j = Σ_i w_ij

      Doubly-stochastic weight:
          ṽ_ij = w_ij / sC_j  +  w_ij / sR_i

      Total weight:  W_i  = Σ_j ṽ_ij
      Target point:  ȳ_i  = (Σ_j ṽ_ij y_j) / W_i

  Implementation: a single chunked KDTree pass builds the COO arrays
  (rows, cols, weights).  Row/col sums and barycentres are then computed
  with numpy bincount — no N×M sparse matrix is ever constructed.

M-step — preconditioned conjugate gradient
  Solves the sparse linear system  M · d[:,k] = b[:,k]  independently
  for each coordinate k ∈ {0, 1, 2}:

      M   =  diag(W_i + β·|N_i|)  −  β · A          (symmetric PSD)
      b_i =  W_i · (ȳ_i − x_i)_k

  Preconditioning with the diagonal of M (Jacobi preconditioner) gives
  O(√κ) convergence vs. O(κ) for unaccelerated Jacobi.  The previous
  outer-iteration solution warm-starts the solve.

Annealing
  σ ← max(σ / 2,  σ_min)   every `period_sigma` outer iterations.

DEFAULT PARAMETERS (matching the original C++ NonLinearRegistration)
  sigma        = 3.0    initial bandwidth (same units as coordinates)
  beta         = 100.0  regularisation weight
  dist_cutoff  = 15.0   search radius
  max_iter     = 80
  icm_iter     = 50     max CG iterations per outer iteration
  period_sigma = 40     sigma halved every 40 outer iterations
  sigma_min    = 0.1
  e_chunk      = 2000   vertices per KDTree query batch
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix, diags as sp_diags
from scipy.sparse.linalg import cg as sp_cg


# ---------------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------------

def estimate_registration_params(
    mov_pts: np.ndarray,
    ref_pts: np.ndarray,
    *,
    max_iter: int = 80,
    sigma_min: float = 0.1,
    n_sample: int = 2000,
    seed: int = 0,
) -> dict:
    """Estimate good EM-ICP parameters from the two input surfaces.

    A random subsample of the moving surface is queried against the
    reference KDTree to obtain the nearest-neighbour distance distribution,
    which directly characterises the initial surface-to-surface gap.

    Parameters
    ----------
    mov_pts : ndarray (N, 3)
    ref_pts : ndarray (M, 3)
    max_iter : int
        Outer iterations (needed to compute period_sigma).
    sigma_min : float
        Annealing floor (needed to compute period_sigma).
    n_sample : int
        Number of moving points to subsample for the distance estimate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys ``sigma``, ``dist_cutoff``, ``period_sigma``.

    Notes
    -----
    Heuristics
      sigma        = 50th percentile of NN distances (median gap)
      dist_cutoff  = 99th percentile of NN distances × 1.5
                     (floor: sigma × 3)
      period_sigma = max_iter // ceil(log2(sigma / sigma_min))
                     (number of halvings needed from sigma to sigma_min,
                      spread evenly across the outer iterations)
    """
    rng = np.random.default_rng(seed)
    mov_pts = np.asarray(mov_pts, dtype=float)
    ref_pts = np.asarray(ref_pts, dtype=float)

    idx = rng.choice(len(mov_pts), size=min(n_sample, len(mov_pts)), replace=False)
    nn_dists, _ = KDTree(ref_pts).query(mov_pts[idx], k=1, workers=-1)

    sigma = float(np.percentile(nn_dists, 50))
    sigma = max(sigma, sigma_min * 2)          # floor: at least two halvings

    dist_cutoff = float(np.percentile(nn_dists, 99)) * 1.5
    dist_cutoff = max(dist_cutoff, sigma * 3)  # always at least 3σ

    n_halvings = max(1, math.ceil(math.log2(sigma / sigma_min)))
    period_sigma = max(1, max_iter // n_halvings)

    return {
        "sigma":        round(sigma,       4),
        "dist_cutoff":  round(dist_cutoff, 4),
        "period_sigma": period_sigma,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def nonrigid_icp(
    mov_pts: np.ndarray,
    mov_normals: np.ndarray,
    ref_pts: np.ndarray,
    ref_normals: np.ndarray,
    adjacency: csr_matrix,
    *,
    init_def_field: np.ndarray | None = None,
    sigma: float = 3.0,
    beta: float = 100.0,
    dist_cutoff: float = 15.0,
    max_iter: int = 80,
    icm_iter: int = 50,
    period_sigma: int = 40,
    sigma_min: float = 0.1,
    e_chunk: int = 2000,
    verbose: bool = True,
) -> np.ndarray:
    """Register a moving surface onto a reference using non-rigid EM-ICP.

    The algorithm iterates between:
      - computing doubly-stochastic fuzzy correspondences (E-step), and
      - solving for the deformation field with Laplacian regularisation
        via preconditioned conjugate gradient (M-step).

    Parameters
    ----------
    mov_pts : ndarray (N, 3)
        Vertices of the moving surface.
    mov_normals : ndarray (N, 3)
        Unit normals at each moving vertex.
    ref_pts : ndarray (M, 3)
        Vertices of the reference surface.
    ref_normals : ndarray (M, 3)
        Unit normals at each reference vertex.
    adjacency : csr_matrix (N, N)
        Symmetric mesh adjacency matrix (from ``mesh.adjacency_csr``).
    init_def_field : ndarray (N, 3) or None
        Initial deformation field.  None initialises to zero (default).
        Used by ``nonrigid_icp_multires`` to warm-start the finest level
        from the interpolated coarse solution.
    sigma : float
        Initial bandwidth of the exponential correspondence kernel.
        Should be on the order of the expected initial surface-to-surface
        distance (e.g. set to dist_cutoff / 3 or larger for coarse alignment).
    beta : float
        Regularisation weight (higher → smoother deformation).
    dist_cutoff : float
        Maximum search radius for candidate correspondences.
    max_iter : int
        Number of outer EM iterations.
    icm_iter : int
        Maximum number of conjugate gradient iterations per outer iteration.
    period_sigma : int
        Number of outer iterations between each halving of sigma.
    sigma_min : float
        Minimum value of sigma (annealing floor).
    e_chunk : int
        Number of vertices processed per KDTree query batch in the E-step.
        Has no effect on results; lower values reduce peak memory.
    verbose : bool
        Print iteration progress.

    Returns
    -------
    def_field : ndarray (N, 3)
        Per-vertex deformation field.
        The warped surface is ``mov_pts + def_field``.
    """
    mov_pts = np.asarray(mov_pts, dtype=float)
    mov_normals = np.asarray(mov_normals, dtype=float)
    ref_pts = np.asarray(ref_pts, dtype=float)
    ref_normals = np.asarray(ref_normals, dtype=float)

    N = len(mov_pts)
    M = len(ref_pts)

    if init_def_field is not None:
        def_field = np.asarray(init_def_field, dtype=float).copy()
    else:
        def_field = np.zeros((N, 3), dtype=float)

    # Precompute adjacency statistics (fixed throughout)
    neigh_count = np.asarray(adjacency.sum(axis=1), dtype=float).ravel()  # (N,)

    ref_tree = KDTree(ref_pts)

    for it in range(max_iter):
        transformed = mov_pts + def_field  # (N, 3)

        # ------------------------------------------------------------
        # E-step: build COO arrays in a single chunked pass, then
        # compute doubly-stochastic barycentres without a N×M matrix.
        # ------------------------------------------------------------
        rows_parts: list[np.ndarray] = []
        cols_parts: list[np.ndarray] = []
        wvals_parts: list[np.ndarray] = []

        for start in range(0, N, e_chunk):
            end = min(start + e_chunk, N)
            nbrs_chunk = ref_tree.query_ball_point(
                transformed[start:end], dist_cutoff,
                return_sorted=False, workers=-1,
            )
            for local_i, nbrs in enumerate(nbrs_chunk):
                if not nbrs:
                    continue
                i = start + local_i
                nbrs_arr = np.asarray(nbrs, dtype=np.int32)
                diffs    = ref_pts[nbrs_arr] - transformed[i]
                dists    = np.linalg.norm(diffs, axis=1)
                valid    = (ref_normals[nbrs_arr] @ mov_normals[i]) >= 0.0
                nbrs_v   = nbrs_arr[valid]
                if len(nbrs_v) == 0:
                    continue
                wv = np.exp(-dists[valid] / sigma)
                rows_parts.append(np.full(len(nbrs_v), i, dtype=np.int32))
                cols_parts.append(nbrs_v)
                wvals_parts.append(wv)

        if not rows_parts:
            if verbose:
                print(f"  iter {it:3d}: no correspondences — stopping early.")
            break

        rows  = np.concatenate(rows_parts)
        cols  = np.concatenate(cols_parts)
        wvals = np.concatenate(wvals_parts)
        del rows_parts, cols_parts, wvals_parts

        # Row / column sums
        row_sums = np.bincount(rows, weights=wvals, minlength=N)
        col_sums = np.bincount(cols, weights=wvals, minlength=M)

        row_inv = np.zeros(N)
        nz_r = row_sums > 0.0
        row_inv[nz_r] = 1.0 / row_sums[nz_r]

        col_inv = np.zeros(M)
        nz_c = col_sums > 0.0
        col_inv[nz_c] = 1.0 / col_sums[nz_c]

        # Doubly-stochastic weights: ṽ_ij = w_ij·(1/sC_j + 1/sR_i)
        v_tilde = wvals * (col_inv[cols] + row_inv[rows])
        del wvals

        # Weighted barycentre per moving vertex
        weight_out = np.bincount(rows, weights=v_tilde, minlength=N)
        corresBary = np.empty((N, 3), dtype=float)
        for k in range(3):
            corresBary[:, k] = np.bincount(
                rows, weights=v_tilde * ref_pts[cols, k], minlength=N
            )
        del v_tilde, rows, cols

        inlier_mask = weight_out > 0.0
        corresBary[inlier_mask] /= weight_out[inlier_mask, np.newaxis]

        # ------------------------------------------------------------
        # M-step: preconditioned conjugate gradient
        # Minimises:  Σ_i W_i ‖x_i + d_i − ȳ_i‖²
        #           + β Σ_{(i,j)∈edges} ‖d_i − d_j‖²
        # Equivalent to solving the symmetric PSD system:
        #   M · d[:,k] = b[:,k]   for k ∈ {0, 1, 2}
        # where
        #   M   = diag(W_i + β·|N_i|)  −  β · adjacency
        #   b_i = W_i · (ȳ_i − x_i)_k
        # Jacobi preconditioner M_inv = diag(1 / diag(M)).
        # Each coordinate is solved independently; the previous d warm-starts.
        # ------------------------------------------------------------
        target_offset = corresBary - mov_pts  # (N, 3)

        diag_vals = weight_out + beta * neigh_count          # (N,)
        M_mat  = sp_diags(diag_vals) - beta * adjacency     # (N, N) PSD
        M_prec = sp_diags(1.0 / np.maximum(diag_vals, 1e-10))  # Jacobi precond
        rhs    = weight_out[:, np.newaxis] * target_offset  # (N, 3)

        for k in range(3):
            def_field[:, k], _ = sp_cg(
                M_mat, rhs[:, k],
                x0=def_field[:, k],
                M=M_prec,
                rtol=1e-5,
                maxiter=icm_iter,
            )

        # ------------------------------------------------------------
        # Annealing: halve sigma every period_sigma iterations
        # ------------------------------------------------------------
        if (it + 1) % period_sigma == 0:
            sigma = max(sigma / 2.0, sigma_min)

        if verbose:
            n_inliers = int(inlier_mask.sum())
            print(
                f"  iter {it + 1:3d}/{max_iter}"
                f"  σ={sigma:.3f}"
                f"  inliers={n_inliers}/{N}"
            )

    return def_field


# ---------------------------------------------------------------------------
# Multi-resolution helpers
# ---------------------------------------------------------------------------

def _interpolate_field(
    field_coarse: np.ndarray,
    pts_coarse: np.ndarray,
    pts_fine: np.ndarray,
    k: int = 4,
) -> np.ndarray:
    """Inverse-distance weighted interpolation of a deformation field.

    For each vertex in *pts_fine*, locates the *k* nearest vertices in
    *pts_coarse* and computes a weighted average of their deformation
    vectors.  Weights are proportional to 1/distance, giving exact
    transfer when a fine vertex coincides with a coarse vertex.

    Parameters
    ----------
    field_coarse : ndarray (N_c, 3)
    pts_coarse   : ndarray (N_c, 3)
    pts_fine     : ndarray (N_f, 3)
    k            : int — number of neighbours (4 is typically sufficient)

    Returns
    -------
    ndarray (N_f, 3)
    """
    dists, idxs = KDTree(pts_coarse).query(pts_fine, k=k, workers=-1)
    w = 1.0 / np.maximum(dists, 1e-10)   # (N_f, k)
    w /= w.sum(axis=1, keepdims=True)
    # einsum: for each fine vertex sum  w[i,k] * field_coarse[idxs[i,k]]
    return np.einsum("nk,nkd->nd", w, field_coarse[idxs])


def _build_level(
    pts: np.ndarray,
    faces: list,
    target_n: int,
) -> tuple[np.ndarray, np.ndarray, list, csr_matrix]:
    """Decimate *pts/faces* to *target_n* vertices and build adjacency.

    Returns
    -------
    (pts_l, normals_l, faces_l, adjacency_l)
    """
    from pyclarcs.mesh import decimate_surface, compute_vertex_normals, adjacency_csr
    d_pts, d_faces = decimate_surface(pts, faces, target_n)
    d_normals = compute_vertex_normals(d_pts, d_faces)
    d_adj = adjacency_csr(d_faces, len(d_pts))
    return d_pts, d_normals, d_faces, d_adj


# ---------------------------------------------------------------------------
# Multi-resolution registration
# ---------------------------------------------------------------------------

def nonrigid_icp_multires(
    mov_pts: np.ndarray,
    mov_normals: np.ndarray,
    ref_pts: np.ndarray,
    ref_normals: np.ndarray,
    mov_polygons: list,
    *,
    n_levels: int = 3,
    target_n_coarsest: int = 2000,
    sigma: float | None = None,
    beta: float = 100.0,
    dist_cutoff: float | None = None,
    max_iter: int = 80,
    icm_iter: int = 50,
    period_sigma: int | None = None,
    sigma_min: float = 0.1,
    e_chunk: int = 2000,
    verbose: bool = True,
) -> np.ndarray:
    """Multi-resolution non-rigid EM-ICP surface registration.

    Builds a hierarchy of *n_levels* progressively decimated copies of the
    moving surface (always decimated from the original, not recursively).
    Registration runs from the coarsest level to the finest:

      coarsest (target_n_coarsest pts)
          → run nonrigid_icp  (max_iter iterations)
          → interpolate deformation field to next finer mesh (IDW, k=4)
      …
      finest (original mesh, len(mov_pts) pts)
          → run nonrigid_icp  (max_iter // 2 iterations, warm-started)

    At each level the KDTree is always queried against the **full-resolution
    reference**, so the hierarchy only affects the moving surface.

    Sigma, dist_cutoff and period_sigma are re-estimated at each level from
    the current residual (transformed minus reference) unless explicit values
    are provided.  Because the residual decreases as the hierarchy progresses,
    the kernel automatically narrows at finer scales.

    Parameters
    ----------
    mov_pts : ndarray (N, 3)
    mov_normals : ndarray (N, 3)
    ref_pts : ndarray (M, 3)
    ref_normals : ndarray (M, 3)
    mov_polygons : list of face index lists
        Polygon connectivity of the moving surface (needed for decimation
        and adjacency construction at each level).
    n_levels : int
        Number of resolution levels including the finest.  With
        ``n_levels=1`` the method is identical to ``nonrigid_icp``.
    target_n_coarsest : int
        Target vertex count at the coarsest level.  Intermediate levels
        are placed geometrically between this value and ``len(mov_pts)``.
    sigma, dist_cutoff, period_sigma : float or None
        Override the auto-estimated values at every level.
    beta : float
        Regularisation weight (same at all levels).
    max_iter : int
        Outer iterations at the **finest** level.  Each coarser level
        uses ``max_iter`` iterations as well (coarse levels are cheap).
    icm_iter : int
        Maximum CG iterations per outer iteration (same at all levels).
    sigma_min : float
        Annealing floor (same at all levels).
    e_chunk : int
        KDTree batch size (same at all levels).
    verbose : bool

    Returns
    -------
    def_field : ndarray (N, 3)
        Per-vertex deformation field at the finest (original) resolution.
        The warped surface is ``mov_pts + def_field``.
    """
    from pyclarcs.mesh import adjacency_csr

    mov_pts     = np.asarray(mov_pts,     dtype=float)
    mov_normals = np.asarray(mov_normals, dtype=float)
    ref_pts     = np.asarray(ref_pts,     dtype=float)
    ref_normals = np.asarray(ref_normals, dtype=float)

    N = len(mov_pts)

    # ------------------------------------------------------------------
    # Build hierarchy: level 0 = finest (original), level L-1 = coarsest
    # ------------------------------------------------------------------
    adj_finest = adjacency_csr(mov_polygons, N)
    # Each entry: (pts, normals, faces, adjacency)
    hierarchy = [(mov_pts, mov_normals, mov_polygons, adj_finest)]

    for lev in range(1, n_levels):
        # Geometrically-spaced target size in log space
        t = lev / (n_levels - 1) if n_levels > 1 else 1.0
        target_n = max(
            target_n_coarsest,
            int(N * (target_n_coarsest / N) ** t),
        )
        if target_n >= len(hierarchy[0][0]) * 0.85:
            if verbose:
                print(
                    f"  [multires] level {lev}: target {target_n} too close "
                    f"to finest ({N}), stopping hierarchy here."
                )
            break
        if verbose:
            print(f"  [multires] building level {lev}: {N} → ~{target_n} vertices…")
        hierarchy.append(_build_level(mov_pts, mov_polygons, target_n))

    n_actual = len(hierarchy)

    # ------------------------------------------------------------------
    # Register coarsest → finest
    # ------------------------------------------------------------------
    def_field_prev: np.ndarray | None = None   # result from coarser level
    pts_prev: np.ndarray | None       = None

    for idx in range(n_actual - 1, -1, -1):
        pts_l, normals_l, _, adj_l = hierarchy[idx]
        N_l = len(pts_l)
        is_finest = (idx == 0)

        # Warm-start from interpolated coarser field
        if def_field_prev is None:
            init_l = None
        else:
            init_l = _interpolate_field(def_field_prev, pts_prev, pts_l)

        # Fewer iterations at the finest level: the coarse levels already
        # captured the large-scale deformation.
        max_iter_l = (max_iter // 2) if is_finest and n_actual > 1 else max_iter

        if verbose:
            label = "finest" if is_finest else f"level {idx}"
            print(
                f"\n  [multires] {label}  {N_l} vertices"
                f"  {max_iter_l} outer iterations"
            )

        # Auto-estimate params from current residual
        transformed_l = pts_l if init_l is None else pts_l + init_l
        sigma_l       = sigma
        cutoff_l      = dist_cutoff
        period_l      = period_sigma

        if sigma_l is None or cutoff_l is None or period_l is None:
            auto = estimate_registration_params(
                transformed_l, ref_pts,
                max_iter=max_iter_l, sigma_min=sigma_min,
            )
            if sigma_l  is None: sigma_l  = auto["sigma"]
            if cutoff_l is None: cutoff_l = auto["dist_cutoff"]
            if period_l is None: period_l = auto["period_sigma"]
            if verbose:
                print(
                    f"    auto params:  σ={sigma_l}  r={cutoff_l}"
                    f"  period_σ={period_l}"
                )

        def_field_l = nonrigid_icp(
            pts_l, normals_l,
            ref_pts, ref_normals,
            adj_l,
            init_def_field=init_l,
            sigma=sigma_l,
            beta=beta,
            dist_cutoff=cutoff_l,
            max_iter=max_iter_l,
            icm_iter=icm_iter,
            period_sigma=period_l,
            sigma_min=sigma_min,
            e_chunk=e_chunk,
            verbose=verbose,
        )

        def_field_prev = def_field_l
        pts_prev       = pts_l

    return def_field_prev  # finest-level result


def apply_deformation(
    points: np.ndarray,
    def_field: np.ndarray,
) -> np.ndarray:
    """Apply a deformation field to a point cloud.

    Parameters
    ----------
    points : ndarray (N, 3)
    def_field : ndarray (N, 3)

    Returns
    -------
    ndarray (N, 3) — warped coordinates  ``points + def_field``.
    """
    return np.asarray(points, dtype=float) + np.asarray(def_field, dtype=float)
