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

M-step — Jacobi ICM (icm_iter inner iterations)
  For each i:
      d_i ← ( W_i (ȳ_i − x_i)  +  β Σ_{j ∈ N_i} d_j )
              / ( β |N_i| + W_i )

  Points with no correspondences (W_i = 0) are interpolated from
  their mesh neighbours:   d_i ← (Σ_{j ∈ N_i} d_j) / |N_i|.

Annealing
  σ ← max(σ / 2,  σ_min)   every `period_sigma` outer iterations.

DEFAULT PARAMETERS (matching the original C++ NonLinearRegistration)
  sigma        = 3.0    initial bandwidth (same units as coordinates)
  beta         = 100.0  regularisation weight
  dist_cutoff  = 15.0   search radius
  max_iter     = 80
  icm_iter     = 120    inner Jacobi steps per outer iteration
  period_sigma = 40     sigma halved every 40 outer iterations
  sigma_min    = 0.1
  e_chunk      = 2000   vertices per KDTree query batch
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix


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
    sigma: float = 3.0,
    beta: float = 100.0,
    dist_cutoff: float = 15.0,
    max_iter: int = 80,
    icm_iter: int = 120,
    period_sigma: int = 40,
    sigma_min: float = 0.1,
    e_chunk: int = 2000,
    verbose: bool = True,
) -> np.ndarray:
    """Register a moving surface onto a reference using non-rigid EM-ICP.

    The algorithm iterates between:
      - computing doubly-stochastic fuzzy correspondences (E-step), and
      - solving for the deformation field with Laplacian regularisation
        via Jacobi ICM (M-step).

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
        Number of Jacobi ICM steps per outer iteration.
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
        # M-step: Jacobi ICM
        # Minimises:  Σ_i W_i ‖x_i + d_i − ȳ_i‖²
        #           + β Σ_{(i,j)∈edges} ‖d_i − d_j‖²
        # Closed-form per-node update (Jacobi):
        #   d_i ← (W_i (ȳ_i − x_i) + β Σ_{j∈N_i} d_j) / (β|N_i| + W_i)
        # ------------------------------------------------------------
        target_offset = corresBary - mov_pts  # (N, 3)

        for _ in range(icm_iter):
            neigh_sum = adjacency @ def_field                   # (N, 3)
            denom = beta * neigh_count + weight_out             # (N,)
            valid_denom = denom > 0.0

            new_field = def_field.copy()
            new_field[valid_denom] = (
                weight_out[valid_denom, np.newaxis] * target_offset[valid_denom]
                + beta * neigh_sum[valid_denom]
            ) / denom[valid_denom, np.newaxis]
            def_field = new_field

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
