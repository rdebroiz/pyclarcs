"""
Fine symmetry plane optimisation via EM-ICP.

Two variants, matching the C++ classes:

- EM_ICPSym          (EM-ICPSym.hh)          : soft correspondences with annealing
- EM_ICPSymCorresSym (EM-ICPSymCorresSym.hh)  : same + doubly-stochastic normalisation

Both alternate between:
  E-step: build a soft correspondence matrix using Gaussian weights and a
          KD-tree search within a radius proportional to sigma.
  M-step: fit the symmetry plane to the weighted correspondences.

The annealing schedule decreases sigma (from sigma_init down to sigma_final)
whenever the plane estimate stops changing.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from pyclarcs._symmetry import SymmetryPlane


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _uniform_subsample(points: np.ndarray, radius: float, tree: KDTree) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-like subsampling: merge points within *radius* into barycenters.

    Matches EM_ICPSym::unif_subsampling_fast().

    Returns
    -------
    barycenters : ndarray (M, 3)
    counts      : ndarray (M,)  – number of original points merged into each
    """
    n = len(points)
    merged = np.zeros(n, dtype=bool)
    bary_list: list[np.ndarray] = []
    count_list: list[int] = []

    for i in range(n):
        if merged[i]:
            continue
        # Find all points within radius of points[i]
        neighbours = tree.query_ball_point(points[i], radius)
        # Keep only the ones not yet merged
        active = [j for j in neighbours if not merged[j]]
        if not active:
            active = [i]
        for j in active:
            merged[j] = True
        bary_list.append(points[active].mean(axis=0))
        count_list.append(len(active))

    return np.array(bary_list), np.array(count_list, dtype=float)


def _e_step(
    work_pts: np.ndarray,
    model_tree: KDTree,
    model_pts: np.ndarray,
    plane: SymmetryPlane,
    sigma_sq: float,
    nu_max: float,
    doubly_stochastic: bool,
) -> tuple[list[list[int]], list[list[float]]]:
    """Compute soft-correspondence weights (E-step).

    For each working point p:
      1. Reflect it:  p' = plane.apply(p)
      2. Find all model points within radius nu_max * sigma_sq of p'
      3. Assign weight  exp(-||p' - q||² / (2 * sigma_sq))

    If *doubly_stochastic* is True, normalise both by row and by column
    (EM_ICPSymCorresSym variant).

    Returns
    -------
    indices : list of length len(work_pts), each element a list of column indices
    weights : corresponding weight lists (already normalised)
    """
    radius = nu_max * sigma_sq
    n_work = len(work_pts)

    reflected = plane.apply(work_pts)  # (M, 3)
    neighbour_ids = model_tree.query_ball_point(reflected, radius)  # list of lists

    # Raw weights
    raw: list[list[float]] = []
    row_sums = np.zeros(n_work)
    for i, (r_pt, nbrs) in enumerate(zip(reflected, neighbour_ids)):
        if not nbrs:
            raw.append([])
            continue
        nbr_arr = model_pts[nbrs]
        dist_sq = np.sum((r_pt - nbr_arr) ** 2, axis=1)
        w = np.exp(-dist_sq / (2.0 * sigma_sq))
        raw.append(w.tolist())
        row_sums[i] = w.sum()

    if not doubly_stochastic:
        # Row-normalise only (EM_ICPSym variant)
        weights: list[list[float]] = []
        for i, (nbrs, w_row) in enumerate(zip(neighbour_ids, raw)):
            if not nbrs or row_sums[i] == 0:
                weights.append([])
            else:
                weights.append([w / row_sums[i] for w in w_row])
        return neighbour_ids, weights

    # Doubly-stochastic normalisation (EM_ICPSymCorresSym variant)
    # First pass: column sums
    col_sums: dict[int, float] = {}
    for i, (nbrs, w_row) in enumerate(zip(neighbour_ids, raw)):
        for j_loc, j_glob in enumerate(nbrs):
            col_sums[j_glob] = col_sums.get(j_glob, 0.0) + w_row[j_loc]

    # Normalise:  w_ij = 2 * raw_ij / (row_sum_i * col_sum_j)
    weights_ds: list[list[float]] = []
    for i, (nbrs, w_row) in enumerate(zip(neighbour_ids, raw)):
        if not nbrs or row_sums[i] == 0:
            weights_ds.append([])
            continue
        w_norm = [
            2.0 * w_row[k] / (row_sums[i] * col_sums[nbrs[k]])
            for k in range(len(nbrs))
            if col_sums.get(nbrs[k], 0) > 0
        ]
        weights_ds.append(w_norm)

    return neighbour_ids, weights_ds


def _m_step(
    work_pts: np.ndarray,
    model_pts: np.ndarray,
    work_counts: np.ndarray,
    neighbour_ids: list[list[int]],
    weights: list[list[float]],
    plane: SymmetryPlane,
) -> None:
    """Fit the symmetry plane to the weighted correspondences (M-step).

    Accumulates (source, target, weight) triplets then calls plane.fit().
    """
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ws: list[float] = []

    for i, (nbrs, w_row) in enumerate(zip(neighbour_ids, weights)):
        if not nbrs:
            continue
        count = float(work_counts[i])
        for k, j in enumerate(nbrs):
            sources.append(work_pts[i])
            targets.append(model_pts[j])
            ws.append(w_row[k] * count)

    if not sources:
        return

    plane.fit(
        np.array(sources),
        np.array(targets),
        np.array(ws),
    )


# ---------------------------------------------------------------------------
# EM_ICPSym
# ---------------------------------------------------------------------------

def em_icp_sym(
    points: np.ndarray,
    initial_plane: SymmetryPlane,
    sigma_init: float = 5.0,
    sigma_final: float = 0.5,
    nu_max: float = 1.5,
    max_iter: int = 400,
    max_iter_final: int = 50,
    convergence_tol: float = 0.01,
    verbose: bool = False,
) -> SymmetryPlane:
    """Fine symmetry optimisation with EM-ICP and simulated annealing.

    Mirrors EM_ICPSym::optimize() from Symmetry/EM-ICPSym.hh.

    Parameters
    ----------
    points : ndarray (N, 3)
    initial_plane : SymmetryPlane
    sigma_init : float – initial bandwidth for Gaussian correspondences
    sigma_final : float – final (minimum) bandwidth
    nu_max : float – neighbourhood radius = nu_max * sigma²
    max_iter : int – annealing phase iterations
    max_iter_final : int – additional iterations at sigma_final
    convergence_tol : float – |Δd| and |Δn| threshold to trigger annealing
    verbose : bool

    Returns
    -------
    SymmetryPlane
    """
    plane = initial_plane.copy()
    tree = KDTree(points)
    sigma = sigma_init
    K = 1.0  # annealing rate (increases after first step)

    # Pre-compute sub-sampled working set for the current sigma
    work_pts, work_counts = _uniform_subsample(points, sigma, tree)

    for it in range(max_iter):
        sigma_sq = sigma * sigma
        d_prev = plane.d
        n_prev = plane.n.copy()

        nbrs, wts = _e_step(work_pts, tree, points, plane, sigma_sq, nu_max, False)
        _m_step(work_pts, points, work_counts, nbrs, wts, plane)

        if verbose:
            print(f"  [EM_ICP] iter {it:4d}  sigma={sigma:.4f}  d={plane.d:.4f}")

        # Annealing: decrease sigma when the plane has converged at this level
        if sigma > sigma_final:
            converged_at_level = (
                abs(plane.d - d_prev) < convergence_tol
                and np.linalg.norm(plane.n - n_prev) < convergence_tol
            )
            if converged_at_level:
                sigma = max(sigma / (K if K > 1 else 2.0), sigma_final)
                work_pts, work_counts = _uniform_subsample(points, sigma, tree)
                if verbose:
                    print(f"    sigma → {sigma:.4f}")

        if it > 0:
            K = 1.03  # matches C++ K=1.03 after first iteration

    # Final refinement at sigma_final with all points
    sigma_sq = sigma_final * sigma_final
    work_pts_full = points
    work_counts_full = np.ones(len(points))
    for _ in range(max_iter_final):
        nbrs, wts = _e_step(
            work_pts_full, tree, points, plane, sigma_sq, nu_max, False
        )
        _m_step(work_pts_full, points, work_counts_full, nbrs, wts, plane)

    if verbose:
        print(f"  [EM_ICP] final: d={plane.d:.4f}  n={plane.n}")

    return plane


# ---------------------------------------------------------------------------
# EM_ICPSymCorresSym
# ---------------------------------------------------------------------------

def em_icp_sym_corres(
    points: np.ndarray,
    initial_plane: SymmetryPlane,
    sigma: float = 0.25,
    nu_max: float = 1.5,
    max_iter: int = 100,
    verbose: bool = False,
) -> SymmetryPlane:
    """Fine symmetry optimisation with doubly-stochastic EM-ICP.

    Mirrors EM_ICPSymCorresSym::optimize() from
    Symmetry/EM-ICPSymCorresSym.hh.

    This variant runs entirely at the given fixed sigma (no annealing) and
    normalises the correspondence matrix to be doubly stochastic (both row
    and column sums equal 1 up to a factor).

    Parameters
    ----------
    points : ndarray (N, 3)
    initial_plane : SymmetryPlane
    sigma : float – fixed bandwidth (default 0.25 as in C++)
    nu_max : float
    max_iter : int
    verbose : bool

    Returns
    -------
    SymmetryPlane
    """
    plane = initial_plane.copy()
    tree = KDTree(points)
    sigma_sq = sigma * sigma
    counts = np.ones(len(points))

    for it in range(max_iter):
        d_prev = plane.d
        n_prev = plane.n.copy()

        nbrs, wts = _e_step(points, tree, points, plane, sigma_sq, nu_max, True)
        _m_step(points, points, counts, nbrs, wts, plane)

        if verbose:
            print(f"  [EM_corres] iter {it:4d}  d={plane.d:.4f}")

    if verbose:
        print(f"  [EM_corres] final: d={plane.d:.4f}  n={plane.n}")

    return plane
