"""
Fine symmetry plane optimisation via EM-ICP (Numba-accelerated).

See the original module docstring for the full mathematical background.
This version delegates all hot loops to JIT-compiled kernels in
``_numba_kernels.py``:

  - ``compute_weights``            — E-step Gaussian weights  (parallel)
  - ``normalise_standard``         — column-wise normalisation (parallel)
  - ``normalise_doubly_stochastic``— doubly-stochastic norm.  (parallel)
  - ``collect_mstep_pairs``        — M-step pair collection
  - ``uniform_subsample_numba``    — greedy uniform subsampling

The scipy KD-tree calls (``query_ball_point``) stay in Python; their
results are converted to CSR format before entering Numba code.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from pyclarcs.symmetry import SymmetryPlane
from pyclarcs._numba_kernels import (
    nbrs_to_csr,
    compute_weights,
    normalise_standard,
    accumulate_col_sums,
    normalise_doubly_stochastic,
    collect_mstep_pairs,
    uniform_subsample_numba,
    _warmup,
)

# Trigger JIT compilation at import time (uses cached bytecode after first run).
_warmup()


# ---------------------------------------------------------------------------
# Uniform subsampling
# ---------------------------------------------------------------------------

def _uniform_subsample(
    points: np.ndarray,
    radius: float,
    tree: KDTree,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge points within *radius* into barycentres (uniform subsampling).

    Uses a pre-computed full neighbourhood query converted to CSR format,
    then delegates the greedy merging loop to the JIT kernel
    ``uniform_subsample_numba``.
    """
    neighbour_ids = tree.query_ball_point(points, radius, workers=-1)
    flat_nbrs, offsets = nbrs_to_csr(neighbour_ids)
    points_c = np.ascontiguousarray(points, dtype=np.float64)
    return uniform_subsample_numba(points_c, flat_nbrs, offsets)


# ---------------------------------------------------------------------------
# E-step: compute soft-correspondence weights
# ---------------------------------------------------------------------------

def _e_step(
    work_pts: np.ndarray,
    model_tree: KDTree,
    model_pts: np.ndarray,
    plane: SymmetryPlane,
    sigma_sq: float,
    nu_max: float,
    doubly_stochastic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the soft-correspondence weights (E-step).

    Returns CSR arrays for direct use in the M-step Numba kernel.

    Returns
    -------
    flat_nbrs    : int64  (K,)   — CSR neighbour indices
    offsets      : int64  (M+1,) — CSR row offsets
    flat_weights : float64 (K,)  — normalised weights
    row_sums     : float64 (M,)  — unnormalised row sums
    """
    radius = np.sqrt(2.0 * nu_max * sigma_sq)
    reflected = np.ascontiguousarray(plane.apply(work_pts), dtype=np.float64)
    neighbour_ids = model_tree.query_ball_point(reflected, radius, workers=-1)
    flat_nbrs, offsets = nbrs_to_csr(neighbour_ids)
    model_c = np.ascontiguousarray(model_pts, dtype=np.float64)

    flat_raw, row_sums = compute_weights(reflected, model_c, flat_nbrs, offsets, sigma_sq)

    if not doubly_stochastic:
        flat_weights = normalise_standard(flat_raw, row_sums, offsets)
    else:
        col_sums = accumulate_col_sums(flat_nbrs, flat_raw, len(model_pts))
        flat_weights = normalise_doubly_stochastic(
            flat_raw, row_sums, col_sums, flat_nbrs, offsets
        )

    return flat_nbrs, offsets, flat_weights, row_sums


# ---------------------------------------------------------------------------
# M-step: fit the symmetry plane to the weighted correspondences
# ---------------------------------------------------------------------------

def _m_step(
    work_pts: np.ndarray,
    model_pts: np.ndarray,
    work_counts: np.ndarray,
    flat_nbrs: np.ndarray,
    offsets: np.ndarray,
    flat_weights: np.ndarray,
    plane: SymmetryPlane,
) -> None:
    """Fit the symmetry plane to the current weighted correspondences (M-step).

    Delegates pair collection to the JIT kernel ``collect_mstep_pairs``,
    then calls ``SymmetryPlane.fit()``.
    """
    if len(flat_nbrs) == 0:
        return

    work_c  = np.ascontiguousarray(work_pts,    dtype=np.float64)
    model_c = np.ascontiguousarray(model_pts,   dtype=np.float64)
    cnts_c  = np.ascontiguousarray(work_counts, dtype=np.float64)

    sources, targets, ws = collect_mstep_pairs(
        work_c, model_c, cnts_c, flat_nbrs, offsets, flat_weights
    )

    mask = ws > 0.0
    if not mask.any():
        return
    plane.fit(sources[mask], targets[mask], ws[mask])


# ---------------------------------------------------------------------------
# EM_ICPSym — fine optimisation with annealing
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
    """Fine symmetry optimisation with EM-ICP and simulated annealing."""
    plane = initial_plane.copy()
    tree = KDTree(points)

    sigma = float(sigma_init)
    K = 1.0

    work_pts, work_counts = _uniform_subsample(points, sigma, tree)
    need_resample = False

    for it in range(max_iter):
        if need_resample:
            work_pts, work_counts = _uniform_subsample(points, sigma, tree)
            need_resample = False

        sigma_sq = sigma * sigma
        d_prev = plane.d
        n_prev = plane.n.copy()

        flat_nbrs, offsets, flat_weights, _ = _e_step(
            work_pts, tree, points, plane, sigma_sq, nu_max, doubly_stochastic=False
        )
        _m_step(work_pts, points, work_counts, flat_nbrs, offsets, flat_weights, plane)

        if verbose:
            print(
                f"  [EM_ICP] iter {it:4d}  σ={sigma:.4f}"
                f"  d={plane.d:.4f}  n={plane.n}"
            )

        converged_here = (
            abs(plane.d - d_prev) < convergence_tol
            and abs(plane.n[0] - n_prev[0]) < convergence_tol
            and abs(plane.n[1] - n_prev[1]) < convergence_tol
            and abs(plane.n[2] - n_prev[2]) < convergence_tol
        )
        if converged_here:
            if sigma > sigma_final:
                sigma = max(sigma / K, sigma_final)
                need_resample = True
                if verbose:
                    print(f"    σ decreased → {sigma:.4f}")
            else:
                if verbose:
                    print(f"  [EM_ICP] converged at iter {it}")
                break

        if it == 0:
            K = 1.03

    # Final refinement at σ_final (full cloud)
    sigma_sq_final = sigma_final * sigma_final
    full_counts = np.ones(len(points), dtype=np.float64)

    for _ in range(max_iter_final):
        flat_nbrs, offsets, flat_weights, _ = _e_step(
            points, tree, points, plane, sigma_sq_final, nu_max, doubly_stochastic=False
        )
        _m_step(points, points, full_counts, flat_nbrs, offsets, flat_weights, plane)

    if verbose:
        print(f"  [EM_ICP] final plane: {plane}")

    return plane


# ---------------------------------------------------------------------------
# EM_ICPSymCorresSym — doubly-stochastic final refinement
# ---------------------------------------------------------------------------

def em_icp_sym_corres(
    points: np.ndarray,
    initial_plane: SymmetryPlane,
    sigma: float = 0.25,
    nu_max: float = 1.5,
    max_iter: int = 100,
    convergence_tol: float = 1e-5,
    verbose: bool = False,
) -> SymmetryPlane:
    """Final symmetry refinement with doubly-stochastic EM-ICP."""
    plane = initial_plane.copy()
    tree = KDTree(points)
    sigma_sq = sigma * sigma
    counts = np.ones(len(points), dtype=np.float64)

    for it in range(max_iter):
        d_prev = plane.d
        n_prev = plane.n.copy()

        flat_nbrs, offsets, flat_weights, _ = _e_step(
            points, tree, points, plane, sigma_sq, nu_max, doubly_stochastic=True
        )
        _m_step(points, points, counts, flat_nbrs, offsets, flat_weights, plane)

        if verbose:
            print(f"  [EM_corres] iter {it:4d}  d={plane.d:.4f}  n={plane.n}")

        if (
            abs(plane.d - d_prev) < convergence_tol
            and np.max(np.abs(plane.n - n_prev)) < convergence_tol
        ):
            if verbose:
                print(f"  [EM_corres] converged at iter {it}")
            break

    if verbose:
        print(f"  [EM_corres] final plane: {plane}")

    return plane
