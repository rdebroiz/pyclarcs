"""
JIT-compiled kernels for the EM-ICP symmetry pipeline (Numba backend).

All functions are compiled on first call and cached to disk (``cache=True``).
Subsequent calls pay zero compilation overhead.

The hot loops in the E-step and the uniform subsampling are the two main
bottlenecks of the pipeline.  Both are pure numerical loops that translate
directly to efficient native code under Numba's ``@njit``.

Sparse neighbourhood format (CSR)
==================================
scipy's ``query_ball_point`` returns a Python list of variable-length lists.
Numba cannot operate on such structures.  We therefore convert the result to
a *Compressed Sparse Row* (CSR) representation before entering any JIT
function:

    flat_nbrs : int64[:]   — concatenated neighbour indices
    offsets   : int64[:]   — offsets[j] .. offsets[j+1] are the neighbours
                             of working point j  (len = n_work + 1)

This conversion is done once per E-step call in ``nbrs_to_csr()``.

Parallelism
===========
``@njit(parallel=True)`` with ``numba.prange`` gives OpenMP-style
parallelism that bypasses the GIL entirely.  On a 20-core machine the
E-step weight computation should scale linearly up to the point where
memory bandwidth saturates.
"""

from __future__ import annotations

import numpy as np
import numba


# ---------------------------------------------------------------------------
# CSR conversion  (Python-level helper, called once per E-step)
# ---------------------------------------------------------------------------

def nbrs_to_csr(
    neighbour_ids: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list-of-lists of neighbour indices to CSR format.

    Parameters
    ----------
    neighbour_ids : list of M lists of int
        Output of ``KDTree.query_ball_point``.

    Returns
    -------
    flat_nbrs : int64 array (total_neighbours,)
    offsets   : int64 array (M + 1,)
        offsets[j] .. offsets[j+1] → neighbours of point j
    """
    counts = np.array([len(n) for n in neighbour_ids], dtype=np.int64)
    offsets = np.zeros(len(neighbour_ids) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    total = int(offsets[-1])
    flat_nbrs = np.empty(total, dtype=np.int64)
    pos = 0
    for nbrs in neighbour_ids:
        for idx in nbrs:
            flat_nbrs[pos] = idx
            pos += 1
    return flat_nbrs, offsets


# ---------------------------------------------------------------------------
# E-step kernel — Gaussian weight computation
# ---------------------------------------------------------------------------

@numba.njit(parallel=True, cache=True)
def compute_weights(
    reflected: np.ndarray,   # (M, 3) float64
    model_pts: np.ndarray,   # (N, 3) float64
    flat_nbrs: np.ndarray,   # (K,)   int64
    offsets: np.ndarray,     # (M+1,) int64
    sigma_sq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute unnormalised Gaussian weights for all (source, neighbour) pairs.

    For each working point j and each of its neighbours i (in CSR format):
        raw[k] = exp( -||reflected[j] - model_pts[i]||² / (2σ²) )

    The outer loop over working points runs in parallel (``prange``).

    Returns
    -------
    flat_raw  : float64 (K,) — unnormalised weight for each CSR entry
    row_sums  : float64 (M,) — Σ_i raw[i,j] for each working point j
    """
    n_work = len(offsets) - 1
    flat_raw = np.zeros(len(flat_nbrs), dtype=np.float64)
    row_sums = np.zeros(n_work, dtype=np.float64)
    inv_2s2 = 1.0 / (2.0 * sigma_sq)

    for j in numba.prange(n_work):
        start = offsets[j]
        end = offsets[j + 1]
        if start == end:
            continue
        rx, ry, rz = reflected[j, 0], reflected[j, 1], reflected[j, 2]
        rs = 0.0
        for k in range(start, end):
            i = flat_nbrs[k]
            dx = rx - model_pts[i, 0]
            dy = ry - model_pts[i, 1]
            dz = rz - model_pts[i, 2]
            w = np.exp(-(dx*dx + dy*dy + dz*dz) * inv_2s2)
            flat_raw[k] = w
            rs += w
        row_sums[j] = rs

    return flat_raw, row_sums


# ---------------------------------------------------------------------------
# Standard (column-wise) normalisation
# ---------------------------------------------------------------------------

@numba.njit(parallel=True, cache=True)
def normalise_standard(
    flat_raw: np.ndarray,   # (K,)   float64
    row_sums: np.ndarray,   # (M,)   float64
    offsets: np.ndarray,    # (M+1,) int64
) -> np.ndarray:
    """Divide each raw weight by its column (working-point) sum.

    Returns
    -------
    flat_weights : float64 (K,)
    """
    n_work = len(offsets) - 1
    flat_weights = np.zeros(len(flat_raw), dtype=np.float64)
    for j in numba.prange(n_work):
        rs = row_sums[j]
        if rs == 0.0:
            continue
        for k in range(offsets[j], offsets[j + 1]):
            flat_weights[k] = flat_raw[k] / rs
    return flat_weights


# ---------------------------------------------------------------------------
# Doubly-stochastic normalisation
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def accumulate_col_sums(
    flat_nbrs: np.ndarray,   # (K,)   int64
    flat_raw: np.ndarray,    # (K,)   float64
    n_model: int,
) -> np.ndarray:
    """Accumulate col_sums[i] = Σ_j raw[i,j] for each model point i."""
    col_sums = np.zeros(n_model, dtype=np.float64)
    for k in range(len(flat_nbrs)):
        col_sums[flat_nbrs[k]] += flat_raw[k]
    return col_sums


@numba.njit(parallel=True, cache=True)
def normalise_doubly_stochastic(
    flat_raw: np.ndarray,    # (K,)   float64
    row_sums: np.ndarray,    # (M,)   float64
    col_sums: np.ndarray,    # (N,)   float64
    flat_nbrs: np.ndarray,   # (K,)   int64
    offsets: np.ndarray,     # (M+1,) int64
) -> np.ndarray:
    """Doubly-stochastic normalisation: A_{i,j} = 2 * raw / (row_sum * col_sum)."""
    n_work = len(offsets) - 1
    flat_weights = np.zeros(len(flat_raw), dtype=np.float64)
    for j in numba.prange(n_work):
        rs = row_sums[j]
        if rs == 0.0:
            continue
        for k in range(offsets[j], offsets[j + 1]):
            cs = col_sums[flat_nbrs[k]]
            if cs > 0.0:
                flat_weights[k] = 2.0 * flat_raw[k] / (rs * cs)
    return flat_weights


# ---------------------------------------------------------------------------
# M-step: collect (source, target, weight) triplets
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def collect_mstep_pairs(
    work_pts: np.ndarray,      # (M, 3) float64
    model_pts: np.ndarray,     # (N, 3) float64
    work_counts: np.ndarray,   # (M,)   float64
    flat_nbrs: np.ndarray,     # (K,)   int64
    offsets: np.ndarray,       # (M+1,) int64
    flat_weights: np.ndarray,  # (K,)   float64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten the sparse correspondence matrix into dense triplet arrays.

    Returns
    -------
    sources : float64 (K, 3)
    targets : float64 (K, 3)
    ws      : float64 (K,)
    """
    total = len(flat_nbrs)
    sources = np.empty((total, 3), dtype=np.float64)
    targets = np.empty((total, 3), dtype=np.float64)
    ws = np.empty(total, dtype=np.float64)

    n_work = len(offsets) - 1
    for j in range(n_work):
        count = work_counts[j]
        for k in range(offsets[j], offsets[j + 1]):
            i = flat_nbrs[k]
            sources[k, 0] = work_pts[j, 0]
            sources[k, 1] = work_pts[j, 1]
            sources[k, 2] = work_pts[j, 2]
            targets[k, 0] = model_pts[i, 0]
            targets[k, 1] = model_pts[i, 1]
            targets[k, 2] = model_pts[i, 2]
            ws[k] = flat_weights[k] * count

    return sources, targets, ws


# ---------------------------------------------------------------------------
# Uniform subsampling (greedy, sequential — but JIT gives ~30x over CPython)
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def uniform_subsample_numba(
    points: np.ndarray,     # (N, 3) float64
    flat_nbrs: np.ndarray,  # (K,)   int64  — pre-computed at radius σ
    offsets: np.ndarray,    # (N+1,) int64
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy uniform subsampling: merge neighbours into barycentres.

    Parameters
    ----------
    points     : (N, 3) full point cloud
    flat_nbrs  : CSR flat neighbour indices (from query_ball_point at radius σ)
    offsets    : CSR offsets

    Returns
    -------
    barycentres : (M, 3) — representative points
    counts      : (M,)   — number of original points merged into each
    """
    n = len(points)
    merged = np.zeros(n, dtype=numba.boolean)

    # Pre-allocate output at maximum size (M ≤ N)
    bary = np.empty((n, 3), dtype=np.float64)
    cnts = np.empty(n, dtype=np.float64)
    m = 0  # number of barycentres found so far

    for i in range(n):
        if merged[i]:
            continue

        # Collect unmerged neighbours of i (including i itself)
        sx, sy, sz = 0.0, 0.0, 0.0
        cnt = 0
        for k in range(offsets[i], offsets[i + 1]):
            j = flat_nbrs[k]
            if not merged[j]:
                merged[j] = True
                sx += points[j, 0]
                sy += points[j, 1]
                sz += points[j, 2]
                cnt += 1

        if cnt == 0:
            # i had no unmerged neighbours — treat it alone
            merged[i] = True
            sx, sy, sz = points[i, 0], points[i, 1], points[i, 2]
            cnt = 1

        bary[m, 0] = sx / cnt
        bary[m, 1] = sy / cnt
        bary[m, 2] = sz / cnt
        cnts[m] = float(cnt)
        m += 1

    return bary[:m], cnts[:m]


# ---------------------------------------------------------------------------
# Warm-up: trigger JIT compilation at import time on small dummy data
# so the first real call is not slow.
# ---------------------------------------------------------------------------

def _warmup() -> None:
    """Pre-compile all kernels with tiny dummy arrays."""
    dummy_pts = np.zeros((4, 3), dtype=np.float64)
    dummy_ref = np.zeros((2, 3), dtype=np.float64)
    flat = np.array([0, 1, 2, 3], dtype=np.int64)
    off  = np.array([0, 2, 4], dtype=np.int64)   # 2 working pts, 2 nbrs each

    raw, rs = compute_weights(dummy_ref, dummy_pts, flat, off, 1.0)
    fw = normalise_standard(raw, rs, off)
    cs = accumulate_col_sums(flat, raw, 4)
    normalise_doubly_stochastic(raw, rs, cs, flat, off)
    collect_mstep_pairs(dummy_pts[:2], dummy_pts, np.ones(2), flat, off, fw)

    # subsampling warm-up
    off2 = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    flat2 = np.array([0, 1, 2, 3], dtype=np.int64)
    uniform_subsample_numba(dummy_pts, flat2, off2)
