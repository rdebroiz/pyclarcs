"""
Fine symmetry plane optimisation via EM-ICP.

THEORETICAL BACKGROUND
=======================
The fine stage implements the EM-ICP framework described in:

  Combès B., Hennessy R., Waddington J., Roberts N., Prima S.
  "Automatic symmetry plane estimation of bilateral objects in point clouds."
  CVPR 2008.  [ref 7]

and placed in the broader context of:

  Abadie A., Combès B., Haegelen C., Prima S.
  "CLARCS, a C++ Library for Automated Registration and Comparison of Surfaces."
  MeshMed'2011.  [the CLARCS paper]

THE MAP / EM FRAMEWORK (Section 2.2 of the CLARCS paper)
==========================================================
The distance between a point set X¹ and a transformed version T(X²) is
defined as the MAP objective:

    δ²(X¹, X²) = min_{A, T} [
        Σ_{i,j}  A_{i,j} ||x_i − T(x_j)||²          (data attachment)
      + 2σ²  Σ_{i,j}  A_{i,j} log A_{i,j}            (barrier on A)
      + α L(T)                                          (regularisation of T)
    ]

subject to  ∀i  Σ_j A_{i,j} = 1  and  ∀i,j  A_{i,j} ≥ 0.

The soft assignment matrix A encodes point-to-point correspondences in a
*fuzzy* way.  With the probabilistic interpretation:
  - each x_i ∈ X¹ is an independent draw from a Gaussian Mixture Model (GMM)
    whose means are the points T(x_j) of T(X²) and whose covariance is σ²I;
  - A_{i,j} is the posterior probability that x_i came from the j-th component.

This MAP problem is solved iteratively by the EM algorithm:

  **E-step** (expectation) — update A given the current T̃:

      Ã_{i,j} = exp(−||x_i − T̃(x_j)||² / (2σ²))
               / Σ_k  exp(−||x_i − T̃(x_k)||² / (2σ²))

  **M-step** (maximisation) — update T given the current Ã:

      T̄ = argmin_T  Σ_{i,j}  Ã_{i,j} ||x_i − T(x_j)||²  +  α L(T)

SYMMETRY PLANE SPECIALISATION
===============================
For the symmetry plane computation:
  - T is constrained to be a *reflection* (parametrised by (n, d))
  - α = 0  (no prior on the plane)
  - X¹ = X² = the same surface point cloud  X

The M-step then reduces to the closed-form plane fit implemented in
``SymmetryPlane.fit()`` (see ``_symmetry.py``).

IMPLEMENTATION NOTE — E-STEP TRANSPOSITION
============================================
The paper's E-step formula normalises *by row* (for each target point x_i,
the weights over all source points x_j sum to 1).

The C++ EM_ICPSym implementation (``Symmetry/EM-ICPSym.hh``) normalises
*by column* (for each source point x_j, the weights over all model points
x_i sum to 1).  Since X¹ = X², the two normalisation choices are equivalent
by symmetry of the criterion.  This implementation follows the C++ convention
(column-wise normalisation).

KD-TREE CUT-OFF FOR SPEED AND ROBUSTNESS
==========================================
Rather than summing over all N² pairs, the paper mentions a cut-off distance
above which pairs are eliminated.  This is implemented here as a radius search:
only model points within  r = ν_max × σ²  of the reflected source point are
considered as correspondences (``scipy.spatial.KDTree.query_ball_point``).

This serves two purposes:
  1. **Speed**: avoids O(N²) complexity; only O(N × k̄) pairs are retained,
     where k̄ is the average neighbourhood size (typically small).
  2. **Robustness to outliers**: pairs that are far apart receive negligible
     weight anyway (Gaussian decay), so excluding them does not change the
     result meaningfully while discarding genuine outliers.

ANNEALING / MULTI-SCALE σ
==========================
A crucial feature of the CLARCS framework is running EM with *decreasing* σ:

  "we devised a scheme in which several EM algorithms are successively run
   with decreasing σ values, with a large starting value when the point sets
   to register are far from each other." — Section 2.2 of the CLARCS paper

Large σ → broad Gaussians → smooth, convex criterion → good global behaviour
Small σ → narrow Gaussians → approaches hard ICP → precise local solution

In practice (matching the C++ EM_ICPSym::Annealing()):
  - We start with σ = σ_init and anneal down to σ = σ_final.
  - σ decreases by a factor K (initially K=1, then K=1.03 after the first
    iteration) whenever the plane estimate has converged at the current σ.
  - After the annealing phase, 50 additional iterations are run at σ_final
    using the full (un-subsampled) point cloud.

UNIFORM SUBSAMPLING FOR SPEED
===============================
At coarse σ, many points in a small neighbourhood carry essentially the same
information.  To avoid redundant computation the working set is periodically
*uniformly sub-sampled*: points within radius σ are merged into their
barycentre, and the resulting (fewer) representative points carry the weight
of all the merged points (stored in ``work_counts``).

This matches ``EM_ICPSym::unif_subsampling_fast()`` in the C++ codebase.

TWO VARIANTS
=============
``em_icp_sym``        (mirrors EM_ICPSym in EM-ICPSym.hh)
    Standard variant with row-normalised correspondences and annealing.

``em_icp_sym_corres`` (mirrors EM_ICPSymCorresSym in EM-ICPSymCorresSym.hh)
    Doubly-stochastic variant: the correspondence matrix A is normalised so
    that both rows AND columns sum to 1 (up to a factor of 2).  This enforces
    a stronger "one-to-one" constraint on the correspondences and is run at
    a fixed small σ (no annealing) as a final refinement step.

    NOTE ON THE C++ IMPLEMENTATION
    --------------------------------
    In the C++ EM_ICPSymCorresSym::E_Step(), the column sum is indexed as
    ``sumColumn[i]`` (row index) instead of ``sumColumn[col_idx]`` (actual
    column index of the non-zero entry).  This appears to be a latent bug
    that has no effect in practice because the final refinement uses the full
    cloud (Y_work = Y = X), making the correspondence matrix square and the
    two indices coincide when correspondences are near-diagonal.
    The present Python implementation uses the mathematically correct formula:
    column sums are accumulated at the actual column indices.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from pyclarcs._symmetry import SymmetryPlane


# ---------------------------------------------------------------------------
# Uniform subsampling
# ---------------------------------------------------------------------------

def _uniform_subsample(
    points: np.ndarray,
    radius: float,
    tree: KDTree,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge points within *radius* into barycentres (uniform subsampling).

    Iterates through the point set in order.  For each unprocessed point p_i,
    all unprocessed points within distance *radius* of p_i are gathered,
    replaced by their barycentre, and marked as processed.

    The output barycentres form a sparser working set, and ``counts[k]``
    records how many original points were merged into barycentre k.  The
    counts are later used as weights in the M-step (a barycentre representing
    10 points contributes 10× more than one representing a single point).

    This matches ``EM_ICPSym::unif_subsampling_fast()`` in the C++ codebase,
    where the subsampling radius is taken equal to the current σ.

    Parameters
    ----------
    points : ndarray (N, 3)
    radius : float   – merging radius (= current σ during annealing)
    tree   : KDTree  – pre-built on *points* for efficient range queries

    Returns
    -------
    barycentres : ndarray (M, 3)  – representative points (M ≤ N)
    counts      : ndarray (M,)   – number of original points merged into each
    """
    n = len(points)
    merged = np.zeros(n, dtype=bool)    # True once a point has been merged
    bary_list: list[np.ndarray] = []
    count_list: list[int] = []

    for i in range(n):
        if merged[i]:
            continue   # already merged into a previous barycentre

        # Find all (unmerged) points within the merging radius
        neighbours = tree.query_ball_point(points[i], radius)
        active = [j for j in neighbours if not merged[j]]
        if not active:
            active = [i]

        # Mark as merged and compute the barycentre
        for j in active:
            merged[j] = True
        bary_list.append(points[active].mean(axis=0))
        count_list.append(len(active))

    return np.array(bary_list), np.array(count_list, dtype=float)


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
) -> tuple[list[list[int]], list[list[float]]]:
    """Compute the soft-correspondence matrix A (E-step).

    For each working point  x_j  in *work_pts*:

      1. Reflect:  x_j' = T(x_j)   (current plane estimate T)
      2. Find all model points  x_i  within radius  r = ν_max × σ²  of  x_j'
      3. Compute raw weights:
             raw_{i,j} = exp(−||x_j' − x_i||² / (2σ²))

    **Standard variant** (``doubly_stochastic=False``):
      Normalise *by column* (for each working point x_j, weights over all
      matched model points sum to 1):
             A_{i,j} = raw_{i,j} / Σ_k raw_{k,j}

      This matches ``EM_ICPSym::nu_fast()`` in the C++ codebase.

    **Doubly-stochastic variant** (``doubly_stochastic=True``):
      Normalise so that both row sums (over j for fixed i) and column sums
      (over i for fixed j) are equal.  The normalisation used here is:
             A_{i,j} = 2 × raw_{i,j} / (row_sum_i × col_sum_j)
      where  row_sum_i = Σ_k raw_{i,k}  and  col_sum_j = Σ_k raw_{k,j}.

      This matches ``EM_ICPSymCorresSym::E_Step()`` in the C++ codebase
      (with the column-sum index correction described in the module docstring).

    The cut-off radius  r = ν_max × σ²  eliminates pairs that would receive
    negligibly small weights, giving O(N) instead of O(N²) complexity.

    Parameters
    ----------
    work_pts        : ndarray (M, 3) – current sub-sampled source points
    model_tree      : KDTree built on the full model cloud
    model_pts       : ndarray (N, 3) – full model cloud
    plane           : SymmetryPlane  – current plane estimate
    sigma_sq        : float – σ² (bandwidth squared)
    nu_max          : float – cut-off factor; radius = ν_max × σ²
    doubly_stochastic : bool – use doubly-stochastic normalisation

    Returns
    -------
    neighbour_ids : list of M lists of int  – column indices of non-zero A entries
    weights       : list of M lists of float – corresponding normalised weights
    """
    radius = nu_max * sigma_sq       # cut-off distance for neighbour search
    n_work = len(work_pts)

    # Reflect all working points in one batch
    reflected = plane.apply(work_pts)   # T(x_j) for all j

    # Range search: for each reflected point T(x_j), find all model points
    # x_i within the cut-off radius.  Returns a list of index lists.
    neighbour_ids = model_tree.query_ball_point(reflected, radius)

    # Compute raw (unnormalised) Gaussian weights
    raw: list[list[float]] = []
    col_sums: dict[int, float] = {}   # accumulated column sums (for doubly-stochastic)
    row_sums: list[float] = []

    for i, (r_pt, nbrs) in enumerate(zip(reflected, neighbour_ids)):
        if not nbrs:
            raw.append([])
            row_sums.append(0.0)
            continue

        # Squared distances from T(x_j) to each neighbour x_i
        # Shape: (|nbrs|, 3) → reduce to (|nbrs|,)
        nbr_arr = model_pts[nbrs]
        dist_sq = np.sum((r_pt - nbr_arr) ** 2, axis=1)

        # Gaussian kernel:  exp(−||T(x_j) − x_i||² / (2σ²))
        w = np.exp(-dist_sq / (2.0 * sigma_sq))
        raw.append(w.tolist())

        rs = float(w.sum())
        row_sums.append(rs)

        if doubly_stochastic:
            # Accumulate column sums: for each model point i that is a
            # neighbour, add its raw weight to col_sums[i]
            for k, j_glob in enumerate(nbrs):
                col_sums[j_glob] = col_sums.get(j_glob, 0.0) + w[k]

    # --- Normalisation ---
    if not doubly_stochastic:
        # Column-wise normalisation (matches EM_ICPSym):
        # for each working point j, weights over its matched model points sum to 1
        weights: list[list[float]] = []
        for i, (nbrs, w_row, rs) in enumerate(zip(neighbour_ids, raw, row_sums)):
            if not nbrs or rs == 0.0:
                weights.append([])
            else:
                weights.append([w / rs for w in w_row])
        return neighbour_ids, weights

    # Doubly-stochastic normalisation (matches EM_ICPSymCorresSym):
    # A_{i,j} = 2 × raw_{i,j} / (row_sum_i × col_sum_j)
    weights_ds: list[list[float]] = []
    for i, (nbrs, w_row, rs) in enumerate(zip(neighbour_ids, raw, row_sums)):
        if not nbrs or rs == 0.0:
            weights_ds.append([])
            continue
        row: list[float] = []
        for k, j_glob in enumerate(nbrs):
            cs = col_sums.get(j_glob, 0.0)
            if cs > 0.0 and rs > 0.0:
                row.append(2.0 * w_row[k] / (rs * cs))
            else:
                row.append(0.0)
        weights_ds.append(row)

    return neighbour_ids, weights_ds


# ---------------------------------------------------------------------------
# M-step: fit the symmetry plane to the weighted correspondences
# ---------------------------------------------------------------------------

def _m_step(
    work_pts: np.ndarray,
    model_pts: np.ndarray,
    work_counts: np.ndarray,
    neighbour_ids: list[list[int]],
    weights: list[list[float]],
    plane: SymmetryPlane,
) -> None:
    """Fit the symmetry plane to the current weighted correspondences (M-step).

    Collects all (source, target, weight) triplets from the sparse
    correspondence matrix A and calls ``SymmetryPlane.fit()``.

    The effective weight of a pair (x_j, x_i) is:
        effective_weight = A_{i,j} × count_j
    where  count_j  is the number of original points represented by the
    barycentre x_j (from uniform subsampling).  This ensures that barycentres
    representing many original points contribute proportionally more to the fit.

    Parameters
    ----------
    work_pts      : ndarray (M, 3)  – current working (possibly sub-sampled) points
    model_pts     : ndarray (N, 3)  – full model cloud
    work_counts   : ndarray (M,)    – number of original points per barycentre
    neighbour_ids : list of M lists – column indices of non-zero A entries
    weights       : list of M lists – corresponding normalised weights
    plane         : SymmetryPlane   – updated in place
    """
    # Accumulate the (source, target, weight) lists for SymmetryPlane.fit()
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ws: list[float] = []

    for j, (nbrs, w_row) in enumerate(zip(neighbour_ids, weights)):
        if not nbrs:
            continue
        count = float(work_counts[j])
        for k, i_glob in enumerate(nbrs):
            # source = working point x_j (to be reflected)
            # target = model point x_i  (desired position after reflection)
            sources.append(work_pts[j])
            targets.append(model_pts[i_glob])
            ws.append(w_row[k] * count)

    if not sources:
        return   # no correspondences found (σ too small or cloud too sparse)

    plane.fit(
        np.array(sources),
        np.array(targets),
        np.array(ws),
    )


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
    """Fine symmetry optimisation with EM-ICP and simulated annealing.

    Implements the annealing EM-ICP loop from ``EM_ICPSym::optimize()``
    (``Symmetry/EM-ICPSym.hh``).

    Algorithm
    ---------
    1. Initialise  σ = σ_init,  K = 1  (annealing rate, starts at 1).
    2. Compute a uniform sub-sample of the cloud at radius σ.
    3. **E-step**: for each sub-sampled point, reflect and find soft
       correspondences with all model points within radius  ν_max × σ².
    4. **M-step**: fit the symmetry plane to the weighted correspondences.
    5. **Annealing**: if the plane has not changed by more than *convergence_tol*,
       decrease  σ ← max(σ/K, σ_final)  and re-subsample.
       After the first iteration, set K=1.03 (slow geometric cooling).
    6. Repeat from 3 until *max_iter* or σ == σ_final throughout.
    7. **Final refinement**: run 50 more E/M steps at σ_final using the full
       un-subsampled point cloud for maximum accuracy.

    Parameters
    ----------
    points        : ndarray (N, 3) – surface point cloud
    initial_plane : SymmetryPlane  – from the coarse stage
    sigma_init    : float – starting bandwidth (default 5.0, in C++ ``sigma=5``)
    sigma_final   : float – minimum bandwidth (default 0.5, in C++ ``sigma_final=0.5``)
    nu_max        : float – neighbourhood radius factor (default 1.5,
                            in C++ ``nuMax=1.5``)
    max_iter      : int   – annealing-phase iterations (default 400,
                            in C++ ``it_max=400``)
    max_iter_final: int   – additional iterations at σ_final (default 50,
                            in C++ the final ``supit < 50`` loop)
    convergence_tol : float – threshold for the annealing trigger (default 0.01,
                              matches the C++ ``<0.01`` comparisons)
    verbose       : bool  – print per-iteration progress

    Returns
    -------
    SymmetryPlane – finely optimised plane (to be passed to ``em_icp_sym_corres``)
    """
    plane = initial_plane.copy()
    tree = KDTree(points)

    # Annealing parameters matching the C++ implementation:
    # K=1 initially (no decrease on the very first convergence detection),
    # then K=1.03 from the second iteration onward.
    sigma = float(sigma_init)
    K = 1.0  # C++: initially K=1, then K=1.03 after it>0

    # Initial uniform sub-sample at the starting σ
    work_pts, work_counts = _uniform_subsample(points, sigma, tree)
    need_resample = False   # flag: recompute the working set after σ changes

    for it in range(max_iter):
        if need_resample:
            work_pts, work_counts = _uniform_subsample(points, sigma, tree)
            need_resample = False

        sigma_sq = sigma * sigma
        d_prev = plane.d
        n_prev = plane.n.copy()

        # E-step: compute soft correspondences
        nbrs, wts = _e_step(
            work_pts, tree, points, plane, sigma_sq, nu_max, doubly_stochastic=False
        )
        # M-step: fit the plane
        _m_step(work_pts, points, work_counts, nbrs, wts, plane)

        if verbose:
            print(
                f"  [EM_ICP] iter {it:4d}  σ={sigma:.4f}"
                f"  d={plane.d:.4f}  n={plane.n}"
            )

        # --- Annealing trigger ---
        # Decrease σ when the plane estimate has converged at the current level
        # (both |Δd| and each component of |Δn| below the tolerance).
        # This matches the C++ condition in EM_ICPSym::Annealing().
        if sigma > sigma_final:
            converged_here = (
                abs(plane.d - d_prev) < convergence_tol
                and abs(plane.n[0] - n_prev[0]) < convergence_tol
                and abs(plane.n[1] - n_prev[1]) < convergence_tol
                and abs(plane.n[2] - n_prev[2]) < convergence_tol
            )
            if converged_here:
                sigma = max(sigma / K, sigma_final)
                need_resample = True   # new σ → new sub-sample radius
                if verbose:
                    print(f"    σ decreased → {sigma:.4f}")

        # Set K = 1.03 after the first iteration (slow geometric cooling)
        # matches  if(it>0) K=1.03;  in the C++ code
        if it == 0:
            K = 1.03

    # --- Final refinement at σ_final (full cloud, no sub-sampling) ---
    # Matches the C++ block: sigma=sigma_final; n_deci.resize(Y->size(),1);
    # Y_work=(*Y);  for(supit<50) { E_step; M_step; }
    sigma_sq_final = sigma_final * sigma_final
    full_counts = np.ones(len(points))

    for _ in range(max_iter_final):
        nbrs, wts = _e_step(
            points, tree, points, plane, sigma_sq_final, nu_max, doubly_stochastic=False
        )
        _m_step(points, points, full_counts, nbrs, wts, plane)

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
    verbose: bool = False,
) -> SymmetryPlane:
    """Final symmetry refinement with doubly-stochastic EM-ICP.

    Implements ``EM_ICPSymCorresSym::optimize()``
    (``Symmetry/EM-ICPSymCorresSym.hh``).

    This variant differs from ``em_icp_sym`` in two ways:
      1. **Doubly-stochastic normalisation**: the correspondence matrix A is
         normalised so that both its row sums and column sums are equal (up
         to a factor of 2).  This enforces an approximate one-to-one mapping
         between the two sides of the surface, reducing "many-to-one"
         artefacts where a single well-symmetric point dominates.
      2. **No annealing**: the algorithm runs entirely at the given fixed σ
         (σ = 0.25 mm by default, a very narrow bandwidth corresponding to
         near-hard correspondences).  It is intended to be run *after*
         ``em_icp_sym`` has already brought σ close to σ_final, so annealing
         is no longer necessary.

    Parameters
    ----------
    points        : ndarray (N, 3) – surface point cloud
    initial_plane : SymmetryPlane  – from the ``em_icp_sym`` stage
    sigma         : float – fixed bandwidth (default 0.25,
                            in C++ ``setSigma(0.25)``)
    nu_max        : float – neighbourhood radius factor (default 1.5)
    max_iter      : int   – number of EM iterations (default 100,
                            in C++ ``it_max=100``)
    verbose       : bool  – print per-iteration progress

    Returns
    -------
    SymmetryPlane – final refined plane
    """
    plane = initial_plane.copy()
    tree = KDTree(points)
    sigma_sq = sigma * sigma

    # Use the full point cloud (no subsampling), weight 1 per point.
    # Matches:  sigma = sigma_final;  n_deci.resize(Y->size(),1);  Y_work = (*Y);
    counts = np.ones(len(points))

    for it in range(max_iter):
        # E-step with doubly-stochastic normalisation
        nbrs, wts = _e_step(
            points, tree, points, plane, sigma_sq, nu_max, doubly_stochastic=True
        )
        # M-step: plane fitting
        _m_step(points, points, counts, nbrs, wts, plane)

        if verbose:
            print(f"  [EM_corres] iter {it:4d}  d={plane.d:.4f}  n={plane.n}")

    if verbose:
        print(f"  [EM_corres] final plane: {plane}")

    return plane
