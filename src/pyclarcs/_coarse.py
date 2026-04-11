"""
Coarse symmetry plane optimisation (ICP with trimmed estimator).

ROLE IN THE PIPELINE
=====================
After the principal-axis initialisation, the plane estimate may still be
off by several degrees.  The coarse stage runs a classic ICP loop to
quickly bring the plane into the basin of attraction of the fine EM-ICP
optimiser.

ALGORITHM
==========
The coarse optimiser is a special case of the general EM-ICP framework
(see ``_fine.py``) in the limit σ → 0:

  When σ → 0, the soft correspondences  A_{i,j} ∝ exp(−||x_i−T(x_j)||²/2σ²)
  collapse to hard, nearest-neighbour correspondences.  This is exactly the
  classic ICP algorithm of Besl & McKay (1992), which alternates:

    C-step: for each point p, find its nearest neighbour  NN(T(p))
            in the original cloud.
    T-step: fit T to minimise  Σ ||NN(T(p_i)) − T(p_i)||²

  In our case T is a reflection (symmetry plane), so the C-step finds:
    - Reflect each source point:  p' = T(p)
    - Look up the nearest neighbour of p' in the model cloud:  q = NN(p')
  and the T-step fits a new symmetry plane to the (source=p, target=q) pairs
  using the closed-form M-step from ``SymmetryPlane.fit()``.

TRIMMED ESTIMATOR (ROBUSTNESS TO OUTLIERS)
===========================================
The paper mentions a "cut-off distance" mechanism for "increased speed and
robustness to outliers" (Section 2.2 of the CLARCS paper).  In the coarse
stage this is implemented as a *trimmed estimator*: after computing all
nearest-neighbour distances, the ``trim_fraction`` of pairs with the largest
distances are discarded before fitting the plane.

This is equivalent to the TrimEstimator logic in the C++ class
``PointCloudSymmetryPlane`` (``Symmetry/PointCloudSymmetryPlane.hh``):
  nb_out = trim_fraction = 0.2  (20% of pairs rejected per iteration)
  reject = True

MULTI-SCALE SCHEME
===================
The paper states: "we devised a scheme in which several EM algorithms are
successively run with decreasing σ values, with a large starting value when
the point sets to register are far from each other."

The coarse stage mirrors this by working on *random sub-samples* of the
cloud, going from a coarse sub-sample (few points) to the full cloud.  This
multi-resolution approach is faster than running ICP directly on the full
set of 80 000+ points, and helps escape local minima.

The resolution schedule matches the C++ ``resol_factor[i] = 1 − 1/2^i``:
  Level 0: 0% of points  (uses max(50, 0) points → at least 50)
  Level 1: 50% of points
  Level 2: 75% of points
  Level 3: 87.5% of points
  Level 4: full cloud

Mirrors
-------
  PointCloudSymmetryPlane  (Symmetry/PointCloudSymmetryPlane.hh)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from pyclarcs._symmetry import SymmetryPlane


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _random_subsample(
    points: np.ndarray,
    n_keep: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a random sub-sample of *points* with exactly *n_keep* rows.

    Uses uniform sampling without replacement.

    Parameters
    ----------
    points : ndarray (N, 3)
    n_keep : int  – number of points to keep (clamped to [1, N])
    rng    : numpy random Generator for reproducibility

    Returns
    -------
    ndarray (n_keep, 3)
    """
    n_keep = min(max(1, n_keep), len(points))
    idx = rng.choice(len(points), size=n_keep, replace=False)
    return points[idx]


def _icp_step(
    points: np.ndarray,
    model_tree: KDTree,
    model_pts: np.ndarray,
    plane: SymmetryPlane,
    trim_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One ICP correspondence step with trimmed estimator.

    For each point  p  in *points*:
      1. Reflect:  p' = T(p)   (T = current symmetry plane estimate)
      2. Find NN:  q  = nearest neighbour of p' in the model cloud

    Then reject the ``trim_fraction`` of pairs (p, q) that have the largest
    distance  ||p' − q||  (these are likely outliers or poorly symmetric
    regions).

    Returns the kept (source, target) pairs for use in ``SymmetryPlane.fit()``.

    Parameters
    ----------
    points      : ndarray (M, 3) – current sub-sampled working set
    model_tree  : KDTree built on the full model cloud
    model_pts   : ndarray (N, 3) – full model cloud (= model_tree data)
    plane       : SymmetryPlane  – current plane estimate
    trim_fraction : float in [0, 1) – fraction of worst pairs to discard

    Returns
    -------
    source : ndarray (K, 3)  – kept source points  (K = M × (1 − trim_fraction))
    target : ndarray (K, 3)  – their NN correspondences after reflection
    """
    # Reflect every working point through the current plane estimate
    reflected = plane.apply(points)           # T(p_i) for all i

    # Find the nearest neighbour of each reflected point in the model cloud.
    # distances[i] = ||T(p_i) − NN(T(p_i))||
    # indices[i]   = index in model_pts of the nearest neighbour
    distances, indices = model_tree.query(reflected)

    # Trimmed estimator: keep only the n_keep closest pairs
    n_keep = max(1, len(points) - int(len(points) * trim_fraction))
    if trim_fraction > 0.0:
        # argsort gives indices that would sort distances in ascending order;
        # we keep the first n_keep (the pairs with the smallest distances)
        keep = np.argsort(distances)[:n_keep]
    else:
        keep = np.arange(len(points))

    source = points[keep]                   # original (unreflected) source points
    target = model_pts[indices[keep]]       # nearest neighbours of their reflections
    return source, target


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def coarse_symmetry(
    points: np.ndarray,
    initial_plane: SymmetryPlane,
    trim_fraction: float = 0.2,
    max_iter: int = 1500,
    n_resolutions: int = 4,
    convergence_tol: float = 5e-3,
    verbose: bool = False,
    seed: int | None = None,
) -> SymmetryPlane:
    """ICP-based coarse symmetry plane optimisation with trimmed estimator.

    Runs a multi-resolution ICP loop that goes from coarse sub-samples to the
    full point cloud.  At each resolution level the loop iterates until
    convergence (no significant change in the plane parameters) or until
    *max_iter* is reached.

    Parameters
    ----------
    points : ndarray (N, 3)
        The surface point cloud.
    initial_plane : SymmetryPlane
        Starting estimate (from the principal-axis initialisation).
    trim_fraction : float
        Fraction of worst point pairs to discard at each ICP step.
        Default 0.2 (= 20%) matches the ``nb_out`` parameter in the C++.
    max_iter : int
        Maximum number of ICP iterations per resolution level.
        Default 1500 matches the ``it_max1`` parameter in the C++.
    n_resolutions : int
        Number of sub-sampling resolution levels before the full cloud.
        Default 4 matches the ``coarse_resol=3`` (plus the full-res pass)
        in the C++, following the schedule  1 − 1/2^i.
    convergence_tol : float
        The loop stops at a given resolution when both:
          |Δd| < tol  AND  ||Δn|| < tol
        for 30 consecutive iterations (``nb_small_update == 30`` in C++).
    verbose : bool
        Print per-iteration progress.
    seed : int or None
        Random seed for the sub-sampling steps (for reproducibility).

    Returns
    -------
    SymmetryPlane
        Optimised coarse plane, to be used as initialisation for EM-ICP.
    """
    rng = np.random.default_rng(seed)
    plane = initial_plane.copy()

    # Build the KD-tree on the *full* model cloud once.
    # This is the "model" cloud; the working sub-sample only changes the
    # set of source points but correspondences are always looked up here.
    full_tree = KDTree(points)

    # Resolution schedule matching  resol_factor[i] = 1 − 1/2^i  from C++:
    # the number of points at level i is  N × (1 − 1/2^i), clamped to ≥ 50.
    resolutions = [
        max(50, int(len(points) * (1.0 - 1.0 / (2 ** i))))
        for i in range(n_resolutions)
    ]
    resolutions.append(len(points))   # final pass at full resolution

    for level, n_pts in enumerate(resolutions):
        if verbose:
            print(
                f"  [coarse] resolution {level}/{len(resolutions) - 1}:"
                f" {n_pts}/{len(points)} pts"
            )

        # Sub-sample the source cloud for this resolution level
        if n_pts < len(points):
            sub = _random_subsample(points, n_pts, rng)
        else:
            sub = points   # full cloud at the last level

        # --- ICP loop at this resolution level ---
        d_prev = plane.d + 1e9   # large sentinel to force at least one iteration
        n_prev = np.zeros(3)
        nb_small = 0            # counter for "30 consecutive small updates"

        for it in range(max_iter):
            # C-step (trimmed): reflect + NN search + trim outliers
            source, target = _icp_step(sub, full_tree, points, plane, trim_fraction)

            # T-step: fit the symmetry plane to the remaining pairs
            # Weights are uniform (hard ICP, no M-estimator at this stage)
            plane.fit(source, target)

            # --- Convergence check ---
            # Match the C++ condition: 30 successive iterations where both
            # |Δd| < tol  AND  max_component(Δn) < tol
            delta_d = abs(plane.d - d_prev)
            delta_n = np.linalg.norm(plane.n - n_prev)

            if delta_d < convergence_tol and delta_n < convergence_tol:
                nb_small += 1
                if nb_small >= 30:
                    if verbose:
                        print(f"    converged at iter {it} (30× small update)")
                    break
            else:
                nb_small = 0

            d_prev = plane.d
            n_prev = plane.n.copy()

    return plane
