"""
Coarse symmetry plane optimisation.

Mirrors PointCloudSymmetryPlane from Symmetry/PointCloudSymmetryPlane.hh.

Algorithm (ICP-style with trimmed estimator):
1. Reflect each point through the current plane estimate.
2. For each reflected point find its nearest neighbour in the original cloud.
3. Reject the `trim_fraction` of pairs with the largest distance (trimmed est.).
4. Fit a new symmetry plane to the remaining correspondences.
5. Repeat from 1 until convergence or max_iter.

A multi-resolution scheme starts with a random sub-sample of the cloud and
progressively adds more points, matching the coarse_resol logic from C++.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from pyclarcs._symmetry import SymmetryPlane


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_subsample(points: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Return a random sub-sample of *points* keeping *fraction* of them."""
    n_keep = max(1, int(len(points) * fraction))
    idx = rng.choice(len(points), size=n_keep, replace=False)
    return points[idx]


def _icp_step(
    points: np.ndarray,
    tree: KDTree,
    plane: SymmetryPlane,
    trim_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One ICP step: reflect, find NN, trim outliers.

    Returns
    -------
    source  : ndarray (M, 3) – kept source points
    target  : ndarray (M, 3) – their NN correspondences
    weights : ndarray (M,)   – all ones (trimmed = rejected, not down-weighted)
    """
    reflected = plane.apply(points)
    dists, indices = tree.query(reflected)

    n_keep = len(points) - int(len(points) * trim_fraction)
    n_keep = max(1, n_keep)

    if trim_fraction > 0:
        keep = np.argsort(dists)[:n_keep]
    else:
        keep = np.arange(len(points))

    source = points[keep]
    # target[i] is the NN in the model cloud of the reflection of source[i]
    target = np.asarray(tree.data)[indices[keep]]
    weights = np.ones(len(keep), dtype=float)
    return source, target, weights


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
    """ICP-based coarse symmetry plane optimisation.

    Matches the PointCloudSymmetryPlane::optimize() logic.

    Parameters
    ----------
    points : ndarray (N, 3)
    initial_plane : SymmetryPlane – starting estimate
    trim_fraction : float – fraction of worst pairs rejected per iteration
    max_iter : int – maximum iterations per resolution level
    n_resolutions : int – number of sub-sampling levels (coarse → fine)
    convergence_tol : float – stop when both |Δd| and |Δn| are below this
    verbose : bool
    seed : int or None – for reproducibility of sub-sampling

    Returns
    -------
    SymmetryPlane – optimised plane
    """
    rng = np.random.default_rng(seed)
    plane = initial_plane.copy()

    # Build the full-resolution KD-tree once (model cloud)
    full_tree = KDTree(points)

    # Resolution schedule: fractions of the full point set
    # Matches  resol_factor[i] = 1 - 1/2^i  from C++
    resolutions = [
        max(50, int(len(points) * (1.0 - 1.0 / (2 ** i))))
        for i in range(n_resolutions)
    ]
    resolutions.append(len(points))  # final pass at full resolution

    for level, n_pts in enumerate(resolutions):
        if verbose:
            print(f"  [coarse] resolution {level}/{len(resolutions)-1}: {n_pts} pts")

        if n_pts < len(points):
            sub = _random_subsample(points, n_pts / len(points), rng)
        else:
            sub = points

        d_prev = plane.d + 1e9  # force at least one iteration
        n_prev = np.zeros(3)
        nb_small = 0

        for it in range(max_iter):
            source, target, weights = _icp_step(sub, full_tree, plane, trim_fraction)
            plane.fit(source, target, weights)

            # Convergence check
            if (
                abs(plane.d - d_prev) < convergence_tol
                and np.linalg.norm(plane.n - n_prev) < convergence_tol
            ):
                nb_small += 1
                if nb_small >= 30:
                    if verbose:
                        print(f"    converged at iter {it}")
                    break
            else:
                nb_small = 0

            d_prev = plane.d
            n_prev = plane.n.copy()

    return plane
