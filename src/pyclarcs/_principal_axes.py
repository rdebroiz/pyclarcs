"""
Principal-axes initialisation for the symmetry plane.

ROLE IN THE PIPELINE
=====================
Before running the EM-ICP optimisation, we need a reasonable initial estimate
of the symmetry plane.  The CLARCS method (and the present Python port) uses
the three *principal axes* of the point cloud as candidate symmetry planes.

For an anatomical surface that is *approximately* bilaterally symmetric
(e.g. a skull or endocranium), the mid-sagittal plane is expected to be
well aligned with the principal axis associated with the largest moment of
inertia (the axis of maximal extension of the cloud).  The other two axes
give two additional candidates.

INERTIA TENSOR
==============
Given a point cloud {p_i}, its inertia tensor about its centroid g is the
3×3 symmetric matrix:

         ┌  Σ(y²+z²)   −Σxy    −Σxz  ┐
    I =  │  −Σxy     Σ(x²+z²)  −Σyz  │
         └  −Σxz       −Σyz   Σ(x²+y²)┘

where (x, y, z) = p_i − g are the centred coordinates.

Each eigenvalue of I is the moment of inertia about the corresponding
principal axis (eigenvector).  The eigenvector with the *smallest* eigenvalue
corresponds to the axis along which the cloud is the most spread.

Candidate symmetry planes are then defined by taking each eigenvector as the
plane normal, with the offset  d = n · g  so that the plane passes through
the centroid.

SELECTION OF THE BEST CANDIDATE
================================
Among the three candidates, the best one is selected by evaluating the
*symmetry residual*: the sum of distances from each reflected point to its
nearest neighbour in the original cloud.  This criterion directly measures
how well the plane explains the bilateral structure of the cloud, which is
exactly the criterion minimised by the EM-ICP algorithm that follows.

The candidate with the lowest residual is used to initialise the optimisation.

Reference
---------
  Combès et al. "Automatic symmetry plane estimation of bilateral objects in
  point clouds." CVPR 2008.  (Section 3: initialisation strategy)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from pyclarcs._symmetry import SymmetryPlane


# ---------------------------------------------------------------------------
# Inertia tensor computation
# ---------------------------------------------------------------------------

def _inertia_tensor(points: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Compute the 3×3 inertia tensor of a point cloud about *centre*.

    The tensor is:

         I_xx = Σ (y²+z²),  I_yy = Σ (x²+z²),  I_zz = Σ (x²+y²)
         I_xy = Σ xy,        I_xz = Σ xz,         I_yz = Σ yz

    Assembled into the symmetric matrix:

         ┌  I_xx   −I_xy   −I_xz ┐
         │ −I_xy    I_yy   −I_yz │
         └ −I_xz   −I_yz    I_zz ┘

    This matches the manual construction in ``principal_axes::optimize()``
    from ``Symmetry/principal_axes.hh`` in the C++ codebase.

    Parameters
    ----------
    points : ndarray (N, 3)
    centre : ndarray (3,)  – point about which inertia is computed (usually the
                             centroid of the cloud)

    Returns
    -------
    ndarray (3, 3) – symmetric inertia tensor
    """
    # Centred coordinates
    p = points - centre
    x, y, z = p[:, 0], p[:, 1], p[:, 2]

    # Diagonal moments
    Ixx = float(np.sum(y * y + z * z))   # resistance to rotation about x-axis
    Iyy = float(np.sum(x * x + z * z))   # resistance to rotation about y-axis
    Izz = float(np.sum(y * y + x * x))   # resistance to rotation about z-axis

    # Off-diagonal (products of inertia, negated for the standard convention)
    Ixy = float(np.sum(x * y))
    Ixz = float(np.sum(x * z))
    Iyz = float(np.sum(y * z))

    return np.array([
        [ Ixx,  -Ixy,  -Ixz],
        [-Ixy,   Iyy,  -Iyz],
        [-Ixz,  -Iyz,   Izz],
    ])


# ---------------------------------------------------------------------------
# Symmetry residual (quality measure for a candidate plane)
# ---------------------------------------------------------------------------

def _symmetry_residual(points: np.ndarray, plane: SymmetryPlane) -> float:
    """Measure how well *plane* explains the bilateral symmetry of *points*.

    For each point p, we compute its reflection  p' = T(p), then find the
    nearest neighbour of p' in the original cloud.  The residual is the sum
    of those nearest-neighbour distances:

        R(plane) = Σ_i  min_j ||T(p_i) − p_j||

    A lower residual means the cloud is more symmetric with respect to the
    plane.  This criterion is used to rank the three principal-axis candidates
    and pick the best initialisation for the EM-ICP.

    This matches ``principal_axes::evaluate_dist_sym()`` from the C++ codebase.

    Parameters
    ----------
    points : ndarray (N, 3)
    plane  : SymmetryPlane

    Returns
    -------
    float – sum of nearest-neighbour distances after reflection
    """
    reflected = plane.apply(points)    # T(p_i) for all i
    tree = KDTree(points)
    dists, _ = tree.query(reflected)   # min_j ||T(p_i) − p_j||
    return float(np.sum(dists))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def principal_axes_candidates(points: np.ndarray) -> list[SymmetryPlane]:
    """Return the three principal-axis symmetry plane candidates, best first.

    Algorithm
    ---------
    1. Compute the centroid g of the point cloud.
    2. Build the 3×3 inertia tensor I about g.
    3. Compute the eigenvectors of I (= principal axes) via ``numpy.linalg.eigh``,
       which returns them sorted by ascending eigenvalue.
    4. For each eigenvector  v_i, define the candidate plane:
           n = v_i  (normalised, sign adjusted so that  n·g ≥ 0)
           d = n · g  (plane passes through the centroid)
    5. Evaluate the symmetry residual for each candidate.
    6. Return the three planes sorted from lowest to highest residual.

    Parameters
    ----------
    points : ndarray (N, 3)

    Returns
    -------
    list of three SymmetryPlane objects, sorted so index 0 is the best
    initialisation candidate (lowest symmetry residual).
    """
    # Step 1 — centroid
    centre = points.mean(axis=0)

    # Step 2 & 3 — inertia tensor and eigendecomposition
    # eigh() is used instead of eig() because the inertia tensor is symmetric:
    # it returns real eigenvalues and an orthonormal eigenvector basis.
    I = _inertia_tensor(points, centre)
    _, eigenvectors = np.linalg.eigh(I)   # columns = eigenvectors (ascending order)

    # Step 4 & 5 — build candidate planes and evaluate residuals
    candidates: list[tuple[float, SymmetryPlane]] = []
    for i in range(3):
        v = eigenvectors[:, i].copy()
        v /= np.linalg.norm(v)

        # Sign convention: the normal points toward the same side as the centroid
        # (ensures d = n·g > 0, consistent with the rest of the pipeline)
        if np.dot(v, centre) < 0:
            v = -v

        d = float(np.dot(v, centre))
        plane = SymmetryPlane(v, d)
        residual = _symmetry_residual(points, plane)
        candidates.append((residual, plane))

    # Step 6 — sort from best (lowest residual) to worst
    candidates.sort(key=lambda t: t[0])
    return [p for _, p in candidates]


def best_principal_axis_plane(points: np.ndarray) -> SymmetryPlane:
    """Return the single best principal-axis symmetry plane for *points*.

    This is a convenience wrapper around ``principal_axes_candidates()``
    that returns only the top-ranked candidate.

    Parameters
    ----------
    points : ndarray (N, 3)

    Returns
    -------
    SymmetryPlane
        The principal-axis plane with the lowest symmetry residual, used as
        the starting point for the coarse and fine optimisation stages.
    """
    return principal_axes_candidates(points)[0]
