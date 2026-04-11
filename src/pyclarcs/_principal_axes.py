"""
Principal-axes initialisation for the symmetry plane.

Mirrors the principal_axes class from Symmetry/principal_axes.hh.

The inertia tensor of the point cloud is diagonalised to obtain three
candidate symmetry planes (one per principal axis).  The best candidate
is the one minimising the sum of distances from each reflected point to
its nearest neighbour in the original cloud.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from pyclarcs._symmetry import SymmetryPlane


def _inertia_tensor(points: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Return the 3x3 inertia tensor of a point cloud around *centre*.

    This matches the manual construction in principal_axes::optimize():

        Ipx = Σ (y²+z²),  Ipy = Σ (x²+z²),  Ipz = Σ (y²+x²)
        Ipxy = Σ xy,       Ipxz = Σ xz,       Ipyz = Σ yz

    The symmetric matrix is then::

        [ Ipx   -Ipxy  -Ipxz ]
        [-Ipxy   Ipy   -Ipyz ]
        [-Ipxz  -Ipyz   Ipz  ]
    """
    p = points - centre
    x, y, z = p[:, 0], p[:, 1], p[:, 2]

    Ipx  = np.sum(y * y + z * z)
    Ipy  = np.sum(x * x + z * z)
    Ipz  = np.sum(y * y + x * x)
    Ipxy = np.sum(x * y)
    Ipxz = np.sum(x * z)
    Ipyz = np.sum(y * z)

    return np.array([
        [ Ipx,  -Ipxy, -Ipxz],
        [-Ipxy,  Ipy,  -Ipyz],
        [-Ipxz, -Ipyz,  Ipz ],
    ])


def _symmetry_residual(points: np.ndarray, plane: SymmetryPlane) -> float:
    """Sum of squared distances from each reflected point to its NN.

    Matches evaluate_dist_sym() from principal_axes.hh.
    """
    reflected = plane.apply(points)
    tree = KDTree(points)
    dists, _ = tree.query(reflected)
    return float(np.sum(dists))


def principal_axes_candidates(points: np.ndarray) -> list[SymmetryPlane]:
    """Return three candidate symmetry planes derived from the principal axes.

    The planes are **sorted** so that index 0 is the one with the lowest
    symmetry residual (best candidate for initialisation).

    Parameters
    ----------
    points : ndarray of shape (N, 3)

    Returns
    -------
    list of three SymmetryPlane objects, best first.
    """
    centre = points.mean(axis=0)
    I = _inertia_tensor(points, centre)

    # eigh returns eigenvalues in ascending order, eigenvectors as columns
    _, eigenvectors = np.linalg.eigh(I)

    planes: list[tuple[float, SymmetryPlane]] = []
    for i in range(3):
        v = eigenvectors[:, i]
        v = v / np.linalg.norm(v)
        # Ensure the normal points toward the side where n·g > 0
        if np.dot(v, centre) < 0:
            v = -v
        d = float(np.dot(v, centre))
        plane = SymmetryPlane(v, d)
        residual = _symmetry_residual(points, plane)
        planes.append((residual, plane))

    planes.sort(key=lambda t: t[0])
    return [p for _, p in planes]


def best_principal_axis_plane(points: np.ndarray) -> SymmetryPlane:
    """Return the single best principal-axis symmetry plane for *points*."""
    return principal_axes_candidates(points)[0]
