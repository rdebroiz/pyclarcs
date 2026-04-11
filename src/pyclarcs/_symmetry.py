"""
Core symmetry plane class.

A symmetry plane is defined by its unit normal n and offset d:
    plane equation : n · x = d
    reflection of p: p' = p - 2*(n·p - d)*n  =  (I - 2*n*nᵀ)*p + 2*d*n

This mirrors the Symmetry class in Transform/Symmetry.hh from the C++ codebase.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


class SymmetryPlane:
    """Reflexive symmetry plane in 3-D."""

    def __init__(self, n: ArrayLike | None = None, d: float = 0.0) -> None:
        """
        Parameters
        ----------
        n : array-like of shape (3,), optional
            Unit normal to the plane (will be normalised). Defaults to (1, 0, 0).
        d : float
            Signed distance from the origin: the plane passes through the point n*d.
        """
        if n is None:
            n = np.array([1.0, 0.0, 0.0])
        self.n = np.asarray(n, dtype=float).copy()
        norm = np.linalg.norm(self.n)
        if norm > 0:
            self.n /= norm
        self.d = float(d)
        self._build_matrix()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_matrix(self) -> None:
        """Pre-compute the 3x3 reflection matrix R and translation t.

        Reflection: p' = R @ p + t
        where R = I - 2*n*nᵀ  and  t = 2*d*n
        """
        n = self.n
        self._R = np.eye(3) - 2.0 * np.outer(n, n)
        self._t = 2.0 * self.d * n

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Reflect an array of points through the plane.

        Parameters
        ----------
        points : ndarray of shape (N, 3) or (3,)

        Returns
        -------
        ndarray, same shape as input
        """
        pts = np.asarray(points, dtype=float)
        return pts @ self._R.T + self._t

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from each point to the plane.

        Positive on the side of n, negative on the other side.

        Parameters
        ----------
        points : ndarray of shape (N, 3)
        """
        return np.dot(points, self.n) - self.d

    def project(self, points: np.ndarray) -> np.ndarray:
        """Orthogonal projection of points onto the plane."""
        dist = self.signed_distance(points)
        return points - np.outer(dist, self.n)

    # ------------------------------------------------------------------
    # Plane fitting (M-step)
    # ------------------------------------------------------------------

    def fit(
        self,
        source: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        """Fit the symmetry plane so that source[i] reflects close to target[i].

        Implements the weighted algebraic fitting from Symmetry::set() in the
        C++ codebase.  The optimal normal n is the eigenvector of M associated
        with the *smallest* eigenvalue, where:

            M = Σ_i w_i * ( z_i * z_iᵀ  −  Δ_i * Δ_iᵀ )
            z_i = source[i] + target[i] - g_src - g_tgt
            Δ_i = target[i] - source[i]

        Parameters
        ----------
        source  : ndarray (N, 3) – points to be reflected
        target  : ndarray (N, 3) – desired positions after reflection
        weights : ndarray (N,), optional – per-pair weights (default: uniform 1)
        """
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        n_pts = len(source)

        if weights is None:
            weights = np.ones(n_pts, dtype=float)
        else:
            weights = np.asarray(weights, dtype=float)

        w_sum = weights.sum()
        if w_sum == 0:
            return

        # Weighted centres of mass
        g_src = (source * weights[:, None]).sum(axis=0) / w_sum
        g_tgt = (target * weights[:, None]).sum(axis=0) / w_sum

        # Build the symmetric 3x3 matrix M (vectorised)
        z = source + target - g_src - g_tgt  # (N, 3)
        delta = target - source               # (N, 3)

        M = (
            np.einsum("ij,ik,i->jk", z, z, weights)
            - np.einsum("ij,ik,i->jk", delta, delta, weights)
        )

        # Eigenvector for the smallest eigenvalue → plane normal
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        n_new = eigenvectors[:, 0]  # eigh returns ascending eigenvalues

        # Plane passes through the mid-point of the two centres of mass
        d_new = np.dot((g_src + g_tgt) * 0.5, n_new)

        # Normalise sign convention: d ≥ 0
        if d_new < 0:
            n_new = -n_new
            d_new = -d_new

        self.n = n_new / np.linalg.norm(n_new)
        self.d = float(d_new)
        self._build_matrix()

    # ------------------------------------------------------------------
    # Serialisation (.pl format from the C++ codebase)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save to a .pl text file (compatible with the C++ ZZ_SYMC format).

        Format::

            n  <nx>  <ny>  <nz>
            p  <px>  <py>  <pz>

        where p = n * d is a point on the plane.
        """
        p = self.n * self.d
        with open(path, "w") as fh:
            fh.write(f"n {self.n[0]} {self.n[1]} {self.n[2]}\n")
            fh.write(f"p {p[0]} {p[1]} {p[2]}\n")

    @classmethod
    def load(cls, path: str) -> "SymmetryPlane":
        """Load from a .pl file (C++ ZZ_SYMC format).

        The file stores the normal n and a point p on the plane.
        d is recovered as d = n · p.
        """
        with open(path) as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        n_vals = list(map(float, lines[0].split()[1:4]))
        p_vals = list(map(float, lines[1].split()[1:4]))
        n = np.array(n_vals)
        p = np.array(p_vals)
        d = float(np.dot(n, p))
        return cls(n, d)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def copy(self) -> "SymmetryPlane":
        return SymmetryPlane(self.n.copy(), self.d)

    def __repr__(self) -> str:
        return (
            f"SymmetryPlane(n=[{self.n[0]:.4f}, {self.n[1]:.4f}, {self.n[2]:.4f}],"
            f" d={self.d:.4f})"
        )
