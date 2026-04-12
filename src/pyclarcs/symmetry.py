"""
Core symmetry plane class.

THEORETICAL BACKGROUND
======================
The method implemented in this module is described in:

  Combès B., Hennessy R., Waddington J., Roberts N., Prima S.
  "Automatic symmetry plane estimation of bilateral objects in point clouds."
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR'2008).
  Anchorage, United States (2008).  [ref 7 in the CLARCS paper]

and used within the broader CLARCS framework:

  Abadie A., Combès B., Haegelen C., Prima S.
  "CLARCS, a C++ Library for Automated Registration and Comparison of Surfaces:
   Medical Applications."
  MICCAI Workshop on Mesh Processing in Medical Image Analysis (MeshMed'2011).
  Toronto, Canada, pp. 117-126.

SYMMETRY PLANE PARAMETRISATION
================================
A symmetry plane is parametrised by:
  - n : a unit normal vector in ℝ³
  - d : the signed distance from the origin to the plane

The plane equation is:  { x ∈ ℝ³ | n · x = d }

A point p on the same side as n*d satisfies n · p > d, and its reflection
through the plane is:

    p' = p − 2 (n·p − d) n                            (geometric formula)
       = (I − 2 n nᵀ) p + 2 d n                        (matrix form)
       =    R · p      +   t                            (compact notation)

where  R = I − 2 n nᵀ  is the reflection matrix (symmetric, R² = I),
and    t = 2 d n        is the translation vector.

This decomposition is precomputed in ``_build_matrix()`` and cached so that
``apply()`` is fast.

PLANE FITTING (M-STEP)
=======================
Given N weighted point pairs (source[i], target[i]) with weights w[i], the
optimal symmetry plane is obtained by minimising the weighted sum of squared
distances between each reflected source point and its corresponding target:

    argmin_{n, d}  Σ_i  w_i · ||target[i] − T(source[i])||²

where T(·) is the reflection through the plane (n, d).

Following the closed-form solution from the CLARCS paper (CVPR 2008):
Let  g_s = weighted mean of source points,
     g_t = weighted mean of target points.

Define for each pair:
    z_i  = source[i] + target[i] − g_s − g_t      (midpoint deviation)
    Δ_i  = target[i] − source[i]                   (displacement vector)

Build the symmetric 3×3 matrix:
    M = Σ_i  w_i · ( z_i z_iᵀ − Δ_i Δ_iᵀ )

The optimal normal n is the eigenvector of M associated with the *smallest*
eigenvalue (minimises n·M·n).  Intuitively, n must be roughly parallel to
the Δ_i vectors (each source reflects to its target), which are penalised
by the −Δ_i Δ_iᵀ term.

Once n is found, d is recovered from the midpoint of the two centres:
    d = n · (g_s + g_t) / 2

FILE FORMAT (.pl)
=================
The symmetry plane is serialised in the ASCII text format used by the
original ZZ_SYMC C++ tool:

    n  <nx>  <ny>  <nz>
    p  <px>  <py>  <pz>

where p = n · d is a point lying on the plane.  At load time, d is recovered
as d = n · p.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


class SymmetryPlane:
    """Reflexive symmetry plane in 3-D (parametrised by unit normal n and offset d).

    See module docstring for the full mathematical background.
    """

    def __init__(self, n: ArrayLike | None = None, d: float = 0.0) -> None:
        """
        Parameters
        ----------
        n : array-like of shape (3,), optional
            Unit normal to the plane.  Will be normalised automatically.
            Defaults to (1, 0, 0) (the YZ plane).
        d : float
            Signed distance from the origin:  n · x = d  defines the plane.
            The point n*d therefore lies on the plane.
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
        """Pre-compute and cache the 3×3 reflection matrix R and translation t.

        The reflection  p' = R p + t  with
            R = I − 2 n nᵀ    (symmetric, orthogonal: Rᵀ = R, R² = I)
            t = 2 d n

        is the closed-form reflection through the plane  {x | n·x = d}.
        This pre-computation makes ``apply()`` a single matrix-vector multiply.
        """
        n = self.n
        # Reflection matrix: I − 2 n nᵀ
        self._R = np.eye(3) - 2.0 * np.outer(n, n)
        # Translation: 2 d n
        self._t = 2.0 * self.d * n

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Reflect an array of points through the symmetry plane.

        The reflection formula  p' = R p + t  (see ``_build_matrix``) is
        applied to every row of *points*.

        Parameters
        ----------
        points : ndarray of shape (N, 3) or (3,)
            Input points.

        Returns
        -------
        ndarray, same shape as input
            Reflected points.
        """
        pts = np.asarray(points, dtype=float)
        # R is symmetric so R @ p = p @ Rᵀ = p @ R  (equivalent, but
        # using matmul with broadcasting is cleaner for batches)
        return pts @ self._R.T + self._t

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from each point to the plane  n·x = d.

        Positive on the side where n·x > d (the normal side),
        negative on the opposite side.

        Parameters
        ----------
        points : ndarray of shape (N, 3)

        Returns
        -------
        ndarray of shape (N,)
        """
        return np.dot(points, self.n) - self.d

    def project(self, points: np.ndarray) -> np.ndarray:
        """Orthogonal projection of points onto the plane.

        Each point is displaced by  −dist(p) · n  where dist(p) is its
        signed distance to the plane.

        Parameters
        ----------
        points : ndarray of shape (N, 3)

        Returns
        -------
        ndarray of shape (N, 3)
        """
        dist = self.signed_distance(points)
        return points - np.outer(dist, self.n)

    # ------------------------------------------------------------------
    # Plane fitting (M-step of the EM-ICP algorithm)
    # ------------------------------------------------------------------

    def fit(
        self,
        source: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        """Fit the symmetry plane so that source[i] reflects close to target[i].

        This implements the closed-form M-step from the CLARCS paper (CVPR 2008),
        which minimises the weighted sum:

            Σ_i  w_i · ||target[i] − T(source[i])||²

        over all planes T parametrised by (n, d).

        Algorithm
        ---------
        1. Compute weighted centres of mass  g_s  and  g_t.
        2. Build the 3×3 matrix  M = Σ_i w_i (z_i z_iᵀ − Δ_i Δ_iᵀ)
           with  z_i = source[i]+target[i]−g_s−g_t  and  Δ_i = target[i]−source[i].
        3. The optimal normal n is the eigenvector of M for the smallest eigenvalue
           (computed via numpy's symmetric eigendecomposition, which sorts
           eigenvalues in ascending order).
        4. Recover  d = n · (g_s + g_t) / 2  and enforce  d ≥ 0.

        Parameters
        ----------
        source  : ndarray (N, 3) – points to be reflected (play the role of X²)
        target  : ndarray (N, 3) – desired positions after reflection (role of X¹)
        weights : ndarray (N,), optional – soft-correspondence weights A_{i,j}
                  (default: uniform 1, corresponding to hard ICP)
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
            return  # degenerate: no valid correspondences, keep current plane

        # Step 1 — weighted centres of mass
        # g_s ≈ mean position of source points (weighted),
        # g_t ≈ mean position of target points (weighted).
        # For a perfect bilateral surface, g_s ≈ g_t and both lie near the plane.
        g_s = (source * weights[:, None]).sum(axis=0) / w_sum
        g_t = (target * weights[:, None]).sum(axis=0) / w_sum

        # Step 2 — symmetric 3×3 matrix M (vectorised)
        #
        # z_i  = (source[i] + target[i]) − (g_s + g_t)
        #       encodes the deviation of the midpoint from the global midpoint
        # Δ_i  = target[i] − source[i]
        #       points from source to target, approximately parallel to n
        #
        # The term  +z_i z_iᵀ  forces n to lie in the plane containing all
        # midpoints; the term  −Δ_i Δ_iᵀ  enforces n ∥ Δ_i (as expected for
        # a symmetry: if source reflects to target, their difference is ⊥ to
        # the plane, i.e. ∥ n).
        z = source + target - g_s - g_t       # (N, 3)
        delta = target - source                # (N, 3)

        # Efficient batched outer-product sum: M_jk = Σ_i w_i (z_ij z_ik − Δ_ij Δ_ik)
        M = (
            np.einsum("ij,ik,i->jk", z, z, weights)
            - np.einsum("ij,ik,i->jk", delta, delta, weights)
        )

        # Step 3 — eigendecomposition of the symmetric matrix M.
        # numpy.linalg.eigh is guaranteed to return eigenvalues in ascending
        # order for symmetric/Hermitian matrices.
        # The eigenvector for the *smallest* eigenvalue minimises n·M·n,
        # i.e. it is the optimal plane normal.
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        n_new = eigenvectors[:, 0]  # column 0 → smallest eigenvalue

        # Step 4 — recover d and enforce the sign convention d ≥ 0
        # d = n · midpoint of the two centres of mass
        d_new = np.dot((g_s + g_t) * 0.5, n_new)
        if d_new < 0:
            n_new = -n_new
            d_new = -d_new

        self.n = n_new / np.linalg.norm(n_new)
        self.d = float(d_new)
        self._build_matrix()

    # ------------------------------------------------------------------
    # Serialisation (.pl format used by the C++ ZZ_SYMC tool)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the symmetry plane to a .pl text file.

        The format is compatible with the C++ ZZ_SYMC tool (see
        ``Symmetry::save()`` in Transform/Symmetry.hh)::

            n  <nx>  <ny>  <nz>
            p  <px>  <py>  <pz>

        where  p = n * d  is a point lying on the plane.

        Parameters
        ----------
        path : str
            Output file path (conventionally with a .pl extension).
        """
        p = self.n * self.d  # a representative point on the plane
        with open(path, "w") as fh:
            fh.write(f"n {self.n[0]} {self.n[1]} {self.n[2]}\n")
            fh.write(f"p {p[0]} {p[1]} {p[2]}\n")

    @classmethod
    def load(cls, path: str) -> "SymmetryPlane":
        """Load a symmetry plane from a .pl text file.

        The file stores the unit normal n and a point p on the plane.
        The offset d is recovered as  d = n · p.

        Parameters
        ----------
        path : str
            Path to a .pl file written by ``save()`` or by the C++ ZZ_SYMC tool.

        Returns
        -------
        SymmetryPlane
        """
        with open(path) as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        # Line 0: "n  nx  ny  nz"
        n_vals = list(map(float, lines[0].split()[1:4]))
        # Line 1: "p  px  py  pz"  (a point on the plane)
        p_vals = list(map(float, lines[1].split()[1:4]))
        n = np.array(n_vals)
        p = np.array(p_vals)
        # d = n · p  (distance from origin to the plane along n)
        d = float(np.dot(n, p))
        return cls(n, d)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def copy(self) -> "SymmetryPlane":
        """Return a deep copy of this plane."""
        return SymmetryPlane(self.n.copy(), self.d)

    def __repr__(self) -> str:
        return (
            f"SymmetryPlane(n=[{self.n[0]:.4f}, {self.n[1]:.4f}, {self.n[2]:.4f}],"
            f" d={self.d:.4f})"
        )
