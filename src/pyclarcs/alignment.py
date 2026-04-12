"""
Surface alignment utilities (preprocessing for registration).

These functions implement the methods from ``RegisterUtil.cc`` in the C++
CLARCS codebase.  They are pure numpy operations: each function returns a
new point array and never modifies its input.

Functions
---------
align_center_of_mass      — translate to superpose centres of mass
align_rescale             — translate + scale to match dispersion
align_to_symmetry_plane   — rigid alignment to the canonical symmetry plane
reorient_axes             — permute coordinate axes

Reference
---------
  Abadie A., Combès B., Haegelen C., Prima S.
  "CLARCS, a C++ Library for Automated Registration and Comparison of
  Surfaces: Medical Applications."
  MeshMed 2011.
"""

from __future__ import annotations

import numpy as np

from pyclarcs.symmetry import SymmetryPlane


# ---------------------------------------------------------------------------
# Centre-of-mass alignment
# ---------------------------------------------------------------------------

def align_center_of_mass(
    points: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Translate *points* so its centre of mass coincides with *target*'s.

    Mirrors ``RegisterUtil -m centerofmass`` from the C++ CLARCS codebase.

    Parameters
    ----------
    points : ndarray (N, 3) — surface to move
    target : ndarray (M, 3) — reference surface

    Returns
    -------
    ndarray (N, 3) — translated copy of *points*
    """
    t = target.mean(axis=0) - points.mean(axis=0)
    return points + t


# ---------------------------------------------------------------------------
# Rescale (centre of mass + dispersion)
# ---------------------------------------------------------------------------

def align_rescale(
    points: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Translate and uniformly scale *points* to match *target*'s centre of
    mass and mean dispersion.

    Dispersion is defined as the mean Euclidean distance from each point to
    the centroid (same formula as ``superposeDispersion`` in the C++ code).

    Mirrors ``RegisterUtil -m rescale`` from the C++ CLARCS codebase.

    Parameters
    ----------
    points : ndarray (N, 3) — surface to move / rescale
    target : ndarray (M, 3) — reference surface

    Returns
    -------
    ndarray (N, 3) — translated and scaled copy of *points*
    """
    c_src = points.mean(axis=0)
    c_tgt = target.mean(axis=0)

    d_src = float(np.sqrt(np.sum((points - c_src) ** 2, axis=1)).mean())
    d_tgt = float(np.sqrt(np.sum((target - c_tgt) ** 2, axis=1)).mean())

    scale = d_tgt / d_src if d_src > 0.0 else 1.0

    # Faithful to the C++ applyToVect: result = v * scale + translation
    # with translation = c_tgt - c_src (from superpose_center_of_mass).
    return points * scale + (c_tgt - c_src)


# ---------------------------------------------------------------------------
# Symmetry-plane recentering
# ---------------------------------------------------------------------------

def align_to_symmetry_plane(
    points: np.ndarray,
    plane: SymmetryPlane,
) -> np.ndarray:
    """Rigidly align *points* so that *plane* becomes the canonical plane
    ``n = [1, 0, 0], d = 0`` (the YZ plane at x = 0).

    Algorithm
    ---------
    1. Adjust the plane normal so it points toward +x.
    2. Build a source frame on the plane:
       - ``dep1`` = orthogonal projection of the surface centroid onto the plane
       - ``e1``   = plane normal ``n`` (→ canonical x-axis)
       - ``e2``   = in-plane unit direction toward the projection of ``[0, 30, 0]``
                    (→ canonical y-axis; fallback to ``[0, 0, 30]`` if collinear)
       - ``e3``   = ``cross(e1, e2)`` (→ canonical z-axis)
    3. Build the 3×3 rotation ``R`` whose rows are ``[e1, e2, e3]``, so that
       ``R @ e1 = [1,0,0]``, ``R @ e2 = [0,1,0]``, ``R @ e3 = [0,0,1]``.
    4. Apply: ``result = R @ (p − dep1)``  for every point ``p``.

    Mirrors ``RegisterUtil -m symmetry`` from the C++ CLARCS codebase.

    Parameters
    ----------
    points : ndarray (N, 3)
    plane  : SymmetryPlane — symmetry plane as estimated by ``sym-plane``

    Returns
    -------
    ndarray (N, 3) — rigidly transformed copy of *points*
    """
    n = plane.n.copy()

    # Step 1 — ensure normal points toward +x
    if float(np.dot(n, np.array([1.0, 0.0, 0.0]))) < 0.0:
        n = -n

    # Step 2a — dep1: projection of surface centroid onto the plane
    centroid = points.mean(axis=0)
    dep1 = centroid - (float(np.dot(centroid, n)) - plane.d) * n

    # Step 2b — e2: in-plane direction toward projection of [0, 30, 0]
    ref_pt = np.array([0.0, 30.0, 0.0])
    proj_ref = ref_pt - (float(np.dot(ref_pt, n)) - plane.d) * n
    e2_dir = proj_ref - dep1

    if np.linalg.norm(e2_dir) < 1e-10:
        # Fallback: [0, 0, 30] if [0, 30, 0] is nearly parallel to n
        ref_pt = np.array([0.0, 0.0, 30.0])
        proj_ref = ref_pt - (float(np.dot(ref_pt, n)) - plane.d) * n
        e2_dir = proj_ref - dep1

    e2 = e2_dir / np.linalg.norm(e2_dir)

    # Step 2c — e3 = cross(e1, e2)
    e3 = np.cross(n, e2)
    e3 /= np.linalg.norm(e3)

    # Step 3 — rotation matrix (rows = source-frame basis vectors)
    R = np.array([n, e2, e3])   # R @ n = [1,0,0], R @ e2 = [0,1,0], R @ e3 = [0,0,1]

    # Step 4 — translate to dep1, then rotate
    return (points - dep1) @ R.T


# ---------------------------------------------------------------------------
# Axis permutation
# ---------------------------------------------------------------------------

def reorient_axes(
    points: np.ndarray,
    x_to: int,
    y_to: int,
    z_to: int,
) -> np.ndarray:
    """Permute the coordinate axes of *points*.

    ``x_to``, ``y_to``, ``z_to`` are the *destination* column indices for the
    current x, y, z axes.  They must form a permutation of ``{0, 1, 2}``.

    Examples:
    - ``(0, 1, 2)`` — identity (no change)
    - ``(2, 1, 0)`` — swap x and z
    - ``(1, 2, 0)`` — cyclic permutation x→1, y→2, z→0

    Mirrors ``RegisterUtil -m orient`` from the C++ CLARCS codebase.

    Parameters
    ----------
    points : ndarray (N, 3)
    x_to   : int in {0, 1, 2} — destination column for the current x axis
    y_to   : int in {0, 1, 2} — destination column for the current y axis
    z_to   : int in {0, 1, 2} — destination column for the current z axis

    Returns
    -------
    ndarray (N, 3) — axis-permuted copy of *points*

    Raises
    ------
    ValueError if (x_to, y_to, z_to) is not a valid permutation of {0, 1, 2}.
    """
    if sorted([x_to, y_to, z_to]) != [0, 1, 2]:
        raise ValueError(
            f"(x_to={x_to}, y_to={y_to}, z_to={z_to}) must be a permutation"
            " of {0, 1, 2}."
        )
    result = np.empty_like(points)
    result[:, x_to] = points[:, 0]
    result[:, y_to] = points[:, 1]
    result[:, z_to] = points[:, 2]
    return result
