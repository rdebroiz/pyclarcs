"""
VTK legacy file I/O using pyvtk.

ROLE IN THE PIPELINE
=====================
The input and output of ZZ_SYMC / pyclarcs are VTK legacy files (.vtk),
the format used throughout the CLARCS C++ library.  The data model is a
*polydata* mesh (triangulated or polygonal surface), which stores:
  - a list of 3-D vertex coordinates (the point cloud);
  - a list of polygonal faces described by vertex index lists (optional).

The CLARCS paper uses surfaces with tens of thousands to hundreds of
thousands of points:
  - skull surfaces from CT scans: ~82 000 – 137 000 points
  - subcortical structures:       ~5 000 – 10 000 points

This module provides three functions:
  - ``load_surface``   : read a VTK file → (points array, face list)
  - ``save_surface``   : write a VTK file from (points, faces, optional scalars)
  - ``save_plane_vtk`` : write a thin rectangular patch representing the plane

VTK LEGACY FORMAT
==================
The legacy VTK ASCII format begins with a five-line header, followed by
POINTS, POLYGONS, and optionally POINT_DATA sections.  The ``pyvtk`` library
transparently handles reading and writing of this format.

SYMMETRY-PLANE VISUALISATION
==============================
The C++ function ``surfaceConversion()`` (CompOnSurface/SurfaceConversion.hh)
constructs a rectangular VTK patch that represents the symmetry plane within
the bounding box of the surface.  ``save_plane_vtk()`` replicates this
behaviour:

  1. The centre of the rectangle is the orthogonal projection of the
     bounding-box centre onto the plane.
  2. Two orthonormal vectors in the plane (u, v) are constructed from
     the cross product of n with a reference direction.
  3. The half-side length is the diagonal of the bounding box (×margin),
     making the patch always larger than the surface it is drawn on.
"""

from __future__ import annotations

import numpy as np
import pyvtk

from pyclarcs._symmetry import SymmetryPlane


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_surface(path: str) -> tuple[np.ndarray, list[list[int]]]:
    """Read a VTK legacy polydata file.

    The function reads the file using ``pyvtk.VtkData.fromfile()``.  Both
    ASCII and binary VTK files are supported by pyvtk.

    For the purposes of the symmetry algorithm only the **point coordinates**
    are strictly needed (the algorithm treats the surface as an unordered
    point cloud).  Face connectivity is preserved so that it can be written
    back when saving the reflected surface.

    Parameters
    ----------
    path : str
        Path to the .vtk file.

    Returns
    -------
    points : ndarray of shape (N, 3), dtype float64
        3-D coordinates of the N surface vertices.
    polygons : list of face index lists (may be empty for pure point clouds)
        Each element is a list of vertex indices describing one polygon.
    """
    data = pyvtk.VtkData.fromfile(path)
    structure = data.structure

    # Vertex coordinates — shape (N, 3)
    points = np.array(structure.points, dtype=float)

    # Face connectivity (not required by the symmetry algorithm but preserved
    # for writing the reflected surface)
    polygons: list[list[int]] = []
    if hasattr(structure, "polygons") and structure.polygons:
        polygons = [list(f) for f in structure.polygons]
    elif hasattr(structure, "cells") and structure.cells:
        polygons = [list(c) for c in structure.cells]

    return points, polygons


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_surface(
    path: str,
    points: np.ndarray,
    polygons: list[list[int]] | None = None,
    scalars: np.ndarray | None = None,
    scalars_name: str = "scalars",
) -> None:
    """Write a VTK legacy polydata file.

    This function is used both to save the *reflected* surface (the symmetric
    image of the input with respect to the estimated plane) and to save any
    intermediate surface with per-vertex scalar data (e.g. asymmetry fields).

    The output is always ASCII format for maximum compatibility.

    Parameters
    ----------
    path : str
        Output .vtk path.
    points : ndarray (N, 3)
        Vertex coordinates.
    polygons : list of face index lists, optional
        If omitted or empty, only the point cloud is written (no faces).
    scalars : ndarray (N,), optional
        Per-vertex scalar field (e.g. asymmetry norm or distance field).
        If provided, written as a POINT_DATA SCALARS block.
    scalars_name : str
        Name tag for the scalar array inside the VTK file (default "scalars").
    """
    pts_list = points.tolist()
    polys = polygons if polygons is not None else []

    structure = pyvtk.PolyData(points=pts_list, polygons=polys)

    if scalars is not None:
        # Attach per-vertex scalar field (e.g. asymmetry magnitude)
        point_data = pyvtk.PointData(
            pyvtk.Scalars(scalars.tolist(), name=scalars_name, lookup_table="default")
        )
        vtk_data = pyvtk.VtkData(structure, point_data)
    else:
        vtk_data = pyvtk.VtkData(structure)

    vtk_data.tofile(path, "ascii")


# ---------------------------------------------------------------------------
# Symmetry-plane visualisation patch
# ---------------------------------------------------------------------------

def save_plane_vtk(
    path: str,
    plane: SymmetryPlane,
    bounds: tuple[float, float, float, float, float, float],
    margin: float = 1.05,
) -> None:
    """Write a rectangular VTK patch representing the symmetry plane.

    The patch is centred on the orthogonal projection of the bounding-box
    centre onto the plane, and is large enough to cover the entire surface.
    This mirrors the ``surfaceConversion(Plane p, ...)`` function from
    ``CompOnSurface/SurfaceConversion.hh`` in the C++ codebase.

    Construction
    ------------
    Let  c   = centre of the bounding box.
    Let  c'  = projection of c onto the plane  (= c − (n·c − d) n).
    Let  diag = length of the bounding-box diagonal.

    Two orthonormal in-plane vectors are built as:
        u = (n × ref) / ||n × ref||     where ref is a vector not parallel to n
        v = n × u

    The four corners of the rectangle are then:
        c' ± half·u  ±  half·v
    with  half = diag × margin / 2.

    Parameters
    ----------
    path : str
        Output .vtk path for the plane patch.
    plane : SymmetryPlane
    bounds : (xmin, xmax, ymin, ymax, zmin, zmax)
        Axis-aligned bounding box of the input surface, used to size and
        centre the rectangular patch.
    margin : float
        Factor by which the patch is enlarged beyond the bounding box diagonal
        (default 1.05 → 5% larger than the diagonal).
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    # Centre of the bounding box
    box_centre = np.array(
        [(xmin + xmax) / 2.0,
         (ymin + ymax) / 2.0,
         (zmin + zmax) / 2.0]
    )

    # Half-size of the rectangle = half the bounding-box diagonal (with margin)
    box_diag = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
    half = box_diag * margin / 2.0

    # Project the bounding-box centre onto the plane:
    #   c' = c − (n·c − d) n
    plane_centre = plane.project(box_centre)

    # Build two orthonormal vectors spanning the symmetry plane.
    # The cross product  n × ref  is guaranteed to lie in the plane
    # (perpendicular to n) and be non-zero as long as ref ∦ n.
    n = plane.n
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(n, ref)) > 0.9:   # n is nearly parallel to x → use y
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref);  u /= np.linalg.norm(u)
    v = np.cross(n, u);    v /= np.linalg.norm(v)

    # Four corners of the rectangular patch (counter-clockwise winding)
    corners = [
        plane_centre + half * u + half * v,   # top-right
        plane_centre - half * u + half * v,   # top-left
        plane_centre - half * u - half * v,   # bottom-left
        plane_centre + half * u - half * v,   # bottom-right
    ]

    points = [c.tolist() for c in corners]
    polygons = [[0, 1, 2, 3]]   # one quad face

    structure = pyvtk.PolyData(points=points, polygons=polygons)
    pyvtk.VtkData(structure).tofile(path, "ascii")
