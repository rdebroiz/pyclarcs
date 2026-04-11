"""
VTK legacy file I/O using pyvtk.

Provides:
- load_surface  : read a VTK polydata file → (points, polygons)
- save_surface  : write a VTK polydata file from (points, polygons, scalars)
- save_plane_vtk: write a rectangular VTK patch representing the symmetry plane
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

    Parameters
    ----------
    path : str
        Path to the .vtk file.

    Returns
    -------
    points : ndarray of shape (N, 3)
    polygons : list of face index lists (may be empty for pure point clouds)
    """
    data = pyvtk.VtkData.fromfile(path)
    structure = data.structure

    # Extract points
    points = np.array(structure.points, dtype=float)

    # Extract polygon/cell connectivity (if present)
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

    Parameters
    ----------
    path : str
        Output .vtk path.
    points : ndarray (N, 3)
    polygons : list of face index lists, optional
    scalars : ndarray (N,), optional – scalar field per vertex
    scalars_name : str
        Name stored in the VTK header for the scalar array.
    """
    pts_list = points.tolist()
    polys = polygons if polygons is not None else []

    structure = pyvtk.PolyData(points=pts_list, polygons=polys)

    point_data_items = []
    if scalars is not None:
        point_data_items.append(
            pyvtk.Scalars(scalars.tolist(), name=scalars_name, lookup_table="default")
        )

    if point_data_items:
        vtk_data = pyvtk.VtkData(structure, pyvtk.PointData(*point_data_items))
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

    The rectangle is centred on the plane's closest point to the bounding-box
    centre and spans the bounding box, mirroring the surfaceConversion()
    function from CompOnSurface/SurfaceConversion.hh.

    Parameters
    ----------
    path : str
        Output .vtk path.
    plane : SymmetryPlane
    bounds : (xmin, xmax, ymin, ymax, zmin, zmax)
    margin : float
        Scale factor for the rectangle size relative to the bounding box.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    box_centre = np.array(
        [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
    )
    box_diag = np.linalg.norm(
        [xmax - xmin, ymax - ymin, zmax - zmin]
    )
    half = box_diag * margin / 2.0

    # Project bounding-box centre onto the plane
    plane_centre = plane.project(box_centre)

    # Build two orthonormal vectors in the plane
    n = plane.n
    # Choose an arbitrary vector not parallel to n
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(n, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)

    # Four corners of the rectangle
    corners = [
        plane_centre + half * u + half * v,
        plane_centre - half * u + half * v,
        plane_centre - half * u - half * v,
        plane_centre + half * u - half * v,
    ]

    points = [c.tolist() for c in corners]
    polygons = [[0, 1, 2, 3]]

    structure = pyvtk.PolyData(points=points, polygons=polygons)
    pyvtk.VtkData(structure).tofile(path, "ascii")
