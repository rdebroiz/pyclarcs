"""
VTK-based surface I/O using the vtk Python package (VTK 9+).

Supported formats (read and write)
====================================
Extension   Format
----------  -------
.vtk        VTK legacy (polydata, ASCII or binary)
.vtp        VTK XML PolyData
.vtu        VTK XML UnstructuredGrid  (read only — converted to polydata)
.ply        Stanford PLY
.stl        STereoLithography
.obj        Wavefront OBJ

The format is inferred from the file extension.  ``load_surface`` always
returns a (points, polygons) pair regardless of the source format.
``save_surface`` always writes a PolyData mesh.

Symmetry-plane visualisation
==============================
``save_plane_vtk`` writes a rectangular quad patch centred on the
orthogonal projection of the surface bounding-box centre onto the plane.
The half-size equals half the bounding-box diagonal (×margin).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from pyclarcs._symmetry import SymmetryPlane


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reader_for(path: str) -> vtk.vtkAlgorithm:
    ext = Path(path).suffix.lower()
    readers: dict[str, type] = {
        ".vtk": vtk.vtkPolyDataReader,
        ".vtp": vtk.vtkXMLPolyDataReader,
        ".vtu": vtk.vtkXMLUnstructuredGridReader,
        ".ply": vtk.vtkPLYReader,
        ".stl": vtk.vtkSTLReader,
        ".obj": vtk.vtkOBJReader,
    }
    if ext not in readers:
        raise ValueError(
            f"Unsupported input format '{ext}'. "
            f"Supported: {', '.join(readers)}"
        )
    reader = readers[ext]()
    reader.SetFileName(path)
    return reader


def _polydata_from_reader(reader: vtk.vtkAlgorithm) -> vtk.vtkPolyData:
    """Run the reader and return a vtkPolyData (converting UG if needed)."""
    reader.Update()
    output = reader.GetOutput()

    # vtkUnstructuredGrid → vtkPolyData
    if isinstance(output, vtk.vtkUnstructuredGrid):
        geom = vtk.vtkGeometryFilter()
        geom.SetInputData(output)
        geom.Update()
        output = geom.GetOutput()

    # Ensure triangles are merged into polygons
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(output)
    clean.Update()
    return clean.GetOutput()


def _polydata_to_arrays(
    poly: vtk.vtkPolyData,
) -> tuple[np.ndarray, list[list[int]]]:
    """Extract (points, polygons) arrays from a vtkPolyData."""
    vtk_pts = poly.GetPoints()
    if vtk_pts is None or vtk_pts.GetNumberOfPoints() == 0:
        raise ValueError("The surface contains no points.")

    points = vtk_to_numpy(vtk_pts.GetData()).astype(float)   # (N, 3)

    polygons: list[list[int]] = []
    cells = poly.GetPolys()
    if cells is not None and cells.GetNumberOfCells() > 0:
        cells.InitTraversal()
        id_list = vtk.vtkIdList()
        while cells.GetNextCell(id_list):
            polygons.append([id_list.GetId(i) for i in range(id_list.GetNumberOfIds())])

    return points, polygons


def _arrays_to_polydata(
    points: np.ndarray,
    polygons: list[list[int]],
) -> vtk.vtkPolyData:
    """Build a vtkPolyData from numpy arrays."""
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(points.astype(np.float32), deep=True))

    cells = vtk.vtkCellArray()
    for face in polygons:
        cells.InsertNextCell(len(face))
        for idx in face:
            cells.InsertCellPoint(idx)

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_pts)
    poly.SetPolys(cells)
    return poly


def _writer_for(path: str, poly: vtk.vtkPolyData) -> vtk.vtkAlgorithm:
    ext = Path(path).suffix.lower()

    if ext == ".vtk":
        w = vtk.vtkPolyDataWriter()
        w.SetFileTypeToASCII()
    elif ext == ".vtp":
        w = vtk.vtkXMLPolyDataWriter()
    elif ext == ".ply":
        w = vtk.vtkPLYWriter()
    elif ext == ".stl":
        w = vtk.vtkSTLWriter()
    elif ext == ".obj":
        w = vtk.vtkOBJWriter()
    else:
        raise ValueError(
            f"Unsupported output format '{ext}'. "
            f"Supported: .vtk, .vtp, .ply, .stl, .obj"
        )

    w.SetFileName(path)
    w.SetInputData(poly)
    return w


# ---------------------------------------------------------------------------
# Public API — loading
# ---------------------------------------------------------------------------

def load_surface(path: str) -> tuple[np.ndarray, list[list[int]]]:
    """Read a surface file into (points, polygons).

    Parameters
    ----------
    path : str
        Path to the surface file (.vtk, .vtp, .vtu, .ply, .stl, .obj).

    Returns
    -------
    points : ndarray (N, 3), float64
        Vertex coordinates.
    polygons : list of face index lists
        Each element is a list of vertex indices for one face.
    """
    reader = _reader_for(path)
    poly = _polydata_from_reader(reader)
    return _polydata_to_arrays(poly)


# ---------------------------------------------------------------------------
# Public API — saving
# ---------------------------------------------------------------------------

def save_surface(
    path: str,
    points: np.ndarray,
    polygons: list[list[int]] | None = None,
    scalars: np.ndarray | None = None,
    scalars_name: str = "scalars",
) -> None:
    """Write a surface to file.

    The output format is determined by the file extension.

    Parameters
    ----------
    path : str
        Output path (.vtk, .vtp, .ply, .stl, .obj).
    points : ndarray (N, 3)
        Vertex coordinates.
    polygons : list of face index lists, optional
        Omit or pass [] for a point-cloud-only file.
    scalars : ndarray (N,), optional
        Per-vertex scalar field (written as POINT_DATA).
        Not supported by all writers (.stl, .obj ignore it).
    scalars_name : str
        Name for the scalar array.
    """
    poly = _arrays_to_polydata(points, polygons or [])

    if scalars is not None:
        vtk_scalars = numpy_to_vtk(scalars.astype(np.float32), deep=True)
        vtk_scalars.SetName(scalars_name)
        poly.GetPointData().SetScalars(vtk_scalars)

    writer = _writer_for(path, poly)
    writer.Write()


# ---------------------------------------------------------------------------
# Public API — symmetry-plane visualisation patch
# ---------------------------------------------------------------------------

def save_plane_vtk(
    path: str,
    plane: SymmetryPlane,
    bounds: tuple[float, float, float, float, float, float],
    margin: float = 1.05,
) -> None:
    """Write a rectangular patch visualising the symmetry plane.

    The patch is centred on the orthogonal projection of the surface
    bounding-box centre onto the plane, and sized to cover the entire surface.

    Parameters
    ----------
    path : str
        Output path. Extension determines format (.vtk, .vtp, .ply, …).
    plane : SymmetryPlane
    bounds : (xmin, xmax, ymin, ymax, zmin, zmax)
    margin : float
        Enlargement factor applied to the bounding-box diagonal (default 1.05).
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    box_centre = np.array([
        (xmin + xmax) / 2.0,
        (ymin + ymax) / 2.0,
        (zmin + zmax) / 2.0,
    ])

    box_diag = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
    half = box_diag * margin / 2.0

    plane_centre = np.asarray(plane.project(box_centre)).flatten()

    n = plane.n
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u = u / float(np.linalg.norm(u))
    v = np.cross(n, u)
    v = v / float(np.linalg.norm(v))

    corners = np.array([
        plane_centre + half * u + half * v,
        plane_centre - half * u + half * v,
        plane_centre - half * u - half * v,
        plane_centre + half * u - half * v,
    ], dtype=float)

    save_surface(path, corners, [[0, 1, 2, 3]])
